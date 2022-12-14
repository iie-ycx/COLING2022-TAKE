from TAKE.PositionalEmbedding import PositionalEmbedding
from TAKE.TransformerEncoder import *
from TAKE.TransformerDecoder import *
from TAKE.Utils import _generate_square_subsequent_mask
from TAKE.Utils import universal_sentence_embedding
from TAKE.Utils import new_tensor
from TAKE.Utils import topk
from TAKE.Constants import *
import numpy as np
from transformers import BertModel


class TransformerSeqEncoder(nn.Module):
    def __init__(self, args, vocab2id, id2vocab, emb_matrix=None):
        super(TransformerSeqEncoder, self).__init__()
        self.args = args
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab
        self.enc = BertModel.from_pretrained('bert-base-uncased')

        self.Posit_SEP = args.context_len-1

    def forward(self, batch_numseq_seqlen, mask):
        # context and mask [batch * max_episode_length, 1, context_len]
        # knowledge_token and mask [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]

        batch_size, num_seq, seq_len = batch_numseq_seqlen.size()
        batch_numseq_seqlen = batch_numseq_seqlen.reshape(-1, seq_len)  # [batch_size * num_seq, seq_len]
        mask = mask.reshape(-1, seq_len)  # [batch_size * num_seq, seq_len]

        output = self.enc(batch_numseq_seqlen, attention_mask=mask.float())[0]# [batch_size*num_seq, seq_len, embedding_size]
        

        CLS = output[:,0,:].reshape(batch_size, num_seq, -1)
        output_c = output[:,1:self.Posit_SEP-1,:]
        output_k = output[:,self.Posit_SEP+1:,:]
        # [batch_size * num_seq, embedding_size]
        pooling = universal_sentence_embedding(output, mask)
        pooling /= np.sqrt(pooling.size()[-1])

        mask_c = mask[:,1:self.Posit_SEP-1]
        mask_k = mask[:,self.Posit_SEP+1:]
        pooling_before_SEP = universal_sentence_embedding(output_c, mask_c)
        pooling_after_SEP = universal_sentence_embedding(output_k, mask_k)

        # [batch_size, num_seq, seq_len, embedding_size]
        output = output.reshape(batch_size, num_seq, seq_len, -1)

        # [batch_size, num_seq, embedding_size]
        pooling = pooling.reshape(batch_size, num_seq, -1)
        # [batch_size, num_seq, embedding_size]
        pooling_before_SEP = pooling_before_SEP.reshape(batch_size, num_seq, -1)
        pooling_before_SEP /= np.sqrt(pooling_before_SEP.size()[-1])
        # [batch_size, num_seq, embedding_size]
        pooling_after_SEP = pooling_after_SEP.reshape(batch_size, num_seq, -1)
        pooling_after_SEP /= np.sqrt(pooling_after_SEP.size()[-1])

        # [batch_size, num_seq, seq_len, embedding_size]  
        # [batch_size, num_seq, embedding_size]
        return output, pooling, CLS, pooling_before_SEP, pooling_after_SEP


class TransformerSeqDecoder(nn.Module):
    def __init__(self, args, vocab2id, id2vocab, emb_matrix=None):
        super(TransformerSeqDecoder, self).__init__()
        self.args = args
        self.tgt_vocab_size = len(vocab2id)
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab

        if emb_matrix is not None:
            self.embedding = nn.Sequential(emb_matrix, PositionalEmbedding(self.args.hidden_size, dropout=0.1, max_len=1000))
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )

        decoder_layer = TransformerDecoderLayer(self.args.hidden_size, nhead=self.args.n_heads, dim_feedforward=self.args.ffn_size, dropout=self.args.dropout, activation='gelu')
        self.dec = TransformerDecoder(decoder_layer, num_layers=self.args.n_layers, norm=None)

        self.norm1 = LayerNorm(self.args.hidden_size)
        self.norm2 = LayerNorm(self.args.hidden_size)

        self.attns = nn.ModuleList([BilinearAttention(self.args.hidden_size, self.args.hidden_size, self.args.hidden_size) for i in range(2)])

        self.gen = nn.Sequential(nn.Linear(2*self.args.hidden_size, self.args.hidden_size), nn.Dropout(0.1), nn.Linear(self.args.hidden_size, self.tgt_vocab_size, bias=False), nn.Softmax(dim=-1))

        self.mix = nn.Linear(3*self.args.hidden_size, 3)

    def extend(self, dec_outputs, gen_outputs, memory_weights, source_map):
        # dec_outputs [batch_size, max_target_length???3*hidden_size]
        # gen_outputs [batch_size, max_target_length???tgt_vocab_size]
        # memory_weights [[batch_size, max_target_length???seq_len_q], [batch_size, max_target_length???num_sequences*seq_len_p]]
        # source_map [batch, context_len+knowledge_len, vocab_size]

        # [batch_size, max_target_length???3]
        switcher = F.softmax(self.mix(dec_outputs), dim=-1)

        # p[:, :, 0].unsqueeze(-1)->[batch_size, max_target_length, 1]*[batch_size, max_target_length???tgt_vocab_size]-->[batch_size, max_target_length???tgt_vocab_size]
        dist1 = switcher[:, :, 0].unsqueeze(-1)*gen_outputs
        # [[batch_size, max_target_length???seq_len_q],[batch_size, max_target_length???num_sequences*seq_len_p]]
        # [batch_size, max_target_length???context_len+knowledge_len]
        dist2 = torch.cat([switcher[:, :, i+1].unsqueeze(-1)*memory_weights[i] for i in range(len(memory_weights))], dim=-1)
        # [batch_size, max_target_length???context_len+knowledge_len] ?????? [batch, context_len+knowledge_len, vocab_size]-->[batch_size, max_target_length???tgt_vocab_size]
        dist2 = torch.bmm(dist2, source_map)

        return dist1 + dist2

    def forward(self, encode_memories, encode_masks=None, groundtruth_index=None, source_map = None):
        '''
        input:
        encode_memories [query_rep[0], passage_rep[0]]: [batch_size, seq_len_q, hidden_size], [batch_size, seq_len_p, hidden_size]
        source_map: [batch, context_len+knowledge_len, vocab_size]
        additional_decoder_feature [batch_size, hidden_size]
        groundtruth_index=output [batch_size, max_target_length]
        encode_masks: [batch_size, 1, seq_len_q], [batch_size, num_sequences, seq_len_p]
        encode_weights_[prior_q, prior_p]: [batch_size, 1, seq_len_q] [batch_size, num_sequences, seq_len_p]
        '''
        batch_size = source_map.size(0)
        # [[batch_size, context_len, hidden_size], [batch_size, knowledge_len, hidden_size]]
        encode_memories = [encode.reshape(batch_size, -1, self.args.hidden_size) for encode in encode_memories]
        # [[batch_size, context_len], [batch_size, knowledge_len]]
        encode_masks = [mask.reshape(batch_size, -1) for mask in encode_masks]

        # [batch_size, 1]
        bos = new_tensor([self.vocab2id[BOS_WORD]]*batch_size, requires_grad=False).unsqueeze(1)

        if self.training and groundtruth_index is not None:
            # [batch_size, max_target_length]
            dec_input_index = torch.cat([bos, groundtruth_index[:, :-1]], dim=-1)
            # [batch_size, max_target_length???embedding_size=hidden_size]
            dec_input = self.embedding(dec_input_index)

            #[max_target_length???batch_size, embedding_size=hidden_size]
            dec_outputs = dec_input.transpose(0, 1)
            memory_attns=[] #[batch_size, max_target_length???seq_len_q/num_sequences*seq_len_p]
            c_m = [] #  [batch_size, max_target_length???hidden_size]

            dec_outputs, _, _, _ = self.dec(dec_outputs, encode_memories[0].transpose(0, 1), encode_memories[1].transpose(0, 1),
                                          tgt_mask=_generate_square_subsequent_mask(dec_outputs.size(0)),
                                          tgt_key_padding_mask=~dec_input_index.ne(0),
                                          memory_key_padding_mask1=~encode_masks[0],
                                          memory_key_padding_mask2=~encode_masks[1])


            for i in range(2):
                m_i, _, m_i_weights = self.attns[i](dec_outputs.transpose(0, 1), encode_memories[i],
                    encode_memories[i], mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(),
                                                       encode_masks[i].unsqueeze(1).float()).bool())

                c_m.append(m_i)  # [batch_size, max_target_length???hidden_size]
                memory_attns.append(m_i_weights)  # [batch_size, max_target_length???context_len/knowledge_len]

            # [batch_size, max_target_length???hidden_size]
            dec_outputs = self.norm1(dec_outputs).transpose(0, 1)
            # vector-->[batch_size, max_target_length???2*hidden_size]-->[batch_size, max_target_length???2*hidden_size]
            # [batch_size, max_target_length???tgt_vocab_size]
            gen_outputs = self.gen(torch.cat([dec_input, dec_outputs], dim=-1))

            # torch.cat([dec_outputs]+c_m, dim=-1)->[batch_size, max_target_length???3*hidden_size]
            # ([batch_size, max_target_length???tgt_vocab_size],[batch_size, max_target_length???tgt_vocab_size])
            # source_map [batch, context_len+knowledge_len, vocab_size]
            extended_gen_outputs = self.extend(torch.cat([dec_outputs]+c_m, dim=-1), gen_outputs, memory_attns, source_map)  # [batch * max_episode_length, max_target_length, tgt_vocab_size]
            # [batch_size, max_target_length]
            output_indexes = groundtruth_index

        elif not self.training:
            input_indexes = []
            output_indexes = []
            for t in range(self.args.max_dec_length):
                # [batch_size, 1+len(input_indexes)]
                dec_input_index = torch.cat([bos] + input_indexes, dim=-1)
                # [batch_size, 1+len(input_indexes), hidden_size]
                dec_input = self.embedding(dec_input_index)

                # [1+len(input_indexes), batch_size, hidden_size]
                dec_outputs = dec_input.transpose(0, 1)
                memory_attns = []
                c_m=[]

                # [1+len(input_indexes)???batch_size, hidden_size]
                dec_outputs, _, _ , _ = self.dec(dec_outputs, encode_memories[0].transpose(0, 1),
                                             encode_memories[1].transpose(0, 1),
                                             tgt_mask=_generate_square_subsequent_mask(dec_outputs.size(0)),
                                             tgt_key_padding_mask=~dec_input_index.ne(0),
                                             memory_key_padding_mask1=~encode_masks[0],
                                             memory_key_padding_mask2=~encode_masks[1])

                for i in range(2):
                    # m_i  [batch_size, 1+len(input_indexes)???hidden_size]
                    # m_i_weights  [batch_size, 1+len(input_indexes)???seq_len_q/num_sequences*seq_len_p]
                    m_i, _, m_i_weights = self.attns[i](dec_outputs.transpose(0, 1), encode_memories[i],
                                                        encode_memories[i],
                                                        mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(),
                                                                       encode_masks[i].unsqueeze(1).float()).bool())
                    c_m.append(m_i) # [[batch_size, 1+len(input_indexes)???hidden_size],[batch_size, 1+len(input_indexes)???hidden_size]]
                    memory_attns.append(m_i_weights) # [[batch_size, 1+len(input_indexes)???seq_len_q],[batch_size, 1+len(input_indexes)???num_sequences*seq_len_p]]

                # [batch_size, 1+len(input_indexes), hidden_size]
                dec_outputs = self.norm1(dec_outputs).transpose(0, 1)
                # [batch_size, 1+len(input_indexes)???tgt_vocab_size]
                gen_outputs = self.gen(torch.cat([dec_input, dec_outputs], dim=-1))

                # [batch_size, 1+len(input_indexes)???tgt_vocab_size]
                extended_gen_outputs = self.extend(torch.cat([dec_outputs]+c_m, dim=-1), gen_outputs, memory_attns, source_map)

                # extended_gen_outputs[:, -1]-->[batch_size, tgt_vocab_size]
                # probs (batch,1)
                # indices (batch,1)
                probs, indices = topk(extended_gen_outputs[:, -1], k=1)

                input_indexes.append(indices)
                output_indexes.append(indices)

            # [batch, max_target_length]
            output_indexes=torch.cat(output_indexes, dim=-1)

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indexes