from TAKE.PositionalEmbedding import PositionalEmbedding
from TAKE.TransformerSeqEncoderDecoder import *
from TAKE.Utils import neginf
from TAKE.Utils import _generate_square_subsequent_mask
from TAKE.Utils import build_map
from TAKE.Utils import to_sentence
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm

class DivideAndSelfAttention(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.wc = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.vc = nn.Parameter(torch.Tensor(args.hidden_size, 1))

        self.wk = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.vk = nn.Parameter(torch.Tensor(args.hidden_size, 1))

    def forward(self, context_knowledge, knowledge_piece_mask, episode_mask, mode="train"):
        # context_knowledge [batch * max_episode_length, max_knowledge_pool + context_len, hidden_size]
        # knowledge_piece_mask [batch, max_episode_length, max_knowledge_pool]
        # episode_mask [batch, max_episode_length]
        knowledge_piece_mask = knowledge_piece_mask.reshape(-1, knowledge_piece_mask.size()[-1])  # [batch * max_episode_length, max_knowledge_pool]
        context_pooling_ex = context_knowledge[3]  #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        knowledge_pooling = context_knowledge[4] # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        
        context_attn = torch.tanh(torch.matmul(context_pooling_ex, self.wc.unsqueeze(0)))  #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        context_score = torch.matmul(context_attn, self.vc.unsqueeze(0)).squeeze(-1) #[batch * max_episode_length, max_knowledge_pool]
        
        context_score.masked_fill_(~knowledge_piece_mask, neginf(context_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        context_dist = F.softmax(context_score, 1)  # [batch * max_episode_length, max_knowledge_pool]
        context_vector = torch.bmm(context_dist.unsqueeze(1), context_attn).squeeze(1) # [batch * max_episode_length, hidden_size]

        knowledge_attn = torch.tanh(torch.matmul(knowledge_pooling, self.wk.unsqueeze(0)))  #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        between_know_score = torch.matmul(knowledge_attn, self.vk.unsqueeze(0)).squeeze(-1) #[batch * max_episode_length, max_knowledge_pool]
        
        between_know_score.masked_fill_(~knowledge_piece_mask, neginf(between_know_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        between_know_dist = F.softmax(between_know_score, 1)  # [batch * max_episode_length, max_knowledge_pool]
        between_know_vector = torch.bmm(between_know_dist.unsqueeze(1), knowledge_attn).squeeze(1) # [batch * max_episode_length, hidden_size]

        return context_vector, between_know_vector


class TopicShiftedSelector(nn.Module): #topic shift selector
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        #feature extractor
        self.project_c = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_CLS = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_b = nn.Linear(args.hidden_size, args.hidden_size)
        #mapping
        self.project_q = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size*2, args.hidden_size)
        #additive attention
        self.attn_w = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.attn_u = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.attn_v = nn.Parameter(torch.Tensor(args.hidden_size, 1))


    def forward(self, context_vector, between_know_vector, context_knowledge, knowledge_piece_mask, episode_mask, mode="train"):
        # context_pooling [batch * max_episode_length, hidden_size])
        # knowledge_pooling [batch * max_episode_length, max_knowledge_pool, hidden_size]
        # knowledge_piece_mask [batch, max_episode_length, max_knowledge_pool]
        # episode_mask [batch, max_episode_length]

        knowledge_piece_mask = knowledge_piece_mask.reshape(-1, knowledge_piece_mask.size()[-1])  # [batch * max_episode_length, max_knowledge_pool]
        
        #construct query
        between_know_pro = nn.ReLU()(self.project_b(between_know_vector))
        context_vector_pro = nn.ReLU()(self.project_c(context_vector))
        
        query = torch.cat([context_vector_pro, between_know_pro], dim=-1)
        query_pro = self.project_q(query)  # [batch * max_episode_length, hidden_size]

        #construct key
        CLS = context_knowledge[2]  # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        CLS_pro = nn.ReLU()(self.project_CLS(CLS))# [batch * max_episode_length, max_knowledge_pool, hidden_size]

        knowledge_pooling = context_knowledge[4] # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        
        knowledge_fusion = torch.cat([CLS_pro, knowledge_pooling], dim=-1)
        knowledge_pooling_pro_k = self.project_k(knowledge_fusion) # [batch * max_episode_length, max_knowledge_pool, hidden_size]

        #do attention
        know_attn = torch.matmul(knowledge_pooling_pro_k, self.attn_w.unsqueeze(0))  #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        query_attn = torch.matmul(query_pro, self.attn_u).unsqueeze(1)      #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        knowledge_score = torch.matmul(torch.tanh(know_attn + query_attn), self.attn_v.unsqueeze(0)).squeeze(-1)  #[batch * max_episode_length, max_knowledge_pool]

        knowledge_score.masked_fill_(~knowledge_piece_mask, neginf(knowledge_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        knowledge_dist = F.softmax(knowledge_score, 1)  # [batch * max_episode_length, max_knowledge_pool]

        # knowledge_dist [batch * max_episode_length, max_knowledge_pool]
        return knowledge_dist, knowledge_score


class TopicInheritedSelector(nn.Module): #topic inheritance selector
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)
        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)
        
        #feature extractor
        self.project_b = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_CLS = nn.Linear(args.hidden_size, args.hidden_size)
        #mapping
        self.project_q = nn.Linear(args.hidden_size*2, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size*2, args.hidden_size)
        #additive attention
        self.attn_w = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.attn_u = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.attn_v = nn.Parameter(torch.Tensor(args.hidden_size, 1))

    def forward(self, between_know_vector, knowledge_pooling_use, context_knowledge, knowledge_piece_mask, episode_mask, mode="train"):
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # knowledge_pooling [batch * max_episode_length, max_knowledge_pool, hidden_size]
        # knowledge_piece_mask  [batch, max_episode_length, max_knowledge_pool]
        # episode_mask [batch, max_episode_length]

        batch_size, max_episode_length = episode_mask.size()
        knowledge_piece_mask = knowledge_piece_mask.reshape(-1, knowledge_piece_mask.size()[-1])  # [batch * max_episode_length, max_knowledge_pool]

        #construct query
        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use[:, :(max_episode_length-1)]], dim=1) # [batch, max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(max_episode_length)).transpose(0, 1).reshape(batch_size * max_episode_length, -1)
        between_know_pro = nn.ReLU()(self.project_b(between_know_vector))

        query = torch.cat([state, between_know_pro], dim=-1)
        query_pro = self.project_q(query)  # [batch * max_episode_length, hidden_size]

        #construct key
        knowledge_pooling = context_knowledge[4]

        CLS = context_knowledge[2]
        CLS_pro = nn.ReLU()(self.project_CLS(CLS))

        knowledge_fusion = torch.cat([CLS_pro, knowledge_pooling], dim=-1)
        knowledge_pooling_pro_k = self.project_k(knowledge_fusion)  # [batch * max_episode_length, max_knowledge_pool, hidden_size]


        #do attention
        know_attn = torch.matmul(knowledge_pooling_pro_k, self.attn_w.unsqueeze(0))  #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        query_attn = torch.matmul(query_pro, self.attn_u).unsqueeze(1)      #[batch * max_episode_length, max_knowledge_pool, hidden_size]
        knowledge_score = torch.matmul(torch.tanh(know_attn + query_attn), self.attn_v.unsqueeze(0)).squeeze(-1)  #[batch * max_episode_length, max_knowledge_pool]

        knowledge_score.masked_fill_(~knowledge_piece_mask, neginf(knowledge_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        knowledge_dist = F.softmax(knowledge_score, 1)  # [batch * max_episode_length, max_knowledge_pool]

        # knowledge_dist [batch * max_episode_length, max_knowledge_pool]
        return knowledge_dist, knowledge_score


class TeacherTopicShiftDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)

        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)

        self.project_c = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_m = nn.Linear(args.hidden_size * 5, 2)

    def forward(self, context_pooling, knowledge_pooling_use, episode_mask,  mode="train"):
        # context_pooling [batch * max_episode_length, hidden_size]
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # episode_mask [batch, max_episode_length]

        batch_size, max_episode_length = episode_mask.size()

        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use[:, :(max_episode_length - 1)]], dim=1)  # [batch, max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]

        context_pooling_pro = nn.ReLU()(self.project_c(context_pooling))
        # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(input_embedding.size(1))).transpose(0, 1)

        state = state.reshape(batch_size*max_episode_length, -1)  # [batch * max_episode_length, hidden_size]

        gold_knowledge = knowledge_pooling_use.reshape(batch_size*max_episode_length, -1)
        
        fusion = torch.cat([gold_knowledge, context_pooling_pro, state, context_pooling_pro-state, context_pooling_pro*state], dim=1)  # [batch * max_episode_length, 2 * hidden_size]
        fusion_pro = self.project_m(fusion)
  
        shift_prob = F.softmax(fusion_pro, 1) # [batch * max_episode_length, 2]
        #topic shift probability
        
        return shift_prob, fusion_pro



class StudentTopicShiftDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)


        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)

        self.project_c = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_f = nn.Linear(4 * args.hidden_size, 2)

    def forward(self, context_pooling, knowledge_pooling_use, episode_mask, mode="train"):
        # context_pooling [batch * max_episode_length, hidden_size]
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # episode_mask [batch, max_episode_length]

        batch_size, max_episode_length = episode_mask.size()

        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use[:, :(max_episode_length - 1)]], dim=1)  # [batch, max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]


        # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(input_embedding.size(1))).transpose(0, 1)
        state = state.reshape(batch_size*max_episode_length, -1)  # [batch * max_episode_length, hidden_size]
        
        context_pooling_pro = nn.ReLU()(self.project_c(context_pooling))

        fusion = torch.cat([context_pooling_pro, state, context_pooling_pro-state, context_pooling_pro*state], dim=1)  # [batch * max_episode_length, 2 * hidden_size]
        fusion_pro = self.project_f(fusion) # [batch * max_episode_length, 2]
        shift_prob = F.softmax(fusion_pro, 1) # [batch * max_episode_length, 2]

        return shift_prob, fusion_pro


class TAKE(nn.Module):
    def __init__(self, vocab2id, id2vocab, args):
        super().__init__()

        self.id2vocab = id2vocab
        self.vocab_size = len(id2vocab)
        self.args = args

        self.enc = TransformerSeqEncoder(args, vocab2id, id2vocab, None)  # num_layers, num_heads, src_vocab_size, hidden_size, emb_matrix=None
        self.div = DivideAndSelfAttention(args=args)
        self.topic_shifted_selector = TopicShiftedSelector(args=args)
        self.topic_inherited_selector = TopicInheritedSelector(args=args)
        self.teacher_topic_shift_discriminator = TeacherTopicShiftDiscriminator(args=args)
        self.student_topic_shift_discriminator = StudentTopicShiftDiscriminator(args=args)

    def encoding_layer(self, data):
        context = data['context']  # [batch, max_episode_length, context_len]
        knowledge_token = data['knowledge_pool']   # [batch, max_episode_length, max_knowledge_pool, knowledge_sentence_len]

        context = context.reshape(-1, context.size()[-1])  # [batch * max_episode_length, context_len]
        knowledge_token = knowledge_token.reshape(-1, knowledge_token.size()[-2], knowledge_token.size()[-1]) # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]

        knowledge_token_mask = knowledge_token.ne(0).detach()  # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]
        knowledge_pool_encoded = self.enc(knowledge_token, knowledge_token_mask)

        return {
                'knowledge_pool_encoded': knowledge_pool_encoded, # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len, hidden_size]
                "knowledge_token_mask": knowledge_token_mask}  # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]

    def mixed_initiative_knowledge_selection_layer(self, data, encoded_state, epoch=0):
        knowledge_piece_mask = data["knowledge_piece_mask"]  # [batch, max_episode_length, max_knowledge_pool]
        knowledge_label = data["knowledge_label"]  # [batch, max_episode_length]
        ID_label = data['Initiative_label'].reshape(-1)
        batch_size, max_episode_length = knowledge_label.size()

        # get history knowledge
        knowledge_label = knowledge_label.reshape(-1)  # [batch * max_episode_length]
        knowledge_pooling = encoded_state['knowledge_pool_encoded'][4]  # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        batch_size_max_episode_length, max_knowledge_pool, hidden_size = knowledge_pooling.size()
        offsets = torch.arange(batch_size_max_episode_length, device=knowledge_label.device) * max_knowledge_pool + knowledge_label # [batch * max_episode_length]
        flatten_knowledge_pooling = knowledge_pooling.reshape(batch_size_max_episode_length * max_knowledge_pool, -1) # [batch * max_episode_length * max_knowledge_pool, hidden_size]
        knowledge_pooling_use = flatten_knowledge_pooling[offsets]  # [batch * max_episode_length, hidden_size]
        knowledge_pooling_use = knowledge_pooling_use.reshape(batch_size, max_episode_length, hidden_size) # [batch, max_episode_length, hidden_size]
        #gold knowledge
        
        context_vector, between_know_vector = self.div(encoded_state['knowledge_pool_encoded'], knowledge_piece_mask, data['episode_mask'], self.args.mode)

        topic_shifted_dist, topic_shifted_score = self.topic_shifted_selector(context_vector, between_know_vector, encoded_state['knowledge_pool_encoded'], knowledge_piece_mask, data['episode_mask'], self.args.mode)
        topic_inherited_dist, topic_inherited_score = self.topic_inherited_selector(between_know_vector, knowledge_pooling_use, encoded_state['knowledge_pool_encoded'], knowledge_piece_mask, data['episode_mask'], self.args.mode)

        if self.args.mode == "inference":
            s_shift_prob, s_state_pro = self.student_topic_shift_discriminator(context_vector, knowledge_pooling_use, data['episode_mask'], self.args.mode)
            dis = s_shift_prob.max(1)[1]

        elif self.args.mode == "train":
            t_shift_prob, t_state_pro = self.teacher_topic_shift_discriminator(context_vector, knowledge_pooling_use, data['episode_mask'], self.args.mode)
            s_shift_prob, s_state_pro = self.student_topic_shift_discriminator(context_vector, knowledge_pooling_use, data['episode_mask'], self.args.mode)
            
            dis = ID_label  # [batch * max_episode_length]

            if epoch > self.args.switch_ID:
                init_ratio = 1.0
                curr_ratio = max(init_ratio * (1 - self.args.anneal_rate * (epoch - self.args.switch_ID)), self.args.min_ratio)
                dis_selected = s_shift_prob.max(1)[1]
                for i in range(len(dis)):
                    coin = np.random.choice(2, 1, p=[curr_ratio, 1-curr_ratio])[0]
                    if coin == 0:
                        pass
                    else:
                        dis[i] = dis_selected[i]


        h_shift_prob = F.one_hot(dis, 2) #hard_s_shift_prob  [batch * max_episode_length, 2]
        shift_prob = h_shift_prob[:, 0].reshape(-1, 1) 
        inherit_prob = h_shift_prob[:, 1].reshape(-1, 1)

        final_dist = torch.mul(topic_shifted_dist, shift_prob.expand_as(topic_shifted_dist)) + torch.mul(topic_inherited_dist, inherit_prob.expand_as(topic_inherited_dist))
        

        if self.args.mode == "inference":
            return {
                    'final_dist': final_dist,
                    's_shift_prob': s_shift_prob,
                    'topic_shifted_dist': topic_shifted_dist,
                    'topic_inherited_dist': topic_inherited_dist}

        elif self.args.mode == "train":
            return {
                    'final_dist': final_dist,
                    'topic_shifted_dist': topic_shifted_dist,
                    'topic_inherited_dist': topic_inherited_dist,
                    "t_shift_prob": t_shift_prob,
                    "t_state_pro": t_state_pro,
                    "s_shift_prob": s_shift_prob,
                    "s_state_pro": s_state_pro,
                    "shift_prob": shift_prob,
                    }

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def do_train(self, data, epoch):
        encoded_state = self.encoding_layer(data)
        memory = self.mixed_initiative_knowledge_selection_layer(data, encoded_state, epoch)

        _, final_ks_pred = memory['final_dist'].detach().max(1)  # [batch * max_episode_length]
        _, shifted_ks_pred = memory['topic_shifted_dist'].detach().max(1)  # [batch * max_episode_length]
        _, inherited_ks_pred = memory['topic_inherited_dist'].detach().max(1)  # [batch * max_episode_length]
        _, ID_pred = memory['s_shift_prob'].detach().max(1)  # [batch * max_episode_length]
        
        final_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), final_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        shifted_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), shifted_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        inherited_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), inherited_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        ID_acc = accuracy_score(data['Initiative_label'].reshape(-1).cpu(), ID_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        
        # Initiative dis loss
        masked_Initiative_label = data['Initiative_label'].reshape(-1).masked_fill(~data['episode_mask'].reshape(-1), -1) # [batch * max_episode_length]
        loss_ID = F.nll_loss((memory['s_shift_prob'] + 1e-8).log(), masked_Initiative_label, ignore_index=-1)

        # Knowlegde selection loss
        masked_knowledge_label = data['knowledge_label'].reshape(-1).masked_fill(~data['episode_mask'].reshape(-1), -1) # [batch * max_episode_length]
        loss_ks = F.nll_loss((memory['final_dist'] + 1e-8).log(), masked_knowledge_label, ignore_index=-1)

        # distill Loss
        loss_ID_t = F.nll_loss((memory['t_shift_prob'] + 1e-8).log(), masked_Initiative_label, ignore_index=-1)
        
        input = F.log_softmax(memory['s_state_pro'].squeeze(-1), 1)
        
        KLloss = nn.KLDivLoss(reduction="none", log_target=False)
        loss_KL = KLloss(input, memory['t_shift_prob'].squeeze(-1).detach()).sum(1)  # [batch * max_episode_length, 1]
        
        KL_mask = data['episode_mask'].reshape(-1).float()  # [batch * max_episode_length]
        loss_KL = torch.sum(loss_KL * KL_mask)/torch.sum(KL_mask)
        loss_distill = loss_ID + loss_KL

        return loss_ks, loss_distill, final_ks_acc, shifted_ks_acc, inherited_ks_acc, memory['s_shift_prob'].mean(), loss_ID_t, ID_acc


    def do_inference(self, data):
        encoded_state = self.encoding_layer(data)
        memory = self.mixed_initiative_knowledge_selection_layer(data, encoded_state)

        _, final_ks_pred = memory['final_dist'].max(1)  # [batch * max_episode_length]
        _, shifted_ks_pred = memory['topic_shifted_dist'].detach().max(1)  # [batch * max_episode_length]
        _, inherited_ks_pred = memory['topic_inherited_dist'].detach().max(1)  # [batch * max_episode_length]

        _, ID_pred = memory['s_shift_prob'].detach().max(1)  # [batch * max_episode_length]

        return final_ks_pred, shifted_ks_pred, inherited_ks_pred, ID_pred

    def forward(self, data, method='train', epoch=0):
        if method == 'train':
            return self.do_train(data, epoch)
        elif method == 'inference':
            return self.do_inference(data)

