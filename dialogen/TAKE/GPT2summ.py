import torch
from torch import nn

from transformers import GPT2PreTrainedModel, GPT2Model, AutoModel, GPT2Config, GPT2LMHeadModel


class GPT2Summ(GPT2PreTrainedModel):
    '''succeed from GPT2PreTraninedModel which has implemented the 'generate' func'''

    def __init__(self, tokenizer, segment=None):
        config = GPT2Config.from_pretrained('/data/ycx/pretrain-models/gpt2/')
        super(GPT2Summ, self).__init__(config)
        self.transformer = GPT2LMHeadModel.from_pretrained('/data/ycx/pretrain-models/gpt2/')
        
        self.prompt = [tokenizer.convert_tokens_to_ids('<context>'),
                        tokenizer.convert_tokens_to_ids('<response>')]
        self.user_id = [tokenizer.convert_tokens_to_ids('<user>'), 
                        tokenizer.convert_tokens_to_ids('<system>')]
        self.know_id = tokenizer.convert_tokens_to_ids('<knowledge>')
        
        self.segment = segment
        self.config.vocab_size = len(tokenizer)
        self.transformer.resize_token_embeddings(len(tokenizer))

        #self.lm_head = nn.Linear(config.n_embd, len(tokenizer), bias=False)
        
        self.tie_weights()


    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        
        token_type_ids = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            type_ids = []
            last_special_token = 0
            for j in range(len(ids)):
                if ids[j] in ([self.know_id] + self.prompt):
                    type_ids.append(ids[j])
                    last_special_token = ids[j]
                else:
                    type_ids.append(last_special_token)
            token_type_ids.append(type_ids)
        token_type_ids = torch.tensor(token_type_ids).type_as(input_ids)

        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)


        if self.segment:
            inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        else:
            inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        
        return inputs


    # def forward(self, input_ids, past=None):
    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, use_cache=None, return_dict=None, output_attentions=None, output_hidden_states=None):
        
        transformer_outputs = self.transformer(input_ids, past_key_values=past, token_type_ids=token_type_ids)

        return transformer_outputs

    def batch_decode(self, input_ids, max_len, eos_id, start_id):
        """ greedy decode support batching"""

        output_sequences = self.generate(
            input_ids=input_ids,
            max_length=input_ids.size(1) + max_len,
            do_sample=False,
            early_stopping=True,
            num_beams=1,
            repetition_penalty=1,
            bos_token_id=start_id,
            pad_token_id=0,
            eos_token_id=eos_id,
            length_penalty=1,
            no_repeat_ngram_size=0,
            # decoder_start_token_id=start_id, #
            #use_cache = False
        )
        return output_sequences
