from TAKE.PositionalEmbedding import PositionalEmbedding
from TAKE.TransformerSeqEncoderDecoder import *
from TAKE.GPT2summ import *
from gpt2Dataset import gpt2Dataset
from TAKE.Utils import neginf
from TAKE.Utils import _generate_square_subsequent_mask
from TAKE.Utils import build_map
from TAKE.Utils import to_sentence
from TAKE.Utils import sequence_loss
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
import math



class GPT2_gen(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.dec = GPT2Summ(self.tokenizer, segment=self.args.segment)

    def do_train(self, data):
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
        gen_criterion = lambda logits, targets: sequence_loss(logits, targets, ce, pad_idx=-1)
        lm_input = data['input_ids']
        batch_size = lm_input.size(0)
        episode_length = lm_input.size(1)
        lm_input = lm_input.reshape(batch_size*episode_length,-1)

        token_type_ids = data['token_type_ids'].reshape(batch_size*episode_length,-1)
        lm_target = data['targets'].reshape(batch_size*episode_length, -1)


        rg = self.dec(lm_input)

        loss_g = gen_criterion(rg[0], lm_target).mean()
        
        return loss_g


    def do_inference(self, data):
        dec_in = data['input_ids']
        batch_size = dec_in.size(0)
        episode_length = dec_in.size(1)
        dec_in = dec_in.reshape(batch_size*episode_length,-1)
        dec_out = self.dec.batch_decode(dec_in, self.args.max_dec_length, self.tokenizer.eos_token_id, self.dec.user_id[1]) 
        
        dec_out = dec_out[0].tolist()[dec_in.size(1):]
        
        _hyp = self.tokenizer.decode(dec_out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return _hyp

    def forward(self, data, method='train'):
        if method == 'train':
            return self.do_train(data)
        elif method == 'inference':
            return self.do_inference(data)   


