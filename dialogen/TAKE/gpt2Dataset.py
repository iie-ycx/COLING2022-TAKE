from torch.utils.data import Dataset
from TAKE.Utils import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config, GPT2Tokenizer
import torch

class gpt2Dataset(Dataset):
    def __init__(self, mode, episodes, query, passage, pred, segment=True, max_episode_length=None, knowledge_truncate=64, text_truncate=128, block_size=256, epoch=0, args=None):  # 1e10=1E10
        super(Dataset, self).__init__()
        self.args = args
        self.mode = mode
        self.max_episode_length = max_episode_length
        self.epoch =epoch
        #原数据
        self.episodes = episodes
        self.query = query
        self.passage = passage
        self.pred = pred
        self.segment = segment
        self.answer_file = './datasets/wizard_of_wikipedia/wizard_of_wikipedia.answer'

        self.episode_tensor = []

        #len
        self.knowledge_truncate = knowledge_truncate
        self.text_truncate = text_truncate
        self.block_size = block_size


        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #先尝试不做pad，pad可定义新token
        SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<context>", "<response>", "<user>", "<system>", "<knowledge>"]}
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)

        self.eos_id = self.tokenizer.eos_token_id
        #self.device = torch.device('cuda')
        
        self.prompt = [self.tokenizer.convert_tokens_to_ids('<context>'),
                        self.tokenizer.convert_tokens_to_ids('<response>')]
        self.user_id = [self.tokenizer.convert_tokens_to_ids('<user>'), 
                        self.tokenizer.convert_tokens_to_ids('<system>')]
        self.know_id = self.tokenizer.convert_tokens_to_ids('<knowledge>')
        
        self.load()

    def tokenize(self, text, text_pair=None):
        return self.tokenizer.encode(text, text_pair=text_pair, add_special_tokens=True, add_prefix_space=True)

    def preprocess(self, knowledge, history, user, response=None, segment=True, mode="train"):
    # knowledge: one knowledge sentence 
    # history: list ; one utterance is an item
    # user: list ; 0/1 is item
    # response: one response sentence 
    #pad为0
        if self.mode == "train":
            assert response is not None
            #knowledge
            knowledge_input = [self.know_id] + self.tokenize(knowledge)[:self.knowledge_truncate-2] + [self.eos_id]
            knowledge_type = len(knowledge_input) * [self.know_id]
            #history
            user = [u for u in user.tolist() if u >= 0]   # like [0, 1, 0, 1, 0]  [0, 1, 0] [0]
            history_input, history_type = [], []
            for h, u in zip(history, user):
                tmp = [self.user_id[u]] + self.tokenize(h) + [self.eos_id]
                history_input += tmp
                history_type += len(tmp) * [self.prompt[0]]
            history_input = [self.prompt[0]] + history_input[-(self.text_truncate-1):]
            history_type += [self.prompt[0]]
            history_type = history_type[-self.text_truncate:]
            #response
            response_input = [self.prompt[1]] + self.tokenize(response)
            response_type = len(response_input) * [self.prompt[1]]
            
            #concat
            ids = knowledge_input + history_input + response_input
            type_ids = knowledge_type + history_type + response_type
            tgt = [-1] * (len(knowledge_input) + len(history_input)) + response_input[1:] + [self.eos_id]

            ids = ids[-self.block_size:]
            type_ids = type_ids[-self.block_size:]
            tgt = tgt[-self.block_size:]

            ids = ids + [0] * (self.block_size - len(ids))
            type_ids = type_ids + [0] * (self.block_size - len(type_ids))
            tgt = tgt + [-1] * (self.block_size - len(tgt))
            
            input_ids = torch.tensor(ids).long()
            token_type_ids = torch.tensor(type_ids).long()
            targets = torch.tensor(tgt).long()
            
            if segment:
                return input_ids, token_type_ids, targets
            else:
                return input_ids, token_type_ids, targets

        elif self.mode == "inference":
            
            knowledge_input = [self.know_id] + self.tokenize(knowledge)[:self.knowledge_truncate-2] + [self.eos_id]

            user = [u for u in user.tolist() if u >= 0]
            history_input = []
            for h, u in zip(history, user):
                history_input += [self.user_id[u]] + self.tokenize(h) + [self.eos_id]

            input_ids = knowledge_input + [self.prompt[0]] + history_input[-(self.text_truncate-1):] + [self.prompt[1]]

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            token_type_ids = input_ids.clone().detach()
            targets = input_ids.clone().detach()


            return input_ids, token_type_ids, targets

    def load(self):
        infer_length = 0
        
        for id in range(len(self.episodes)):  #len episodes是样本数而非对话回合数，故此处下标为id，不是i
            episode = self.episodes[id]  #episode有五个回合
            episode_content = {"input_ids": [], "token_type_ids": [], "targets": []}   #已经做过处理，Knowledge pool中第一个即为gold one
            for id_in_episode, example in enumerate(episode):  #遍历每个episodes中的每轮对话回合,超过5则直接break
                if id_in_episode == self.max_episode_length:
                    break
                infer_length = infer_length + 1
                #data form: 'sentences'
                
                # get context
                context = []
                cur_query = self.query[example['query_id']]   #example即episode内容，query[example['query_id']]由tag变为content
                hist_list = example['context_id']
                if len(hist_list) == 0:
                    pass
                else:
                    for hist in hist_list:
                        context.append(self.query[hist])
                context.append(cur_query)
                
                #get response
                response = example['response']

                #get knowledge
                if self.mode == "train":
                    
                    knowledge = self.passage[example['knowledge_pool'][0]]
                else:
                    pred_indice = self.pred[id * 5 + id_in_episode]
                    knowledge = self.passage[example['knowledge_pool'][pred_indice]]
                
                #get user
                user = [0]
                for i in range(id_in_episode):
                    user = user + [1,0]
                user = torch.tensor(user)

                #train infer有所不同
                input_ids, token_type_ids, targets = self.preprocess(knowledge, context, user, response, self.segment, self.mode)
                
                
                if self.mode == "train":
                    
                    episode_content["input_ids"].append(input_ids)
                    episode_content["token_type_ids"].append(token_type_ids)
                    episode_content["targets"].append(targets)

                elif self.mode == "inference":
                    input_ids = torch.tensor([input_ids.tolist()], dtype=torch.long)
                    self.episode_tensor.append(
                        [torch.tensor([id]).long(), input_ids, input_ids, torch.tensor([id_in_episode]).long()])
                

            if self.mode == "train":
                # process episode
                assert len(episode_content["input_ids"]) == len(episode_content["token_type_ids"]) == len(
                    episode_content["targets"])

                episode_mask = torch.zeros(self.max_episode_length)  # [self.max_episode_length]
                episode_mask[:len(episode_content["input_ids"])] = 1
                episode_mask = episode_mask == 1

                #while循环，处理对话至五个一轮
                while len(episode_content["input_ids"]) < self.max_episode_length:
                    episode_content["input_ids"].append(torch.tensor(([0] * self.block_size), requires_grad=False).long())
                    episode_content["token_type_ids"].append(torch.tensor(([0] * self.block_size), requires_grad=False).long())
                    episode_content["targets"].append(torch.tensor(([-1] * self.block_size), requires_grad=False).long())

                assert len(episode_content["input_ids"]) == len(episode_content["token_type_ids"]) == len(
                    episode_content["targets"]) == self.max_episode_length

                id_episode_tensor = torch.tensor([id]).long()  # 单tensor
                input_ids_episode_tensor = torch.stack(episode_content["input_ids"])  # [max_episode_length, context_len]
                token_type_ids_episode_tensor = torch.stack(episode_content["token_type_ids"])  # [max_episode_length, max_dec_length]
                targets_episode_tensor = torch.stack(episode_content["targets"])  # [max_episode_length, max_knowledge_pool, knowledge_sentence_len]
                
                self.episode_tensor.append(
                [id_episode_tensor, input_ids_episode_tensor, token_type_ids_episode_tensor, targets_episode_tensor])

            
            if self.mode == "train":
                self.len = id + 1
            elif self.mode == "inference":
                self.len = infer_length

            

    def __getitem__(self, index):
        episode = self.episode_tensor[index]
        return [episode[0], episode[1], episode[2], episode[3]]

    def __len__(self):
        return self.len

    def context_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['context_id']  # list

    def query_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['query_id']  # string

    def passage_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['knowledge_label']  # list

    def knowledge_pool(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['knowledge_pool']  # list


def collate_fn_gpt2(data):
    id_episodes, input_ids_episodes, token_type_ids_episodes, targets_episodes = zip(*data)

    return {'episode_id': torch.cat(id_episodes),  
            'input_ids': torch.stack(input_ids_episodes),  
            'token_type_ids': torch.stack(token_type_ids_episodes), 
            'targets': torch.stack(targets_episodes), 
    }

