import sys
sys.path.append('./')
from torch.utils.data import DataLoader
from evaluation.Eval_Rouge import *
from evaluation.Eval_Bleu import *
from evaluation.Eval_Meteor import *
from evaluation.Eval_F1 import *
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score
from TAKE.Utils import *
import json
import os


def rounder(num, places):
    return round(num, places)

def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if escape is not None and escape in name:
            continue
        if param.data.dim() > 1:
            xavier_uniform_(param.data)


def freeze_params(model, freeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if freeze is not None and freeze in name:
            param.requires_grad = False


def unfreeze_params(model, unfreeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if unfreeze is not None and unfreeze in name:
            param.requires_grad = True

class CumulativeTrainer(object):
    def __init__(self, theModule, name, model, tokenizer, detokenizer=None, local_rank=None, accumulation_steps=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.name = name
        self.module = theModule

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0

#train mini-batch
    def train_batch(self, epoch, data, method, optimizer, scheduler=None):


        if self.module == 'gen':
            self.accumulation_count += 1
            loss_g= self.model(data, method=method)
            loss = loss_g / self.accumulation_steps

            loss.backward()

            if self.accumulation_count % self.accumulation_steps == 0:
            # The norm is computed over all gradients together,
            # as if they were concatenated into a single vector.
            # return is Total norm of the parameters (viewed as a single vector).
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
            # torch.optim.Adam.step()
            # Performs a single optimization step.
                optimizer.step()
                if scheduler is not None:
                # Learning rate scheduling should be applied after optimizer’s update
                    scheduler.step()
                optimizer.zero_grad()

            return loss_g.cpu().item()


    def serialize(self, epoch, scheduler, saved_model_path):

        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}

        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        print("Saved epoch {} model".format(epoch))

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        checkpoints["time"].append(epoch)

        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            json.dump(checkpoints, w)

#train main 
    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer, scheduler=None):
        self.model.train()  # Sets the module in training mode；
        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)


        if self.module == 'gen':
            start_time = time.time()
            count_batch = 0

            accu_loss_g = 0.
            step = 0

            for j, data in enumerate(train_loader, 0):
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                    data = data_cuda
                count_batch += 1

                loss_g = self.train_batch(epoch, data, method=method, optimizer=optimizer, scheduler=scheduler)
            #accumulate loss
                accu_loss_g += loss_g
                step +=1

                if j >= 0 and j % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print('Training: %s' % self.name)
                
                    if scheduler is not None:
                        # Calculates the learning rate at batch index.
                        print(
                            'Epoch:{}, Step:{}, Batch:{}, loss_g:{}, Time:{}, LR:{}'.format(
                                epoch, scheduler.state_dict()['_step_count'], count_batch,
                                rounder(accu_loss_g / step, 4),
                                rounder(elapsed_time, 2),
                                scheduler.get_last_lr()))
                            
                    accu_loss_g = 0.
                    step = 0
                    sys.stdout.flush()

                del loss_g
            
            sys.stdout.flush()


    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        #  changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        self.eval_model.eval()   
        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            if self.module == 'gen':
                hyps = []

                for k, data in enumerate(test_loader, 0):
                    print("doing {} / total {} in {}".format(k+1, len(test_loader), epoch))
                
                    if torch.cuda.is_available():
                        data_cuda = dict()
                        for key, value in data.items():
                            if isinstance(value, torch.Tensor):
                                data_cuda[key] = value.cuda()
                            else:
                                data_cuda[key] = value
                        data = data_cuda
                    hyp = self.eval_model(data, method=method)

                    episode_id = data['episode_id'].item()  
                    example_id = data['targets'].item()
                    
                    hyps.append([';'.join(dataset.context_id(episode_id, example_id)), 
                                        dataset.query_id(episode_id, example_id), 
                                        dataset.knowledge_pool(episode_id, example_id)[dataset.pred[episode_id * 5 + example_id]],   #选到的知识
                                        hyp]) 

                output_path = os.path.join(output_path, str(epoch) + '.txt')

                file = codecs.open(output_path, "w", "utf-8")
                
                for i in range(len(hyps)):
                    file.write('\t'.join(hyps[i])+os.linesep)
                file.close()

                return output_path, dataset.answer_file
        


    def test(self, method, dataset, collate_fn, batch_size, dataset_name, epoch, output_path):


        if self.module == 'gen':
            with torch.no_grad():
                run_file, answer_file= self.predict(method, dataset, collate_fn, batch_size, dataset_name+"_"+epoch, output_path)


                from transformers import BertTokenizer
                def bert_tokenizer():
                    t = BertTokenizer.from_pretrained(
                        'bert-base-uncased', do_lower_case=True)  # do_lower_case Whether to lower case the input.
                    return t.tokenize, t.vocab, t.ids_to_tokens

                def bert_detokenizer():
                    def detokenizer(tokens):
                        return ' '.join(tokens).replace(' ##', '').strip()
                    return detokenizer

                tokenizer, vocab2id, id2vocab = bert_tokenizer()
                detokenizer = bert_detokenizer()
                print("Start auotimatic evaluation")

                f1 = eval_f1_file(run_file, answer_file, tokenizer, detokenizer)
                print("F1", f1)

                bleus = eval_bleu_file(run_file, answer_file, tokenizer, detokenizer)
                print("BLEU", bleus)

                rouges = eval_rouge_file(run_file, answer_file, tokenizer, detokenizer)
                print("ROUGE", rouges)

                metoers = eval_meteor_file(run_file, answer_file, tokenizer, detokenizer)
                print("METOER", metoers)

                metric_output = {**f1, **bleus, **rouges, **metoers}  
                print({epoch+"_"+dataset_name: metric_output})

                try:
                    with open(os.path.join(output_path, dataset_name + "_result.json"), 'r', encoding='utf-8') as r:
                        result_log = json.load(r)
                    result_log[epoch + "_" + dataset_name] = metric_output
                    with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                        json.dump(result_log, w)

                except FileNotFoundError:
                    with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                        result_log={}
                        result_log[epoch+"_"+dataset_name] = metric_output
                        json.dump(result_log, w)
