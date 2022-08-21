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
            xavier_normal_(param.data)


def freeze_params(model, freeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if freeze is not None and freeze in name:
            param.requires_grad = False


def unfreeze_params(model, unfreeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if unfreeze is not None and unfreeze in name:
            param.requires_grad = True

class CumulativeTrainer(object):
    def __init__(self, name, model, tokenizer, detokenizer, local_rank=None, accumulation_steps=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.name = name

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0

#train mini-batch
    def train_batch(self, epoch, data, method, optimizer, scheduler=None):
        self.accumulation_count += 1

        loss_ks, loss_distill, final_ks_acc, shift_ks_acc, inherit_ks_acc, s_shift_prob, loss_ID, ID_acc = self.model(data, method=method, epoch=epoch)
        lambda_weight = 0.5
        loss_primary = loss_ks
        loss_aboutID = loss_distill + loss_ID
        loss = (loss_primary + loss_aboutID * lambda_weight) / self.accumulation_steps

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



        return loss_ks.cpu().item(), loss_distill.cpu().item(), \
               final_ks_acc, shift_ks_acc, inherit_ks_acc, \
               s_shift_prob.cpu().item(), loss_ID.cpu().item(), ID_acc

    def serialize(self, epoch, scheduler, saved_model_path):

        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}

        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        print("Saved epoch {} model".format(epoch))

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        checkpoints["time"].append(epoch)

        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            json.dump(checkpoints, w)


    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer, scheduler=None):
        self.model.train()  # Sets the module in training mode；
        if torch.cuda.is_available():
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)

        start_time = time.time()
        count_batch = 0

        accu_loss_ks = 0.
        accu_loss_distill = 0.   
        accu_loss_ID = 0.
        accu_final_ks_acc = 0.
        accu_shift_ks_acc = 0.
        accu_inherit_ks_acc = 0.
        accu_ID_acc = 0.
        accu_s_shift_prob = 0.

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

            loss_ks, loss_distill, final_ks_acc, shift_ks_acc, inherit_ks_acc, s_shift_prob, loss_ID, ID_acc = self.train_batch(epoch, data, method=method, optimizer=optimizer, scheduler=scheduler)
            #accumulate loss
            accu_loss_ks += loss_ks
            accu_loss_distill += loss_distill
            accu_loss_ID += loss_ID
            accu_final_ks_acc += final_ks_acc
            accu_shift_ks_acc += shift_ks_acc
            accu_inherit_ks_acc += inherit_ks_acc
            accu_ID_acc += ID_acc
            accu_s_shift_prob += s_shift_prob
            step +=1

            if j >= 0 and j % 500 == 0:
                elapsed_time = time.time() - start_time
                print('Training: %s' % self.name)
                
                if scheduler is not None:
                    # Calculates the learning rate at batch index.
                    print(
                        'Epoch:{}, Step:{}, Batch:{}, loss_ks:{}, loss_distill&student:{}, loss_ID_t:{}, \
                         final_ks_acc:{}, shift_ks_acc:{}, inherit_ks_acc:{}, ID_acc:{}, Time:{}, LR:{}'.format(
                            epoch, scheduler.state_dict()['_step_count'], count_batch,
                            rounder(accu_loss_ks / step, 4),
                            rounder(accu_loss_distill / step, 4),
                            rounder(accu_loss_ID / step, 4),
                            rounder(accu_final_ks_acc / step, 4),
                            rounder(accu_shift_ks_acc / step, 4),
                            rounder(accu_inherit_ks_acc / step, 4),
                            rounder(accu_ID_acc / step, 4),
                            rounder(elapsed_time, 2),
                            scheduler.get_last_lr()))
                            
                accu_loss_ks = 0.
                accu_loss_distill = 0.
                accu_loss_ID = 0.
                accu_final_ks_acc = 0.
                accu_shift_ks_acc = 0.
                accu_inherit_ks_acc = 0.
                accu_ID_acc = 0.
                accu_s_shift_prob = 0.
                
                step = 0


                sys.stdout.flush()

            del loss_ks
            del loss_distill
            del loss_ID
            del final_ks_acc
            del shift_ks_acc
            del inherit_ks_acc
            del ID_acc
            del s_shift_prob
            

        sys.stdout.flush()


    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        #  changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        self.eval_model.eval()   

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_final_ks_pred = []
            accumulative_shift_ks_pred = []
            accumulative_inherit_ks_pred = []
            accumulative_knowledge_label = []
            accumulative_ID_pred = []
            accumulative_ID_label = []
            accumulative_episode_mask = []

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
             
                # final_ks_pred [batch * max_episode_length]
                final_ks_pred, shift_ks_pred, inherit_ks_pred, ID_pred = self.eval_model(data, method=method)  #此处数据是按批传入的

                accumulative_ID_pred.append(ID_pred) # [[batch * max_episode_length],...]
                accumulative_final_ks_pred.append(final_ks_pred)  # [[batch * max_episode_length],...]
                accumulative_shift_ks_pred.append(shift_ks_pred)
                accumulative_inherit_ks_pred.append(inherit_ks_pred)
                accumulative_ID_label.append(data['Initiative_label'].reshape(-1)) # [[batch * max_episode_length],...]
                accumulative_knowledge_label.append(data['knowledge_label'].reshape(-1)) # [[batch * max_episode_length],...]
                accumulative_episode_mask.append(data['episode_mask'].reshape(-1))  # [[batch * max_episode_length],...]

                _, max_episode_length = data['episode_mask'].size()


            accumulative_final_ks_pred = torch.cat(accumulative_final_ks_pred, dim=0)
            accumulative_shift_ks_pred = torch.cat(accumulative_shift_ks_pred, dim=0)
            accumulative_inherit_ks_pred = torch.cat(accumulative_inherit_ks_pred, dim=0)

            accumulative_ID_pred = torch.cat(accumulative_ID_pred, dim=0)
            accumulative_knowledge_label = torch.cat(accumulative_knowledge_label, dim=0)
            accumulative_ID_label = torch.cat(accumulative_ID_label, dim=0)
            accumulative_episode_mask = torch.cat(accumulative_episode_mask, dim=0)
            accumulative_final_ks_acc = accuracy_score(accumulative_knowledge_label.cpu(), accumulative_final_ks_pred.cpu(), sample_weight=accumulative_episode_mask.cpu())
            accumulative_shift_ks_acc = accuracy_score(accumulative_knowledge_label.cpu(), accumulative_shift_ks_pred.cpu(), sample_weight=accumulative_episode_mask.cpu())
            accumulative_inherit_ks_acc = accuracy_score(accumulative_knowledge_label.cpu(), accumulative_inherit_ks_pred.cpu(), sample_weight=accumulative_episode_mask.cpu())

            with open(os.path.join(output_path + '/ks_pred/', epoch + "_ks_pred.json"), 'w', encoding='utf-8') as w:
                json.dump(accumulative_final_ks_pred.tolist(), w)

            accumulative_ID_acc = accuracy_score(accumulative_ID_label.cpu(), accumulative_ID_pred.cpu(), sample_weight=accumulative_episode_mask.cpu())


        return output_path, dataset.answer_file, {"ks_acc": rounder(100*(accumulative_final_ks_acc), 2)}, {"shift_ks_acc": rounder(100*(accumulative_shift_ks_acc), 2)}, {"inherit_ks_acc": rounder(100*(accumulative_inherit_ks_acc), 2)}, {"ID_acc": rounder(100*(accumulative_ID_acc), 2)}


    def test(self, method, dataset, collate_fn, batch_size, dataset_name, epoch, output_path):
        #  disables tracking of gradients in autograd.
        # In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
        with torch.no_grad():
            run_file, answer_file, final_ks_acc, shift_ks_acc, inherit_ks_acc, ID_acc = self.predict(method, dataset, collate_fn, batch_size, dataset_name+"_"+epoch, output_path)

        print("Start auotimatic evaluation")

        print("KNOW_ACC", final_ks_acc)
        print("shift KNOW_ACC", shift_ks_acc)
        print("inherit KNOW_ACC", inherit_ks_acc)
        print("ID_ACC", ID_acc)

        metric_output = {**final_ks_acc, **shift_ks_acc, **inherit_ks_acc, **ID_acc}  
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

        return None
