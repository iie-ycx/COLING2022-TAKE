import sys
sys.path.append('./')
from TAKE.gpt2Dataset import gpt2Dataset, collate_fn_gpt2
from torch import optim
from TAKE.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from TAKE.Model import *
from dataset.Utils_TAKE import *
from transformers import get_constant_schedule, get_linear_schedule_with_warmup
import os
import time
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config, GPT2Tokenizer


def train(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    init_seed(42)

    gpt2_data_path = args.gpt2_data_path+'/'


    if os.path.exists(gpt2_data_path + 'train_TAKE.pkl'):
        query_gpt2 = torch.load(gpt2_data_path + 'query_TAKE.pkl')
        train_samples_gpt2 = torch.load(gpt2_data_path + 'train_TAKE.pkl')
        passage_gpt2 = torch.load(gpt2_data_path + 'passage_TAKE.pkl')


    else:
        episodes, query_gpt2, passage_gpt2 = load_default(args.dataset, gpt2_data_path + args.dataset + '.answer',
                                                                   gpt2_data_path + args.dataset + '.passage',
                                                                   gpt2_data_path + args.dataset + '.pool',
                                                                   gpt2_data_path + args.dataset + '.qrel',
                                                                   gpt2_data_path + args.dataset + '.query'
                                                                   )

        train_episodes_gpt2, dev_episodes_gpt2, test_seen_episodes_gpt2, test_unseen_episodes_gpt2 = split_data(args.dataset, gpt2_data_path + args.dataset + '.split', episodes)

        torch.save(test_seen_episodes_gpt2, gpt2_data_path + 'test_seen_TAKE.pkl')
        torch.save(test_unseen_episodes_gpt2, gpt2_data_path + 'test_unseen_TAKE.pkl')
        torch.save(query_gpt2, gpt2_data_path + 'query_TAKE.pkl')
        torch.save(passage_gpt2, gpt2_data_path + 'passage_TAKE.pkl')
        torch.save(train_episodes, gpt2_data_path + 'train_TAKE.pkl')


    #gen dataset
    gpt2_train_dataset = gpt2Dataset(args.mode, train_samples_gpt2, query_gpt2, passage_gpt2, None, args.segment, args.max_episode_length,
                                        args.knowledge_truncate, args.text_truncate, args.gpt2_truncate, args=args)

    gpt2tokenizer = gpt2_train_dataset.tokenizer
    model = GPT2_gen(gpt2tokenizer, args)
    gen_saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'gen_model/')

    if args.resume is True:
        print("Reading checkpoints...")

        with open(gen_saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)
        last_epoch = checkpoints["time"][-1]

        fuse_dict = torch.load(os.path.join(gen_saved_model_path, '.'.join([str(last_epoch), 'pkl'])))
        model.load_state_dict(fuse_dict["model"])
        print('Loading success, last_epoch is {}'.format(last_epoch))
    else:
        last_epoch = -1
        with open(gen_saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            checkpoints = {"time": []}
            json.dump(checkpoints, w)


    # construct an optimizer object for gen net
    model_optimizer = optim.Adam(model.parameters(), args.Declr) # model.parameters() Returns an iterator over module parameters.This is typically passed to an optimizer.
    model_scheduler = get_constant_schedule(model_optimizer) 
    trainer = CumulativeTrainer('gen', args.name, model, gpt2tokenizer, detokenizer=None, local_rank=args.local_rank, accumulation_steps=args.accumulation_steps)
    model_optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s. 

    #train gen net
    for i in range(last_epoch+1, args.gen_epoches): 
        gpt2_train_dataset = gpt2Dataset(args.mode, train_samples_gpt2, query_gpt2, passage_gpt2, tr_ks_pred, args.segment, args.max_episode_length,
                                        args.knowledge_truncate, args.text_truncate, args.gpt2_truncate, i, args=args)

        args.train_batch_size = 4
        args.accumulation_steps = 16
        trainer.train_epoch('train', gpt2_train_dataset, collate_fn_gpt2, args.train_batch_size, i, model_optimizer, model_scheduler)
        trainer.serialize(i, model_scheduler, saved_model_path=gen_saved_model_path) 
        del gpt2_train_dataset

def inference(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    gpt2_data_path = args.gpt2_data_path+'/'


#-------------------------------------------------------------------------------------------------------------
#预处理数据
    if os.path.exists(gpt2_data_path + 'test_seen_TAKE.pkl'):
        query_gpt2 = torch.load(gpt2_data_path + 'query_TAKE.pkl')
        test_seen_episodes_gpt2 = torch.load(gpt2_data_path + 'test_seen_TAKE.pkl')
        test_unseen_episodes_gpt2 = torch.load(gpt2_data_path + 'test_unseen_TAKE.pkl')
        passage_gpt2 = torch.load(gpt2_data_path + 'passage_TAKE.pkl')
        print("The number of test_seen_episodes:", len(test_seen_episodes_gpt2))
        print("The number of test_unseen_episodes:", len(test_unseen_episodes_gpt2))


    else:
        episodes, query_gpt2, passage_gpt2 = load_default(args.dataset, gpt2_data_path + args.dataset + '.answer',
                                                                   gpt2_data_path + args.dataset + '.passage',
                                                                   gpt2_data_path + args.dataset + '.pool',
                                                                   gpt2_data_path + args.dataset + '.qrel',
                                                                   gpt2_data_path + args.dataset + '.query'
                                                                   )

        train_episodes, dev_episodes, test_seen_episodes_gpt2, test_unseen_episodes_gpt2 = split_data(args.dataset, gpt2_data_path + args.dataset + '.split', episodes)
        print("The number of test_seen_episodes:", len(test_seen_episodes_gpt2))
        print("The number of test_unseen_episodes:", len(test_unseen_episodes_gpt2))
        torch.save(test_seen_episodes_gpt2, gpt2_data_path + 'test_seen_TAKE.pkl')
        torch.save(test_unseen_episodes_gpt2, gpt2_data_path + 'test_unseen_TAKE.pkl')


        print("The number of train_episodes:", len(train_episodes))
        torch.save(query_gpt2, gpt2_data_path + 'query_TAKE.pkl')
        torch.save(passage_gpt2, gpt2_data_path + 'passage_TAKE.pkl')
        torch.save(train_episodes, gpt2_data_path + 'train_TAKE.pkl')
#-------------------------------------------------------------------------------------------------------------


    def inference(module, dataset, epoch=None):
        

        args.inference_batch_size = 1
        file =gen_saved_model_path + str(epoch) + '.pkl'
        model = GPT2_gen(gpt2tokenizer, args)

        model.load_state_dict(torch.load(file)["model"])
        trainer = CumulativeTrainer('gen', args.name, model, gpt2tokenizer, detokenizer, None)

        print('inference {}'.format("test_seen_dataset"))
        trainer.test('inference', gpt2_ts_seen_dataset, collate_fn_gpt2, args.inference_batch_size, 'test_seen', str(epoch), output_path=args.base_output_path + args.name+"/result_gen/")
        print('inference {}'.format("test_unseen_dataset"))
        trainer.test('inference', gpt2_ts_unseen_dataset, collate_fn_gpt2, args.inference_batch_size, 'test_unseen', str(epoch), output_path=args.base_output_path + args.name+"/result_gen/")



    gen_saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'gen_model/')
    
    
    with open(os.path.join(args.base_output_path + args.name + '/ks_pred/', 'test_seen_' + '12' + "_ks_pred.json"), 'r', encoding='utf-8') as r:
        ts_seen_ks_pred = json.load(r)
    with open(os.path.join(args.base_output_path + args.name + '/ks_pred/', 'test_unseen_' + '10' + "_ks_pred.json"), 'r', encoding='utf-8') as r:
        ts_unseen_ks_pred = json.load(r)

    
    gpt2_ts_seen_dataset = gpt2Dataset(args.mode, test_seen_episodes_gpt2, query_gpt2, passage_gpt2, ts_seen_ks_pred, args.segment, args.max_episode_length,
                                         args.knowledge_truncate, args.text_truncate, args.gpt2_truncate, args=args)

    gpt2_ts_unseen_dataset = gpt2Dataset(args.mode, test_unseen_episodes_gpt2, query_gpt2, passage_gpt2, ts_unseen_ks_pred, args.segment, args.max_episode_length,
                                         args.knowledge_truncate, args.text_truncate, args.gpt2_truncate, args=args)
    gpt2tokenizer = gpt2_ts_seen_dataset.tokenizer


    while True:
        with open(gen_saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        r = open(gen_saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        if len(checkpoints["time"]) == 0:
            print('Inference_mode: wait train finish the first epoch...')
            time.sleep(300)
        else:
            for i in checkpoints["time"]:  # i is the index of epoch
                if i in finished_inference["time"]:
                    print("epoch {} already has been inferenced, skip it".format(i))
                    pass
                else:
                    print('Start inference at epoch', i)
                    inference('gen', args.dataset, i)

                    r = open(gen_saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
                    finished_inference = json.load(r) 
                    r.close()

                    finished_inference["time"].append(i)

                    w = open(gen_saved_model_path + "finished_inference.json", 'w', encoding='utf-8')
                    json.dump(finished_inference, w) 
                    w.close()
                    print("finished epoch {} inference".format(i))

            print("Inference_mode: current all model checkpoints are completed...")
            print("Inference_mode: finished %d modes" % len(finished_inference["time"]))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--name", type=str, default='TAKE')
    parser.add_argument("--base_output_path", type=str, default='output/')
    parser.add_argument("--base_data_path", type=str, default='datasets/')
    parser.add_argument("--gpt2_data_path", type=str, default='datasets/wow_gpt2/')
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--GPU", type=int, default=2)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--epoches", type=int, default=15)
    parser.add_argument("--dis_epoches", type=int, default=5)
    parser.add_argument("--gen_epoches", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=16)  
    parser.add_argument("--lr", type=float, default=0.00004) 
    parser.add_argument("--Bertlr", type=float, default=0.00001)  
    parser.add_argument("--Declr", type=float, default=0.00003)  
    parser.add_argument("--train_batch_size", type=int, default=2) 
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--appoint_epoch", type=int, default=-1)
    parser.add_argument("--ks_best_epoch", type=str, default='12')
    parser.add_argument("--anneal_rate", type=float, default=0.05)
    parser.add_argument('--min_ratio', type=float, default=0.5)  

    parser.add_argument("--max_episode_length", type=int, default=5)
    parser.add_argument("--context_len", type=int, default=52)
    parser.add_argument("--knowledge_sentence_len", type=int, default=34)
    parser.add_argument("--max_knowledge_pool_when_train", type=int, default=32) 
    parser.add_argument("--max_knowledge_pool_when_inference", type=int, default=128)

    parser.add_argument('--gpt2_truncate', type=int, default=256) # for gpt2
    parser.add_argument('--knowledge_truncate', type=int, default=64) # for gpt2
    parser.add_argument('--text_truncate', type=int, default=128) # for gpt2
    parser.add_argument('--segment', type=bool, default=True)
    parser.add_argument("--max_dec_length", type=int, default=50)
    parser.add_argument("--min_dec_length", type=int, default=10)


    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dropout", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--ffn_size", type=int, default=768)

    parser.add_argument("--k_hidden_size", type=int, default=768)
    parser.add_argument("--k_n_layers", type=int, default=5)
    parser.add_argument("--k_n_heads", type=int, default=2)
    parser.add_argument("--k_ffn_size", type=int, default=768)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'train':
        train(args)
    else:
        Exception("no ther mode")