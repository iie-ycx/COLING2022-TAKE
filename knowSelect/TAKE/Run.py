import sys
sys.path.append('./')
from TAKE.Dataset import *
from torch import optim
from TAKE.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from TAKE.Model import *
from dataset.Utils_TAKE import *
from transformers import get_constant_schedule, get_linear_schedule_with_warmup
import os
import time



def train(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    init_seed(42)

    data_path = args.base_data_path+args.dataset+'/'

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()
    print('Vocabulary size', len(vocab2id))

    if os.path.exists(data_path + 'train_TAKE.pkl'):
        #load data
        query = torch.load(data_path + 'query_TAKE.pkl')
        train_samples = torch.load(data_path + 'train_TAKE.pkl')
        passage = torch.load(data_path + 'passage_TAKE.pkl')
        print("The number of train_samples:", len(train_samples))

    else:
        episodes, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                                                   data_path + args.dataset + '.passage',
                                                                   data_path + args.dataset + '.pool',
                                                                   data_path + 'ID_label.json',
                                                                   data_path + args.dataset + '.query',
                                                                   tokenizer)

        if args.dataset == "wizard_of_wikipedia":
            train_episodes, dev_episodes, test_seen_episodes, test_unseen_episodes = split_data(args.dataset, data_path + args.dataset + '.split', episodes)
            print("The number of test_seen_episodes:", len(test_seen_episodes))
            print("The number of test_unseen_episodes:", len(test_unseen_episodes))
            torch.save(test_seen_episodes, data_path + 'test_seen_TAKE.pkl')
            torch.save(test_unseen_episodes, data_path + 'test_unseen_TAKE.pkl')

        elif args.dataset == "holl_e":
            train_episodes, dev_episodes, test_episodes, = split_data(args.dataset, data_path + args.dataset + '.split', episodes)
            print("The number of test_episodes:", len(test_episodes))
            torch.save(test_episodes, data_path + 'test_TAKE.pkl')

        print("The number of train_episodes:", len(train_episodes))
        torch.save(query, data_path + 'query_TAKE.pkl')
        torch.save(passage, data_path + 'passage_TAKE.pkl')
        torch.save(train_episodes, data_path + 'train_TAKE.pkl')
        query = torch.load(data_path + 'query_TAKE.pkl')
        train_samples = torch.load(data_path + 'train_TAKE.pkl')
        passage = torch.load(data_path + 'passage_TAKE.pkl')
        print("The number of train_samples:", len(train_samples))


    model = TAKE(vocab2id, id2vocab, args)
    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')

    if args.resume is True:
        print("Reading checkpoints...")

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)
        last_epoch = checkpoints["time"][-1]

        fuse_dict = torch.load(os.path.join(saved_model_path, '.'.join([str(last_epoch), 'pkl'])))
        model.load_state_dict(fuse_dict["model"])
        print('Loading success, last_epoch is {}'.format(last_epoch))
    else:
        init_params(model, "enc.")   #模型非预训练部分参数初始化

        last_epoch = -1
        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            checkpoints = {"time": []}
            json.dump(checkpoints, w)

    all_params = model.parameters()
    Bert_params = []
    ID_params = []
    for pname, p in model.named_parameters():
        if "enc." in pname:
            Bert_params += [p] 
        elif "initiative_discriminator" in pname:
            ID_params += [p]  

    params_id = list(map(id, Bert_params)) + list(map(id, ID_params))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))

    params = [
    {"params": other_params, "lr": args.lr},
    {"params": Bert_params, "lr": args.Bertlr},
    {"params": ID_params, "lr": args.IDlr},
    ]

    len_dataset = len(train_samples)
    epoch_num = args.epoches
    batch_size = args.train_batch_size
    accumulation_steps = args.accumulation_steps
    if len_dataset % (batch_size * accumulation_steps) == 0:
        total_steps = (len_dataset // (batch_size * accumulation_steps)) * epoch_num  
    else:
        total_steps = (len_dataset // (batch_size * accumulation_steps) + 1) * epoch_num
    warm_up_ratio = 0.5

    # construct an optimizer object for dis net
    model_optimizer = optim.Adam(params) # model.parameters() Returns an iterator over module parameters.This is typically passed to an optimizer.
    model_scheduler = get_linear_schedule_with_warmup(model_optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    

    trainer = CumulativeTrainer(args.name, model, tokenizer, detokenizer, args.local_rank, accumulation_steps=args.accumulation_steps)
    model_optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.   

    train_dataset = Dataset(args.mode, train_samples, query, passage, vocab2id, args.max_episode_length,
                            args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference, args.knowledge_sentence_len, args.context_len,
                            args.max_dec_length)
    
    freeze_params(model, "student")
    for i in range(last_epoch+1, args.epoches):  
        if i == args.switch_ID:
            freeze_params(model, "teacher")
            unfreeze_params(model, "student")
        trainer.train_epoch('train', train_dataset, collate_fn, args.train_batch_size, i, model_optimizer, model_scheduler)
        trainer.serialize(i, model_scheduler, saved_model_path=saved_model_path)   #serialize 连载



def inference(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    data_path = args.base_data_path + args.dataset + '/'

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    print('Vocabulary size', len(vocab2id))

    if os.path.exists(data_path + 'test_seen_TAKE.pkl') or os.path.exists(data_path + 'test_TAKE.pkl'):
        query = torch.load(data_path + 'query_TAKE.pkl')
        passage = torch.load(data_path + 'passage_TAKE.pkl')

        test_seen_episodes = torch.load(data_path + 'test_seen_TAKE.pkl')
        test_unseen_episodes = torch.load(data_path + 'test_unseen_TAKE.pkl')
        print("The number of test_seen_episodes:", len(test_seen_episodes))
        print("The number of test_unseen_episodes:", len(test_unseen_episodes))

    else:
        episodes, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                                                   data_path + args.dataset + '.passage',
                                                                   data_path + args.dataset + '.pool',
                                                                   data_path + 'ID_label.json',
                                                                   data_path + args.dataset + '.query',
                                                                   tokenizer)

        train_episodes, dev_episodes, test_seen_episodes, test_unseen_episodes = split_data(args.dataset, data_path + args.dataset + '.split', episodes)
        print("The number of test_seen_episodes:", len(test_seen_episodes))
        print("The number of test_unseen_episodes:", len(test_unseen_episodes))
        torch.save(test_seen_episodes, data_path + 'test_seen_TAKE.pkl')
        torch.save(test_unseen_episodes, data_path + 'test_unseen_TAKE.pkl')


        print("The number of train_episodes:", len(train_episodes))
        torch.save(query, data_path + 'query_TAKE.pkl')
        torch.save(passage, data_path + 'passage_TAKE.pkl')
        torch.save(train_episodes, data_path + 'train_TAKE.pkl')



    test_seen_dataset = Dataset(args.mode, test_seen_episodes, query, passage, vocab2id, args.max_episode_length,
                              args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                              args.knowledge_sentence_len, args.context_len, args.max_dec_length)

    test_unseen_dataset = Dataset(args.mode, test_unseen_episodes, query, passage, vocab2id, args.max_episode_length,
                              args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                              args.knowledge_sentence_len, args.context_len, args.max_dec_length)


    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')

    def inference(dataset, epoch=None):
        file =saved_model_path + str(epoch) + '.pkl'
        if os.path.exists(file):
            model = TAKE(vocab2id, id2vocab, args)

            model.load_state_dict(torch.load(file)["model"])
            trainer = CumulativeTrainer(args.name, model, tokenizer, detokenizer, None)

            print('inference {}'.format("test_seen_dataset"))
            trainer.test('inference', test_seen_dataset, collate_fn, args.inference_batch_size, 'test_seen', str(epoch), output_path=args.base_output_path + args.name+"/")
            print('inference {}'.format("test_unseen_dataset"))
            trainer.test('inference', test_unseen_dataset, collate_fn, args.inference_batch_size, 'test_unseen', str(epoch), output_path=args.base_output_path + args.name+"/")

           
    if not os.path.exists(saved_model_path+"finished_inference.json"):
        finished_inference = {"time": []}
        w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()

    if args.appoint_epoch != -1:
        print('Start inference at epoch', args.appoint_epoch)
        inference(args.dataset, args.appoint_epoch)

        r = open(saved_model_path+"finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        finished_inference["time"].append(args.appoint_epoch)
        w = open(saved_model_path + "finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()
        print("finished epoch {} inference".format(args.appoint_epoch))
        exit()

    while True:
        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
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
                    inference(args.dataset, i)

                    r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
                    finished_inference = json.load(r)
                    r.close()

                    finished_inference["time"].append(i)

                    w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
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
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--GPU", type=int, default=0)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--epoches", type=int, default=15)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5) 
    parser.add_argument("--Bertlr", type=float, default=1e-5)  
    parser.add_argument("--IDlr", type=float, default=6e-5)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--appoint_epoch", type=int, default=-1)
    parser.add_argument("--anneal_rate", type=float, default=0.05)
    parser.add_argument('--min_ratio', type=float, default=0.8)   
    parser.add_argument("--switch_ID", type=int, default=4)

    parser.add_argument("--max_episode_length", type=int, default=5)
    parser.add_argument("--context_len", type=int, default=35)
    parser.add_argument("--max_dec_length", type=int, default=52)
    parser.add_argument("--knowledge_sentence_len", type=int, default=34)
    parser.add_argument("--max_knowledge_pool_when_train", type=int, default=32)
    parser.add_argument("--max_knowledge_pool_when_inference", type=int, default=128)

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