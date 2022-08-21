import codecs
from nltk.translate.meteor_score import *
import random
import re

def rounder(num):  #保留两位小数
    return round(num, 2)

re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""


    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))

def eval_meteor_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    run_dict = {}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            if tokenizer is not None:
                tokenized = detokenizer(tokenizer(temp[3]))
            else:
                tokenized = tokenized
            run_dict[temp[1]+'##<>##'+temp[2]] = tokenized
    #{'current_query_id##<>##background_id':'response_content'}
    
    
    #print(run_dict)
    ref_dict = {}
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            tokenized = temp[3]
            if tokenizer is not None:
                tokenized = detokenizer(tokenizer(temp[3]))
                
            if temp[1] in ref_dict:
                ref_dict[temp[1]].append(tokenized)
            else:
                ref_dict[temp[1]] = [tokenized]
            
   
    meteor = 0.
    #print(ref_dict)

    for id in run_dict: #遍历 键key    # [text1(,text2)] vs text
        #print(id)
        
        meteor += meteor_score([ref_dict[id.split('##<>##')[0]][0].split()], run_dict[id].split())
    return {'METEOR': rounder(meteor*100/len(run_dict))}

