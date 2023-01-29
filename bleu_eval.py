from datasets import load_metric
import numpy as np
import json
import sys
from tokenizer_utils import create_tokenizer
from transformers import AutoTokenizer
from sacremoses import MosesDetokenizer, MosesTokenizer
import os

mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
metric_bleu = load_metric("./bleu.py")
metric_sacrebleu = load_metric("./sacre_bleu.py")
metric_rouge = load_metric("./rouge.py")
tokenizer_mbert = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

def cal_metrics(data):
    refs = [[md.detokenize(mt.tokenize(item[-1]))] for item in data]
    preds = [md.detokenize(mt.tokenize(item[0])) for item in data]
    sacre_results = metric_sacrebleu.compute(predictions=preds, references=refs)
    print('***SacreBLEU score', round(sacre_results['score'], 2))

    refs = [[tokenizer_mbert.tokenize(item[-1])] for item in data]
    preds = [tokenizer_mbert.tokenize(item[0]) for item in data]
    results = metric_bleu.compute(predictions=preds, references=refs)
    print('*** tokenized BLEU score', round(results['bleu']*100, 2))
    

    refs = [item[-1] for item in data]
    preds = [item[0] for item in data]
    results = metric_rouge.compute(predictions=preds, references=refs)
    print('Rouge score', results)

    return sacre_results['score']

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = metric_sacrebleu.compute(predictions=[s1], 
                                        references=[[s2]])['score']
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))

    return sentences[idx]

input_file = sys.argv[1]
if os.path.exists(input_file):
    with open(input_file, 'r') as f:
            data = f.readlines()
            data = [json.loads(item.strip('\n')) for item in data]
    cal_metrics(data)

else:
    path = '/'.join(input_file.split('/')[:-1])
    prefix = input_file.split('/')[-1]
    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(prefix) and sys.argv[2] in f]
    print(files)
    refs = []
    preds = []
    for f in files:
        print('===='+f.split('/')[-1])

        with open(f, 'r') as fi:
            data = fi.readlines()
            data = [json.loads(item.strip('\n')) for item in data]
        
        if not refs:
            refs = [md.detokenize(mt.tokenize(item[-1])) for item in data]
        if not preds:
            preds = [[md.detokenize(mt.tokenize(item[0]))] for item in data]
        else:
            for idx, item in enumerate(data):
                preds[idx].append(item[0])

    preds = [selectBest(item) for item in preds]     
    data_buffer = []
    for p, r in zip(preds, refs):
        data_buffer.append([p,r])
    cal_metrics(data_buffer)


    