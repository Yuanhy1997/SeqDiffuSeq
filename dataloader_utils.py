import logging
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from functools import partial
from mpi4py import MPI
import os
import random
import numpy as np
logging.basicConfig(level=logging.INFO)

def get_dataloader(tokenizer, data_path, batch_size, max_seq_len, max_seq_len_src, args):

    dataset = TextDataset_translation(tokenizer=tokenizer, data_path=data_path, source=args.src, target=args.tgt,
                                        shard=MPI.COMM_WORLD.Get_rank(),
                                        num_shards=MPI.COMM_WORLD.Get_size())
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        drop_last=True,
        shuffle='train' in data_path,
        num_workers=10,
        collate_fn=partial(TextDataset_translation.collate_pad, 
                           args=args,
                           cutoff=max_seq_len, 
                           cutoff_src=max_seq_len_src,
                           padding_token=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else tokenizer.get_vocab()['<pad>']),
    )

    while True:
        for batch in dataloader:
            yield batch

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        has_labels: bool = False
        ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_data()
        if has_labels:
            self.read_labels()

    def read_data(self):
        logging.info("Reading data from {}".format(self.data_path))
        data = pd.read_csv(self.data_path, sep="\t", header=None)  # read text file
        logging.info(f"Tokenizing {len(data)} sentences")

        self.text = data[0].apply(lambda x: x.strip()).tolist()
        if hasattr(self.tokenizer, 'encode_batch'):

            encoded_input = self.tokenizer.encode_batch(self.text)
            self.input_ids = [x.ids for x in encoded_input]
        
        else:
            encoded_input = self.tokenizer(self.text)
            self.input_ids = encoded_input["input_ids"]

        

    def read_labels(self):
        self.labels = pd.read_csv(self.data_path, sep="\t", header=None)[1].tolist()
        # check if labels are already numerical
        self.labels = [str(x) for x in self.labels]
        if isinstance(self.labels[0], int):
            return
        # if not, convert to numerical
        all_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.idx_to_label = {i: label for i, label in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[label] for label in self.labels]
        
        
    
    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, i):
        out_dict = {
            "input_ids": self.input_ids[i],
            # "attention_mask": [1] * len(self.input_ids[i]),
        }
        if hasattr(self, "labels"):
            out_dict["label"] = self.labels[i]
        return out_dict

    @staticmethod
    def collate_pad(batch, cutoff: int):
        max_token_len = 0
        num_elems = len(batch)
        # batch[0] -> __getitem__[0] --> returns a tuple (embeddings, out_dict)

        for i in range(num_elems):
            max_token_len = max(max_token_len, len(batch[i]["input_ids"]))

        max_token_len = min(cutoff, max_token_len)

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        
        has_labels = False
        if "label" in batch[0]:
            labels = torch.zeros(num_elems).long()
            has_labels = True

        for i in range(num_elems):
            toks = batch[i]["input_ids"]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            if has_labels:
                labels[i] = batch[i]["label"]
        
        # TODO: the first return None is just for backward compatibility -- can be removed
        if has_labels:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask, "labels": labels}
        else:
            return None, {"input_ids": tokens, "attention_mask": tokens_mask}


class TextDataset_translation(TextDataset):

    def __init__(
        self,
        tokenizer,
        data_path: str,
        source,
        target,
        shard,
        num_shards,
        ) -> None:
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.shard = shard
        self.src = source
        self.tgt = target
        self.num_shards = num_shards
        self.read_data()

    def read_data(self):
        print("Reading data from {}".format(self.data_path))
        data = [open(self.data_path+'.'+self.src, 'r').readlines(),
                open(self.data_path+'.'+self.tgt, 'r').readlines()]
        print(f"Tokenizing {len(data[0])} sentences")

        data = [[src, tgt] for src, tgt in zip(data[0], data[1])]
        # random.shuffle(data)

        self.src_text = [item[0].strip('\n') for item in data]
        self.tgt_text = [item[1].strip('\n') for item in data]

        bos_idx = (len(self.src_text) // self.num_shards) * self.shard
        eos_idx = (len(self.src_text) // self.num_shards) * (self.shard+1)
        self.src_text = self.src_text[bos_idx:eos_idx]
        self.tgt_text = self.tgt_text[bos_idx:eos_idx]

        print('examples src', self.src_text[0])
        print('examples tgt', self.tgt_text[0])
        
        # check if tokenizer has a method 'encode_batch'
        if hasattr(self.tokenizer, 'encode_batch'):

            encoded_input_src = self.tokenizer.encode_batch(self.src_text)
            self.input_ids_src = [x.ids for x in encoded_input_src]

            encoded_input_tgt = self.tokenizer.encode_batch(self.tgt_text)
            self.input_ids_tgt = [x.ids for x in encoded_input_tgt]
        
        else:

            encoded_input_src = self.tokenizer(self.src_text)
            self.input_ids_src = encoded_input_src["input_ids"]

            encoded_input_tgt = self.tokenizer(self.tgt_text)
            self.input_ids_tgt = encoded_input_tgt["input_ids"]
        
        count_length_src = np.mean([len(item) for item in self.input_ids_src])
        count_length_tgt = np.mean([len(item) for item in self.input_ids_tgt])

        print(f'average number of tokens in source {count_length_src}')
        print(f'average number of tokens in target {count_length_tgt}')

    def __len__(self) -> int:
        return len(self.src_text)

    def __getitem__(self, i):
        out_dict = {
            "encoder_input_ids": self.input_ids_src[i],
            "decoder_input_ids": self.input_ids_tgt[i],
        }
        return out_dict

    @staticmethod
    def collate_pad(batch, args, cutoff: int, cutoff_src: int, padding_token: int):
        max_token_len_src, max_token_len_tgt = cutoff_src, cutoff
        num_elems = len(batch)

        tokens_src = torch.ones(num_elems, max_token_len_src).long() * padding_token
        tokens_mask_src = torch.zeros(num_elems, max_token_len_src).long()

        tokens_tgt = torch.ones(num_elems, max_token_len_tgt).long() * padding_token
        tokens_mask_tgt = torch.zeros(num_elems, max_token_len_tgt).long()

        for i in range(num_elems):
            toks_src = batch[i]["encoder_input_ids"][:max_token_len_src]
            toks_tgt = batch[i]["decoder_input_ids"][:max_token_len_tgt]
            l_s, l_t = len(toks_src), len(toks_tgt)
            tokens_src[i, :l_s] = torch.LongTensor(toks_src)
            tokens_tgt[i, :l_t] = torch.LongTensor(toks_tgt)
            tokens_mask_src[i, :l_s] = 1
            tokens_mask_tgt[i, :] = 1

        return {"input_ids": tokens_src, "attention_mask": tokens_mask_src, 
                    'decoder_input_ids': tokens_tgt, 'decoder_attention_mask': tokens_mask_tgt}, None
