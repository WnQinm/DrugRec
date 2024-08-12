import torch
import random
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments):
        self.drug_data = self._load_json(args.drug_data)
        self.pos2neg = self._load_json(args.pos2neg)
        self.link_data = datasets.load_dataset("json", data_files=args.link_data, split="train")
        self.args = args
        self.total_len = len(self.link_data)

    def _load_json(self, file_path) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)

    def __len__(self):
        return self.total_len

    def _fetch_data(self, target, mode:str):
        if mode == "entity":
            neg_set = [random.choice(self.drug_data[neg]["names"])
                       for neg in random.sample(self.pos2neg[target], self.args.train_group_size - 1)]
            return [target] + neg_set
        elif mode.startswith("desc"):
            neg_set = [self.drug_data[neg]["description"]
                       for neg in random.sample(self.pos2neg[target], self.args.train_group_size - 1)]
            return [self.drug_data[target]["description"]] + neg_set
        else:
            raise NotImplementedError

    def __getitem__(self, item) -> Tuple[List[str], List[str], str, List[str], List[str]]:
        link = self.link_data[item]
        head, tail, link_desc = link["entity1"], link["entity2"], link["description"]

        assert 0 < self.args.train_group_size < len(self.pos2neg[head])
        assert 0 < self.args.train_group_size < len(self.pos2neg[tail])

        head_desc = self._fetch_data(head, "desc")
        tail_desc = self._fetch_data(tail, "desc")
        head = self._fetch_data(head, "entity")
        tail = self._fetch_data(tail, "entity")
        head[0] = self.drug_data[head[0]]["names"][0]
        tail[0] = self.drug_data[tail[0]]["names"][0]

        return head, head_desc, link_desc, tail, tail_desc


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 1024
    passage_max_len: int = 8192

    def tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )

    def sort_neg(self, batch:List[Tuple[str]]) -> List[Tuple[str]]:
        '''
        将负样本部分按长度排好序 可以一定程度减少padding的长度
        '''
        return [batch[0]] + list(zip(*[sorted(i, key=len) for i in zip(*batch[1:])]))

    def __call__(self, features):
        # [bathc_size, ] * (pos + (group_size-1)*neg)
        head = self.sort_neg(features[0])
        head_desc = self.sort_neg(features[1])
        link_desc = features[2]
        tail = self.sort_neg(features[3])
        tail_desc = self.sort_neg(features[4])

        head = list(map(self.tokenize, head))
        head_desc = list(map(self.tokenize, head_desc))
        link_desc = self.tokenize(link_desc)
        tail = list(map(self.tokenize, tail))
        tail_desc = list(map(self.tokenize, tail_desc))

        return head, head_desc, link_desc, tail, tail_desc
