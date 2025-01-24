import random
import json
from dataclasses import dataclass
from typing import List, Tuple

import datasets
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import BatchEncoding

from .arguments import DataArguments


class DrugDataset(Dataset):
    def __init__(self, args: DataArguments):
        super().__init__()
        self.drug_data = self._load_json(args.drug_data)
        self.pos2neg = self._load_json(args.pos2neg)
        self.link_data = datasets.load_dataset("json", data_files=args.link_data, split="train")
        self.args = args
        self.total_len = len(self.link_data)

    def _load_json(self, file_path) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
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
class DrugCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[dataset output]] to List[output1], List[output2]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    input_max_len: int = 8192

    def tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=self.input_max_len,
            return_tensors="pt",
        )

    def sort_neg(self, batch:List[Tuple[str]]) -> List[Tuple[str]]:
        '''
        将负样本部分按长度排好序 可以一定程度减少padding的长度
        '''
        return [batch[0]] + list(*zip(*[sorted(i, key=len) for i in zip(batch[1:])]))

    def __call__(self, features):
        features = features[0]
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


class DiseaseDataset(Dataset):
    def __init__(self, args: DataArguments):
        super().__init__()
        self.all_disease = pd.read_csv(args.all_disease_list)
        self.disease_data = self._load_json(args.mimic_disease_data)
        self.select_num = 8

    def _load_json(self, file_path) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.all_disease)

    def __getitem__(self, index):
        disease = self.all_disease.iloc[index]
        disease_data = self.disease_data[disease['icd_code']]
        if len(disease_data) > self.select_num:
            disease_data = sorted(disease_data, key=lambda x: eval(x["weight"]))[:self.select_num]
        return (
            index,
            disease["icd_code"] + " " + disease["long_title"],
            [d["ChiefComplaint"] for d in disease_data],
            [d["weight"] for d in disease_data],
        )


@dataclass
class DiseaseCollator(DataCollatorWithPadding):
    input_max_len: int = 8192

    def tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=self.input_max_len,
            return_tensors="pt",
        )

    def __call__(self, features) -> Tuple[List[int], BatchEncoding, List[BatchEncoding], List[torch.Tensor]]:
        idx, d_name, d_desc, d_weight = zip(*features)
        d_name = self.tokenize(d_name)
        d_desc = list(map(self.tokenize, d_desc))
        # (("1/8", "6/7", ...), ...)
        d_weight = [
            softmax(torch.tensor([1 - eval(w) for w in dw]), dim=0).unsqueeze_(1)
            for dw in d_weight
        ]
        return idx, d_name, d_desc, d_weight
