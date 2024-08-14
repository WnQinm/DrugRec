import os
from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import TrainingArguments


# TODO json参数控制 https://blog.csdn.net/sjxgghg/article/details/131229480
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        default="./checkpoint/m3/",
        metadata={"help": "Path to pretrained model"}
    )
    tokenizer_path: Optional[str] = field(
        default="./checkpoint/m3/",
        metadata={"help": "Pretrained tokenizer path if not the same as model_name"},
    )
    normlized: bool = field(default=True, metadata={"help": ""})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: float = field(default=0.02, metadata={"help": ""})
    encode_sub_batch_size: int = field(default=1, metadata={"help": ""})
    train_with_fp16: bool = field(default=True, metadata={"help": ""})
    train_with_lora: bool = field(default=True, metadata={"help": ""})
    lora_modules: Union[str, List[str]] = field(
        default=None,
        metadata={"help": ""},
    )


@dataclass
class DataArguments:
    drug_data: str = field(
        default="./data/drugs.json",
        metadata={"help": r"format {drugbank_id: {names:[], description:''}}"},
    )
    pos2neg: str = field(
        default="./data/drugs_neg.json",
        metadata={"help": r"format {drugbank_id(pos): [drugbank_ids(neg)]}"},
    )
    link_data: str = field(
        default="./data/links.json",
        metadata={
            "help": r"all links that can be loaded by 'datasets.load_dataset('json', data_files=link_data, split='train')'"
            r"format [{entity1:drugbank_id, entity2:drugbank_id, description:str}]"
        },
    )
    train_group_size: int = field(default=8)
    input_max_len: int = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    query_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )


@dataclass
class TrainArguments(TrainingArguments):
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
