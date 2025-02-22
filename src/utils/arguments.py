from dataclasses import dataclass, field
from typing import Optional, Union, List
import os

from transformers import TrainingArguments


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
    temperature: float = field(default=0.02, metadata={"help": ""})
    encode_sub_batch_size: int = field(default=1, metadata={"help": ""})
    cache_dir: str = field(
        default="./cache",
        metadata={
            "help": """
            'disease_features.pt':
                - 疾病预先计算好的特征向量 每个疾病对应一个嵌入向量 (disease_num, embed_dim)
                - 嵌入向量来源于疾病描述/常见症状等描述性文本
                - 此处暂时使用mimic iv中病人症状按seq_num加权计算而得
            """
        },
    )

    model_with_fp16: bool = field(default=False, metadata={"help": ""})
    lora_with_fp16: bool = field(default=False, metadata={"help": ""})
    train_with_lora: bool = field(default=False, metadata={"help": ""})
    lora_path: Optional[str] = field(default=None, metadata={"help": "lora weight path"})
    lora_modules: Union[str, List[str]] = field(
        default=None,
        metadata={"help": ""},
    )

    def __post_init__(self):
        if "adapter_config.json" in os.listdir(self.model_path):
            from peft import LoraConfig
            self.train_with_lora = True
            self.lora_path = self.model_path
            self.model_path = LoraConfig.from_pretrained(self.lora_path).base_model_name_or_path


@dataclass
class DataArguments:
    # drug arg
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

    # disease arg
    all_disease_list: str = field(
        default="./data/all_disease.csv",
        metadata={"help": ""}
    )
    mimic_disease_data: str = field(
        default="./data/icd2text.json"
    )

    input_max_len: int = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class TrainArguments(TrainingArguments):
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    training_goal: str = field(
        default="disease", metadata={"help": "drug, disease, mimic"}
    )
