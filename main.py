from transformers import HfArgumentParser
from .src.utils.arguments import ModelArguments, DataArguments, TrainingArguments

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: ModelArguments
data_args: DataArguments
training_args: TrainingArguments

