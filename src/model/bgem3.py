from typing import Dict, Union, List

from ..utils.arguments import ModelArguments

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer
from transformers.tokenization_utils import BatchEncoding


class M3DenseEmbedModel(nn.Module):

    def __init__(self, model_args: ModelArguments):
        super().__init__()
        self.load_model(model_args)

        self.normlized = model_args.normlized
        self.temperature = model_args.temperature
        self.sub_batch_size = model_args.encode_sub_batch_size

        if not model_args.normlized:
            self.temperature = 1.0

    @property
    def device(self):
        return self.model.device

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(**kwargs)

    def load_model(self, model_load_args:ModelArguments):

        torch_dtype = torch.float32
        if model_load_args.model_with_fp16:
            torch_dtype = torch.half
        self.model = AutoModel.from_pretrained(
            model_load_args.model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch_dtype,
            add_pooling_layer=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_args.tokenizer_path)
        self.model: XLMRobertaModel
        self.tokenizer: XLMRobertaTokenizer

        if model_load_args.train_with_lora:
            if model_load_args.lora_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, model_load_args.lora_path)
            else:
                from peft import LoraConfig, TaskType, get_peft_model
                config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, target_modules=model_load_args.lora_modules)
                self.model = get_peft_model(self.model, config)

            if model_load_args.lora_with_fp16:
                for name, param in self.model.named_parameters():
                    if "lora" in name:
                        param.data = param.data.half()
                        if param.grad is not None and param.grad.data is not None:
                            param.grad.data = param.grad.data.half()

    def _encode(self, features) -> Tensor:
        dense_vecs = None
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = last_hidden_state[:, 0]
        if self.normlized:
            dense_vecs = F.normalize(dense_vecs, dim=-1)
        return dense_vecs

    def encode(self, features: BatchEncoding=None) -> Tensor:
        if features is None:
            return None

        if self.sub_batch_size is not None and self.sub_batch_size != -1:
            all_dense_vecs = []
            for i in range(0, len(features['attention_mask']), self.sub_batch_size):
                end_inx = min(i + self.sub_batch_size, len(features['attention_mask']))
                sub_features = {}
                for k, v in features.items():
                    sub_features[k] = v[i:end_inx]
                all_dense_vecs.append(self._encode(sub_features))
            dense_vecs = torch.cat(all_dense_vecs, 0)
        else:
            dense_vecs = self._encode(features)

        return dense_vecs.contiguous()

    def compute_similarity(self, q_reps:Tensor, p_reps:Tensor) -> Tensor:
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def save(self, output_dir: str) -> None:
        _trans_state_dict = lambda state_dict: type(state_dict)({k: v.clone().cpu() for k,v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(self.model.state_dict()))


class M3ForInference(M3DenseEmbedModel):
    def __init__(
        self,
        model_load_args: ModelArguments = None,
        use_fp16: bool = False
    ):
        super().__init__(model_load_args)
        if not torch.cuda.is_available():
            raise NotImplementedError("gpu not found")
        if use_fp16: self.model.half()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

    def __call__(
        self,
        sentences: Union[List[str], str],
        max_length: int = 8192,
    ) -> Tensor:
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)
        all_embeddings = self.encode(inputs)

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


class M3ForScore(M3DenseEmbedModel):
    def __init__(
        self,
        model_load_args: ModelArguments,
        use_fp16: bool = True
    ):
        super().__init__(model_args=model_load_args)
        if not torch.cuda.is_available():
            raise NotImplementedError("gpu not found")
        if use_fp16: self.model.half()
        self.model.eval()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.max_length = 8192

    def select_topk(self, query: Union[str, Tensor], documents: List[str], k=1) -> torch.Tensor:
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        if isinstance(query, str):
            query = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            query = self.encode(query)

        if isinstance(query, Tensor) and len(query.shape)==1:
            query = query.unsqueeze_(dim=0)

        documents = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        documents = self.encode(documents)

        scores = self.dense_score(query, documents).squeeze_()
        return scores.topk(min(k, len(scores))).indices

    def __call__(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        torch.cuda.empty_cache()
        texts = [item['text'] for item in paragraphs]
        topk = self.select_topk(query, texts, topk)
        indices = topk.detach().cpu().numpy().tolist()
        return [paragraphs[int(idx)] for idx in indices]
