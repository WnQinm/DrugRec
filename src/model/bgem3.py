from dataclasses import dataclass
from typing import Dict, Optional, Union, List
import os

from ..utils.arguments import ModelArguments

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer
from transformers.file_utils import ModelOutput


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class M3DenseEmbedModel(nn.Module):

    def __init__(
        self,
        model_load_args: ModelArguments = None,
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        enable_sub_batch: bool = True,
    ):
        super().__init__()
        self.load_model(model_load_args)
        self.vocab_size = self.model.config.vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.enable_sub_batch = enable_sub_batch
        self.temperature = temperature

        if not normlized:
            self.temperature = 1.0

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def load_model(self, model_load_args:ModelArguments):
        if not os.path.exists(model_load_args.model_path):
            raise FileNotFoundError(f"cannot find model {model_load_args.model_path}")

        self.model:XLMRobertaModel = AutoModel.from_pretrained(model_load_args.model_path)
        tokenizer_path = model_load_args.tokenizer_path if model_load_args.tokenizer_path is not None else model_load_args.model_path
        self.tokenizer:XLMRobertaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _encode(self, features) -> Tensor:
        dense_vecs = None
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = self.dense_embedding(last_hidden_state, features['attention_mask'])
        if self.normlized:
            dense_vecs = F.normalize(dense_vecs, dim=-1)
        return dense_vecs

    def encode(self, features: Dict[str, Tensor]=None, sub_batch_size=None) -> Tensor:
        if features is None:
            return None

        if sub_batch_size is not None and sub_batch_size != -1:
            all_dense_vecs = []
            for i in range(0, len(features['attention_mask']), sub_batch_size):
                end_inx = min(i + sub_batch_size, len(features['attention_mask']))
                sub_features = {}
                for k, v in features.items():
                    sub_features[k] = v[i:end_inx]
                all_dense_vecs.append(self._encode(sub_features))
            dense_vecs = torch.cat(all_dense_vecs, 0)
        else:
            dense_vecs = self._encode(features)

        return dense_vecs.contiguous()

    def compute_sub_batch_size(self, features) -> int:
        mapping = [(6000, 1), (5000, 2), (4000, 3), (3000, 3), (2000, 5), (1000, 9), (512, 16), (0, 32)]
        cur_l = features['input_ids'].size(-1)
        for l, b in mapping:
            if cur_l >= l:
                return b

    def compute_similarity(self, q_reps:Tensor, p_reps:Tensor):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    # trainer的data_collator输出的dict键值要匹配forward的输入
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        self.model.train()

        if self.enable_sub_batch:
            q_dense_vecs = self.encode(query, sub_batch_size=self.compute_sub_batch_size(query))
            p_dense_vecs = self.encode(passage, sub_batch_size=self.compute_sub_batch_size(passage))
        else:
            q_dense_vecs = self.encode(query)
            p_dense_vecs = self.encode(passage)

        if self.negatives_cross_device:
            cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
            cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)

            cross_idxs = torch.arange(cross_q_dense_vecs.size(0), device=cross_q_dense_vecs.device, dtype=torch.long)

            cross_targets = cross_idxs * (cross_p_dense_vecs.size(0) // cross_q_dense_vecs.size(0))
            cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

            loss = self.compute_loss(cross_dense_scores, cross_targets)
        else:
            idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)

            targets = idxs * (p_dense_vecs.size(0) // q_dense_vecs.size(0))
            dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
            loss = self.compute_loss(dense_scores, targets)

        return EncoderOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[Tensor]) -> Tensor:
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str) -> None:
        _trans_state_dict = lambda state_dict: type(state_dict)({k: v.clone().cpu() for k,v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(self.model.state_dict()))


class M3ForInference(M3DenseEmbedModel):
    def __init__(
        self,
        model_load_args: ModelArguments = None,
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 1.0,
        use_fp16: bool = True,
        device: str = "cpu"
    ):
        super().__init__(
            model_load_args=model_load_args,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
            negatives_cross_device=False,
            temperature=temperature,
            enable_sub_batch=False,
        )
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def __call__(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 8192,
    ) -> Tensor:
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            batch_size = None
            input_was_string = True

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)
        all_embeddings = self.encode(inputs, batch_size).detach().cpu()

        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


class M3ForScore(M3DenseEmbedModel):
    def __init__(
        self,
        model_load_args: ModelArguments,
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 1.0,
        use_fp16: bool = True,
        device: str = "cpu",
        batch_size: int = 512
    ):
        super().__init__(
            model_load_args=model_load_args,
            normlized=normlized,
            sentence_pooling_method=sentence_pooling_method,
            negatives_cross_device=False,
            temperature=temperature,
            enable_sub_batch=False,
        )
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.batch_size = batch_size
        self.max_length = 8192

    def encode(self, features: Dict[str, Tensor]=None, sub_batch_size=None) -> Tensor:
        torch.cuda.empty_cache()
        return super().encode(features, sub_batch_size)

    def select_topk(self, query: str, documents: List[str], k=1) -> torch.Tensor:
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        query = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        documents = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

        query = self.encode(query)
        documents = self.encode(documents, self.batch_size)

        scores = self.dense_score(query, documents)
        return scores.topk(min(k, len(scores))).indices

    def __call__(self, query, paragraphs: List[Dict[str, str]], topk=5) -> List[Dict[str, str]]:
        texts = [item['text'] for item in paragraphs]
        topk = self.select_topk(query, texts, topk)
        indices = list(topk.detach().cpu().numpy())
        return [paragraphs[int(idx)] for idx in indices]
