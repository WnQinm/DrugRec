import os
import json
import torch
from torch import Tensor
from transformers.tokenization_utils import BatchEncoding
from .bgem3 import M3DenseEmbedModel
from ..utils.arguments import ModelArguments
from typing import Union, List, Tuple


class Model(M3DenseEmbedModel):
    def __init__(self, model_args:ModelArguments):
        super().__init__(model_args)
        self.cache_dir = model_args.cache_dir
        self._init_features()
        self._init_P()

    def _init_features(self):
        feature_path = os.path.join(self.cache_dir, "disease_features.pt")
        if not hasattr(self, "features"):
            if os.path.exists(feature_path):
                self.register_buffer("features", torch.load(feature_path, weights_only=False))
            else:
                raise NotImplementedError(f"features not found")
        self.features: Tensor

    @staticmethod
    def normalize_adj_matrix(adj: Tensor):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        degree = torch.sum(adj, dim=1)

        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return normalized_adj

    def _init_P(self):
        adj_path = os.path.join(self.cache_dir, "adj.pt")
        graph_data_path = os.path.join(self.cache_dir, "disease_graph.json")
        if os.path.exists(adj_path):
            adj = torch.load(adj_path, map_location=self.device, weights_only=False)
        elif os.path.exists(graph_data_path):
            with open(graph_data_path, encoding='utf-8') as f:
                graph_data = json.load(f)
            adj = torch.zeros((len(graph_data['nodes']), len(graph_data['nodes'])), device=self.device)
            for link in graph_data['links']:
                adj[link['source'], link['target']] = link['weight']
                adj[link['target'], link['source']] = link['weight']
            adj = self.normalize_adj_matrix(adj)
            torch.save(adj, adj_path)
        else:
            raise FileNotFoundError(f"[{adj_path}] or [{graph_data_path}] not found")

        mask = (1 - torch.eye(adj.shape[0], device=self.device))
        self.register_buffer('P', mask * adj)
        self.P: Tensor

    def encode(self, features: Union[BatchEncoding, List[BatchEncoding]], weights: List[Tensor]=None) -> Tensor:
        if isinstance(features, BatchEncoding):
            return super(Model, self).encode(features)
        elif isinstance(features, list) and weights is not None:
            # p: (feature: BatchEncoding, weight: Tensor)
            aggregate = lambda p: torch.sum(super(Model, self).encode(p[0]) * p[1], dim=0)
            return torch.stack(list(map(aggregate, zip(features, weights))))
        else:
            raise NotImplementedError

    def entity_embed_loss(self, entity, desc):
        return -torch.mean(torch.diagonal(self.dense_score(entity, desc)))

    def kg_embed_loss(self, idx, desc):
        self.features = self.features.detach()

        self.features[idx] = desc
        px = torch.mm(self.P, self.features)
        return -torch.mean(torch.diagonal(self.dense_score(desc, px[idx])))

    def forward(self, inputs:Tuple[List[int], BatchEncoding, List[BatchEncoding], List[torch.Tensor]]):
        # (bs, ) (bs, ) (bs, desc_len) (bs, desc_len)
        idx, icd_name, desc, weight = inputs
        # (bs, embed_dim)
        icd_name = self.encode(icd_name)
        desc = self.encode(desc, weight)
        # TODO loss大小可能有影响
        return (self.entity_embed_loss(icd_name, desc) + self.kg_embed_loss(idx, desc)) / 2