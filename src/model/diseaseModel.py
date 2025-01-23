import os
import json
import torch
from torch import nn, Tensor
from .bgem3 import M3DenseEmbedModel
from ..utils.arguments import ModelArguments
from typing import Union


class Model(M3DenseEmbedModel):
    def __init__(self, model_args:ModelArguments, features: Union[Tensor, os.PathLike]):
        super().__init__(model_args)
        self.cache_dir = model_args.cache_dir
        self._init_features(features)
        self._init_P()

    def _init_features(self, origin_features: Union[Tensor, os.PathLike]):
        if not hasattr(self, "features"):
            if isinstance(origin_features, Tensor):
                self.register_buffer("features", origin_features)
            elif os.path.exists(origin_features):
                self.register_buffer("features", torch.load(origin_features, weights_only=False))
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

    def forward(self, inputs):
        idx, x = inputs
        # (bs, embed_dim)
        x = self.encode(x)
        self.features[idx] = x
        px = torch.mm(self.P, self.features)
        score = torch.diagonal(self.dense_score(x, px[idx]))
        return -torch.sum(score)