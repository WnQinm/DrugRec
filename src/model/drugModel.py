import torch
from torch import nn, Tensor
from .bgem3 import M3DenseEmbedModel
from ..utils.info_nce import InfoNCE
from typing import List


class Model(M3DenseEmbedModel):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.info_nce = InfoNCE(negative_mode="paired")

    def entity_embed_loss(self, head:Tensor, head_desc:Tensor, tail:Tensor, tail_desc:Tensor) -> Tensor:
        # (bs+bs, embed_size)
        q_dense_vecs = torch.cat([head[:, 0, :], tail[:, 0, :]], dim=0)
        # (bs+bs, group_size, embed_size)
        p_dense_vecs = torch.cat([head_desc, tail_desc], dim=0)

        idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)
        targets = idxs * (p_dense_vecs.size(0) // q_dense_vecs.size(0))
        dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
        loss = self.cross_entropy(dense_scores, targets)

        return loss

    def kg_embed_loss(self, head, link_desc, tail) -> List[Tensor]:
        head_pos = head[:, 0, :]
        head_neg = head[:, 1:, :]
        tail_pos = tail[:, 0, :]
        tail_neg = tail[:, 1:, :]

        return (self.info_nce(head_pos + link_desc, tail_pos, tail_neg),
                self.info_nce(tail_pos - link_desc, head_pos, head_neg))

    def forward(self, inputs):
        # torch.cuda.empty_cache()
        head, head_desc, link_desc, tail, tail_desc = inputs

        # (batch_size, group_size, embed_size)
        f = lambda x: torch.stack(list(map(self.encode, x)), dim=1)
        head, head_desc, tail, tail_desc = map(f, (head, head_desc, tail, tail_desc))
        # (batch_size, embed_size)
        link_desc = self.encode(link_desc)

        loss1 = self.entity_embed_loss(head, head_desc, tail, tail_desc)
        loss2, loss3 = self.kg_embed_loss(head, link_desc, tail)

        return (loss1 + loss2 + loss3) / 3