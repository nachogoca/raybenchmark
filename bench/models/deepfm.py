"""DeepFM (Guo et al. 2017): linear + FM second-order + DNN, shared embeddings."""
from __future__ import annotations

import torch
from torch import nn

from .common import FieldEmbedding, flat_input_dim, mlp
from .schema import N_DENSE, ModelSpec


class DeepFM(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.embed = FieldEmbedding(spec.vocab_per_field, spec.embed_dim)
        self.linear_sparse = FieldEmbedding(spec.vocab_per_field, 1)
        self.linear_dense = nn.Linear(N_DENSE, 1)
        self.dnn = mlp(flat_input_dim(spec), spec.dnn_hidden)
        self.dnn_out = nn.Linear(spec.dnn_hidden[-1], 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        e = self.embed(sparse)                                   # [B,26,d]
        first = self.linear_sparse(sparse).sum(dim=1) + self.linear_dense(dense)  # [B,1]
        sq_of_sum = e.sum(dim=1).pow(2)
        sum_of_sq = e.pow(2).sum(dim=1)
        fm = 0.5 * (sq_of_sum - sum_of_sq).sum(dim=1, keepdim=True)  # [B,1]
        x = torch.cat([e.flatten(1), dense], dim=1)
        deep = self.dnn_out(self.dnn(x))
        return torch.sigmoid(first + fm + deep)
