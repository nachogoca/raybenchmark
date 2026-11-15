"""DCN-V2 (Wang et al. 2021), parallel structure: cross network || DNN -> linear."""
from __future__ import annotations

import torch
from torch import nn

from .common import FieldEmbedding, flat_input_dim, mlp
from .schema import ModelSpec


class CrossNetV2(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for lin in self.layers:
            x = x0 * lin(x) + x
        return x


class DCNv2(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        d0 = flat_input_dim(spec)
        self.embed = FieldEmbedding(spec.vocab_per_field, spec.embed_dim)
        self.cross = CrossNetV2(d0, spec.cross_layers)
        self.dnn = mlp(d0, spec.dnn_hidden)
        self.out = nn.Linear(d0 + spec.dnn_hidden[-1], 1)

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([self.embed(sparse).flatten(1), dense], dim=1)
        return torch.sigmoid(self.out(torch.cat([self.cross(x0), self.dnn(x0)], dim=1)))
