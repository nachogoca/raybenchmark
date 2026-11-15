"""Shared embedding layer: 26 fields, one table per field, offsets applied once."""
from __future__ import annotations

import torch
from torch import nn

from .schema import N_DENSE, N_SPARSE, ModelSpec


class FieldEmbedding(nn.Module):
    """One concatenated table; field offsets are a registered buffer so ONNX gets a single Gather."""

    def __init__(self, vocab_per_field: tuple[int, ...], dim: int):
        super().__init__()
        offsets = torch.tensor([0] + list(vocab_per_field[:-1]), dtype=torch.int64).cumsum(0)
        self.register_buffer("offsets", offsets)
        self.table = nn.Embedding(sum(vocab_per_field), dim)
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, sparse: torch.Tensor) -> torch.Tensor:  # [B,26] -> [B,26,d]
        return self.table(sparse + self.offsets)


def mlp(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers)


def flat_input_dim(spec: ModelSpec) -> int:
    return N_SPARSE * spec.embed_dim + N_DENSE
