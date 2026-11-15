"""Deterministic PyTorch model construction. Importable without onnx/onnxruntime so the
PyTorch serving runtime (and the export script) share one definition and identical weights."""
from __future__ import annotations

import torch

from .dcnv2 import DCNv2
from .deepfm import DeepFM
from .schema import ModelSpec

SEED = 1234


def build(spec: ModelSpec) -> torch.nn.Module:
    torch.manual_seed(SEED)
    model = DeepFM(spec) if spec.arch == "deepfm" else DCNv2(spec)
    return model.eval()
