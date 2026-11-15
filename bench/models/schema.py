"""Criteo-style CTR input schema shared by models, codec, payloads and Triton configs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

N_DENSE = 13
N_SPARSE = 26
PAYLOAD_VOCAB = 10_000  # ids in payloads are < this, valid for every size

DENSE_NAME = "dense"    # float32 [B, 13]
SPARSE_NAME = "sparse"  # int64   [B, 26]
OUTPUT_NAME = "prob"    # float32 [B, 1]

MODELS = ("deepfm", "dcnv2")
SIZES = ("small", "large")

_SIZES_FILE = Path(__file__).with_name("sizes.yaml")


@dataclass(frozen=True)
class ModelSpec:
    arch: str
    size: str
    vocab_per_field: tuple[int, ...]
    embed_dim: int
    dnn_hidden: tuple[int, ...]
    cross_layers: int
    variant: int | None = None      # family member: same architecture, different seeded weights

    @property
    def name(self) -> str:
        base = f"{self.arch}_{self.size}"
        return base if self.variant is None else f"{base}_v{self.variant:02d}"

    @property
    def total_rows(self) -> int:
        return sum(self.vocab_per_field)


def load_spec(arch: str, size: str, variant: int | None = None) -> ModelSpec:
    if arch not in MODELS or size not in SIZES:
        raise ValueError(f"unknown model {arch}/{size}")
    cfg = yaml.safe_load(_SIZES_FILE.read_text())
    vocab = tuple(int(v) for v in cfg["sizes"][size]["vocab_per_field"])
    assert len(vocab) == N_SPARSE and min(vocab) >= PAYLOAD_VOCAB
    return ModelSpec(
        arch=arch,
        size=size,
        vocab_per_field=vocab,
        embed_dim=int(cfg["embed_dim"]),
        dnn_hidden=tuple(int(h) for h in cfg["dnn_hidden"]),
        cross_layers=int(cfg["cross_layers"]),
        variant=variant,
    )


def artifacts_dir(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[2]
    return root / "artifacts" / "onnx"


def onnx_path(spec: ModelSpec, root: Path | None = None) -> Path:
    return artifacts_dir(root) / f"{spec.name}.onnx"


def variant_names(arch: str, size: str, n: int) -> list[str]:
    """Names of an n-member family, hottest first: deepfm_small_v00 ... v{n-1}."""
    return [load_spec(arch, size, i).name for i in range(n)]
