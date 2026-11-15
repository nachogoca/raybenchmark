"""The three stages of the pipeline scenario (F5) and the fan-out reduction (F3).

Every stack runs the SAME numpy code (Ray deployments, FastAPI in-process, Triton python
backend), so differences come from where the stages run and how they are connected.
"""
from __future__ import annotations

import numpy as np

from bench.models.schema import PAYLOAD_VOCAB

_MEAN, _STD = 1.0, 1.5
_HASH = np.int64(2654435761)


def featurize(dense: np.ndarray, sparse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense: log1p + standardise; sparse: multiplicative hash back into the id range."""
    d = ((np.log1p(np.clip(dense, 0.0, None)) - _MEAN) / _STD).astype(np.float32)
    s = ((sparse.astype(np.int64) * _HASH) % PAYLOAD_VOCAB).astype(np.int64)
    return d, s


def postprocess(prob: np.ndarray) -> np.ndarray:
    """Platt-style recalibration of a probability."""
    p = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return (1.0 / (1.0 + np.exp(-(1.1 * logit - 0.05)))).astype(np.float32)


def average(probs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs), axis=0).astype(np.float32)


def fanout_models(family: list[str], primary: str, k: int) -> list[str]:
    """The K members scoring one request: the routed variant and the next K-1 (wrapping)."""
    i = family.index(primary)
    return [family[(i + j) % len(family)] for j in range(k)]
