"""The ONE place ONNX Runtime session options are set. Triton's config.pbtxt mirrors these."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bench.models.schema import DENSE_NAME, OUTPUT_NAME, SPARSE_NAME

GRAPH_OPT = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def env_threads() -> tuple[int, int]:
    """(intra_op, inter_op) from env; defaults: 1 intra (replicas own cores), 1 inter."""
    return int(os.environ.get("ORT_INTRA_THREADS", "1")), int(os.environ.get("ORT_INTER_THREADS", "1"))


def make_session(path: str | Path, intra: int | None = None, inter: int | None = None,
                 provider: str | None = None) -> ort.InferenceSession:
    e_intra, e_inter = env_threads()
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra or e_intra
    so.inter_op_num_threads = inter or e_inter
    so.graph_optimization_level = GRAPH_OPT
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    provider = provider or os.environ.get("ORT_PROVIDER", "CPUExecutionProvider")
    return ort.InferenceSession(str(path), so, providers=[provider])


def predict(sess: ort.InferenceSession, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
    return sess.run([OUTPUT_NAME], {DENSE_NAME: dense, SPARSE_NAME: sparse})[0]
