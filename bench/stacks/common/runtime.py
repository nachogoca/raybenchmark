"""The one place a Python stack turns (dense, sparse) into probabilities.

`ort`   ONNX Runtime on the exported .onnx file (the optimised-graph path).
`torch` PyTorch eager in inference mode, same architecture and seeded weights as the ONNX file
        (bench.models.build), so outputs match ORT to float tolerance.

Threads: `intra_threads` sets ORT intra-op threads / torch.set_num_threads, so the same core
budget applies to both runtimes. Stacks only call the returned runner; they never touch a runtime.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from bench.models.schema import load_spec
from bench.stacks.common.config import ServeConfig

Runner = Callable[[np.ndarray, np.ndarray], np.ndarray]


def make_runner(cfg: ServeConfig) -> Runner:
    if cfg.runtime == "ort":
        from bench.stacks.common.ort_session import make_session, predict

        sess = make_session(cfg.onnx_file, cfg.intra_threads, cfg.inter_threads, cfg.provider)
        return lambda dense, sparse: predict(sess, dense, sparse)
    if cfg.runtime == "torch":
        return _torch_runner(cfg)
    raise ValueError(f"unknown runtime {cfg.runtime}")


def _torch_runner(cfg: ServeConfig) -> Runner:
    import torch

    from bench.models.build import build

    torch.set_num_threads(cfg.intra_threads)
    try:
        torch.set_num_interop_threads(cfg.inter_threads)
    except RuntimeError:  # can only be set once per process (e.g. several replicas in one test)
        pass
    model = build(load_spec(cfg.arch, cfg.size))

    def run(dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(dense), torch.from_numpy(sparse)).numpy()

    return run
