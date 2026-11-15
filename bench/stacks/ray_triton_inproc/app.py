"""Ray Serve replica that embeds Triton via the in-process Python API (`tritonserver`).

Each replica is its own Ray worker process, so each owns one Triton core with ONE model
instance (REPLICAS maps to Ray replicas, not Triton instances). The repo is rendered per
replica into a temp dir so instance count and threads are always consistent with the config.
Runs only inside the NGC-based image (needs libtritonserver.so).
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import numpy as np
from ray import serve
from starlette.requests import Request

from bench.models.schema import DENSE_NAME, OUTPUT_NAME, SPARSE_NAME
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ray_runner import concurrency_args, handle, run_local
from bench.stacks.triton.repo import render_repo


@serve.deployment
class TritonInProc:
    def __init__(self, cfg: dict):
        import tritonserver  # only in the NGC image

        self.cfg = c = ServeConfig.from_args(cfg)
        repo = Path(tempfile.mkdtemp(prefix="triton_repo_"))
        render_repo(repo, c.arch, c.size, instances=1, batching=c.batching,
                    intra_threads=c.intra_threads, onnx_dir=Path(c.onnx_dir),
                    gpu=c.provider.startswith("CUDA"))
        self.server = tritonserver.Server(
            model_repository=str(repo),
            model_control_mode=tritonserver.ModelControlMode.NONE,
            log_error=True,
            metrics=False,
        ).start(wait_until_ready=True)
        self.model = self.server.model(c.model_name)

    async def _infer(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        async for resp in self.model.async_infer(inputs={DENSE_NAME: dense, SPARSE_NAME: sparse}):
            if resp.error:
                raise RuntimeError(str(resp.error))
            return np.from_dlpack(resp.outputs[OUTPUT_NAME])
        raise RuntimeError("no response from Triton")

    async def __call__(self, request: Request):
        return await handle(request, self.cfg, self._infer)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.server.stop()


def app_builder(args: dict):
    cfg = ServeConfig.from_args(args)
    return TritonInProc.options(**concurrency_args(cfg)).bind(cfg.as_dict())


if __name__ == "__main__":
    run_local(app_builder, ServeConfig.from_env())
