"""Ray Serve ingress → Triton (separate process) over async gRPC. Batching is Triton's job."""
from __future__ import annotations

import asyncio
import contextlib
import time

import numpy as np
import tritonclient.grpc.aio as tgrpc
from ray import serve
from starlette.requests import Request

from bench.models.schema import DENSE_NAME, OUTPUT_NAME, SPARSE_NAME
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ray_runner import concurrency_args, handle, run_local


@serve.deployment
class TritonProxy:
    def __init__(self, cfg: dict):
        self.cfg = ServeConfig.from_args(cfg)
        # __init__ already runs inside Ray's event loop, so it cannot block on a coroutine;
        # build the client here and wait for Triton on first use (see _ensure_ready).
        self.client = tgrpc.InferenceServerClient(self.cfg.triton_url)
        self._ready = False
        self._ready_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            deadline = time.time() + 300
            while not self._ready:
                with contextlib.suppress(Exception):
                    if await self.client.is_model_ready(self.cfg.model_name):
                        self._ready = True
                        return
                if time.time() > deadline:
                    raise RuntimeError(f"Triton at {self.cfg.triton_url} not ready")
                await asyncio.sleep(1)

    async def _infer(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        await self._ensure_ready()
        d =tgrpc.InferInput(DENSE_NAME, list(dense.shape), "FP32")
        d.set_data_from_numpy(dense)
        s = tgrpc.InferInput(SPARSE_NAME, list(sparse.shape), "INT64")
        s.set_data_from_numpy(sparse)
        res = await self.client.infer(self.cfg.model_name, [d, s],
                                      outputs=[tgrpc.InferRequestedOutput(OUTPUT_NAME)])
        return res.as_numpy(OUTPUT_NAME)

    async def __call__(self, request: Request):
        return await handle(request, self.cfg, self._infer)


def app_builder(args: dict):
    cfg = ServeConfig.from_args(args)
    return TritonProxy.options(**concurrency_args(cfg)).bind(cfg.as_dict())


if __name__ == "__main__":
    run_local(app_builder, ServeConfig.from_env())
