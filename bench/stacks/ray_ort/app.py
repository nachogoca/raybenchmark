"""Ray Serve + ONNX Runtime in the replica, optional @serve.batch across requests."""
from __future__ import annotations

import numpy as np
from ray import serve
from starlette.requests import Request

from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ray_runner import (
    concurrency_args,
    handle,
    run_local,
    split_rows,
    to_thread,
)
from bench.stacks.common.runtime import make_runner


@serve.deployment
class OrtModel:
    def __init__(self, cfg: dict):
        self.cfg = ServeConfig.from_args(cfg)
        c = self.cfg
        self.run = make_runner(c)
        self._batched.set_max_batch_size(c.batch_max)
        self._batched.set_batch_wait_timeout_s(c.batch_wait_ms / 1000.0)

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.005)
    async def _batched(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        dense = np.concatenate([p[0] for p in pairs])
        sparse = np.concatenate([p[1] for p in pairs])
        prob = await to_thread(self.run, dense, sparse)
        return split_rows(prob, [len(p[0]) for p in pairs])

    async def _infer(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        if self.cfg.batching:
            return await self._batched((dense, sparse))
        return await to_thread(self.run, dense, sparse)

    async def __call__(self, request: Request):
        return await handle(request, self.cfg, self._infer)


def app_builder(args: dict):
    cfg = ServeConfig.from_args(args)
    return OrtModel.options(**concurrency_args(cfg)).bind(cfg.as_dict())


if __name__ == "__main__":
    run_local(app_builder, ServeConfig.from_env())
