"""Ray Serve with a FastAPI app as ingress (`@serve.ingress`): the usual "Ray + FastAPI" setup.

Same ORT session, batching and threading as `ray_ort`; only the HTTP layer differs. `ray_ort`
answers raw Starlette requests directly, this stack routes them through FastAPI inside the
replica, so the gap between the two is the cost of FastAPI's routing/validation layer on Ray.
"""

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from ray import serve

from bench.protocol.kserve_v2 import ProtocolError, decode_request, encode_response
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ray_runner import concurrency_args, run_local, split_rows, to_thread
from bench.stacks.common.runtime import make_runner

api = FastAPI()


@serve.deployment
@serve.ingress(api)
class FastApiOrtModel:
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

    @api.get("/healthz")
    async def healthz(self) -> dict:
        return {"status": "ok", "model": self.cfg.model_name}

    @api.get("/v2/models/{name}/ready")
    async def ready(self, name: str) -> Response:
        return Response(status_code=200 if name == self.cfg.model_name else 404)

    @api.post("/v2/models/{name}/infer")
    async def infer(self, name: str, request: Request) -> Response:
        if name != self.cfg.model_name:
            raise HTTPException(404, f"model {name} not served")
        try:
            dense, sparse = decode_request(await request.body())
        except ProtocolError as e:
            raise HTTPException(400, str(e)) from e
        prob = await self._infer(dense, sparse)
        return Response(encode_response(self.cfg.model_name, prob), media_type="application/json")


def app_builder(args: dict):
    cfg = ServeConfig.from_args(args)
    return FastApiOrtModel.options(**concurrency_args(cfg)).bind(cfg.as_dict())


if __name__ == "__main__":
    run_local(app_builder, ServeConfig.from_env())
