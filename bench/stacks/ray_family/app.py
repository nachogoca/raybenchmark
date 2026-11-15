"""Ray Serve model-family stack: every Ray-native way to serve N related models.

FAMILY_MODE (env / Serve arg `family_mode`):
  mux          one deployment, `@serve.multiplexed` LRU of loaded sessions per replica; model id
               comes from the URL (and the `serve_multiplexed_model_id` header, for affinity).
  deployments  one deployment per variant (each autoscaled 1..2) behind a router ingress.
  fanout       router -> K x MuxModel via DeploymentHandle.options(multiplexed_model_id=...),
               scores averaged (ensemble across variants).
  pipeline     router -> Feature deployment -> MuxModel (routed variant) -> postprocess in router.
"""
from __future__ import annotations

import asyncio

import numpy as np
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bench.protocol.kserve_v2 import ProtocolError, decode_request, encode_response
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.pipeline import average, fanout_models, featurize, postprocess
from bench.stacks.common.ray_runner import concurrency_args, run_local, to_thread


def _runner_for(cfg: ServeConfig, name: str):
    from bench.stacks.common.ort_session import make_session, predict

    sess = make_session(f"{cfg.onnx_dir}/{name}.onnx", cfg.intra_threads, cfg.inter_threads, cfg.provider)
    return lambda d, s: predict(sess, d, s)


def _parse(path: str, family: list[str]) -> tuple[str, str]:
    """-> (kind, model): kind in healthz | ready | infer | bad."""
    if path == "/healthz":
        return "healthz", ""
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "v2" and parts[1] == "models" and parts[2] in family:
        if len(parts) == 4 and parts[3] == "infer":
            return "infer", parts[2]
        if len(parts) == 4 and parts[3] == "ready":
            return "ready", parts[2]
    return "bad", ""


async def _http(request: Request, cfg: ServeConfig, family: list[str], infer) -> Response:
    kind, model = _parse(request.url.path, family)
    if kind == "healthz":
        return JSONResponse({"status": "ok", "family": len(family), "mode": cfg.family_mode})
    if kind == "ready":
        return Response(status_code=200)
    if kind != "infer":
        return Response(status_code=404)
    try:
        dense, sparse = decode_request(await request.body())
    except ProtocolError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    prob = await infer(model, dense, sparse)
    return Response(encode_response(model, prob), media_type="application/json")


def make_mux_class(models_per_replica: int):
    """Factory: the multiplexed LRU size is a decorator argument, so the class is built per config."""

    @serve.deployment
    class MuxModel:
        def __init__(self, cfg: dict):
            self.cfg = ServeConfig.from_args(cfg)
            self.family = self.cfg.family

        @serve.multiplexed(max_num_models_per_replica=models_per_replica)
        async def get_model(self, model_id: str):
            return await asyncio.to_thread(_runner_for, self.cfg, model_id)

        async def predict(self, dense: np.ndarray, sparse: np.ndarray, model_id: str) -> np.ndarray:
            run = await self.get_model(model_id)
            return await to_thread(run, dense, sparse)

        async def _ingress(self, model: str, dense, sparse) -> np.ndarray:
            return await self.predict(dense, sparse, model)

        async def __call__(self, request: Request):          # ingress mode (FAMILY_MODE=mux)
            return await _http(request, self.cfg, self.family, self._ingress)

    return MuxModel


@serve.deployment
class VariantModel:
    def __init__(self, cfg: dict, name: str):
        self.cfg = ServeConfig.from_args(cfg)
        self.run = _runner_for(self.cfg, name)

    async def predict(self, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        return await to_thread(self.run, dense, sparse)


@serve.deployment
class Feature:
    async def transform(self, dense: np.ndarray, sparse: np.ndarray):
        return featurize(dense, sparse)


@serve.deployment
class Router:
    def __init__(self, cfg: dict, models, feature=None):
        self.cfg = ServeConfig.from_args(cfg)
        self.family = self.cfg.family
        self.models, self.feature = models, feature       # handle (mux/fanout/pipeline) or dict

    async def _score(self, model: str, dense, sparse) -> np.ndarray:
        if self.cfg.family_mode == "deployments":
            return await self.models[model].predict.remote(dense, sparse)
        h = self.models.options(multiplexed_model_id=model)
        return await h.predict.remote(dense, sparse, model)

    async def infer(self, model: str, dense, sparse) -> np.ndarray:
        mode = self.cfg.family_mode
        if mode == "fanout":
            members = fanout_models(self.family, model, self.cfg.fanout_k)
            probs = await asyncio.gather(*(self._score(m, dense, sparse) for m in members))
            return average(list(probs))
        if mode == "pipeline":
            d, s = await self.feature.transform.remote(dense, sparse)
            return postprocess(await self._score(model, d, s))
        return await self._score(model, dense, sparse)

    async def __call__(self, request: Request):
        return await _http(request, self.cfg, self.family, self.infer)


def app_builder(args: dict):
    cfg = ServeConfig.from_args(args)
    mode = cfg.family_mode or "mux"
    cfg.family_mode = mode
    plain = {"num_cpus": 0}
    router_opts = {"max_ongoing_requests": 256, "ray_actor_options": plain}
    if mode == "mux":
        return make_mux_class(cfg.mux_models_per_replica).options(**concurrency_args(cfg)).bind(cfg.as_dict())
    if mode == "deployments":
        per = max(min(0.25, cfg.total_cpus / (2 * cfg.variants)), 0.05)
        variants = {
            name: VariantModel.options(
                name=name, num_replicas=1, max_ongoing_requests=cfg.max_ongoing,
                ray_actor_options={"num_cpus": per}).bind(cfg.as_dict(), name)
            for name in cfg.family
        }
        return Router.options(**router_opts).bind(cfg.as_dict(), variants)
    mux = make_mux_class(cfg.mux_models_per_replica).options(**concurrency_args(cfg)).bind(cfg.as_dict())
    feature = Feature.options(num_replicas=1, ray_actor_options=plain, max_ongoing_requests=256).bind() \
        if mode == "pipeline" else None
    return Router.options(**router_opts).bind(cfg.as_dict(), mux, feature)


if __name__ == "__main__":
    run_local(app_builder, ServeConfig.from_env())
