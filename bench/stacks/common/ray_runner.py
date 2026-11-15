"""Shared Ray Serve plumbing: health/infer routing inside a replica and the local runner."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

import numpy as np
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bench.protocol.kserve_v2 import ProtocolError, decode_request, encode_response
from bench.stacks.common.config import ServeConfig

Infer = Callable[[np.ndarray, np.ndarray], Awaitable[np.ndarray]]


async def handle(request: Request, cfg: ServeConfig, infer: Infer) -> Response:
    path = request.url.path
    if path == "/healthz":
        return JSONResponse({"status": "ok", "model": cfg.model_name})
    if path == f"/v2/models/{cfg.model_name}/ready":
        return Response(status_code=200)
    if path != cfg.infer_path:
        return Response(status_code=404)
    try:
        dense, sparse = decode_request(await request.body())
    except ProtocolError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    prob = await infer(dense, sparse)
    return Response(encode_response(cfg.model_name, prob), media_type="application/json")


def split_rows(prob: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    out, i = [], 0
    for n in sizes:
        out.append(prob[i:i + n])
        i += n
    return out


def run_local(app_builder: Callable[[dict], object], cfg: ServeConfig) -> None:
    """Start a one-node Ray + Serve and block. Used by `python -m bench.stacks.<stack>.app`."""
    import ray
    from ray import serve

    ray.init(
        num_cpus=cfg.total_cpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        _metrics_export_port=int(os.environ.get("RAY_METRICS_PORT", "8080")),
        logging_level="warning",
    )
    serve.start(http_options={"host": "0.0.0.0", "port": cfg.port}, proxy_location="HeadOnly")
    serve.run(app_builder(cfg.as_dict()), blocking=True, route_prefix="/")


def concurrency_args(cfg: ServeConfig) -> dict:
    return {
        "num_replicas": cfg.replicas,
        "max_ongoing_requests": cfg.max_ongoing,
        "ray_actor_options": {"num_cpus": cfg.cpus_per_replica},
    }


def to_thread(fn, *a):
    return asyncio.to_thread(fn, *a)
