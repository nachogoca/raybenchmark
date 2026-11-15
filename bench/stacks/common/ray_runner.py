"""Shared Ray Serve plumbing: health/infer routing inside a replica and the local runner."""
from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

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


def enable_grpc_pickle_alias() -> None:
    """Make `grpc_service_pb2` importable in every Ray process (needed to pickle gRPC requests)."""
    shim = str(Path(__file__).resolve().parents[2] / "protocol" / "pb_shim")
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (shim, os.environ.get("PYTHONPATH", "")) if p)
    if shim not in sys.path:
        sys.path.insert(0, shim)


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
    kw = {}
    if cfg.grpc:
        enable_grpc_pickle_alias()
        kw["grpc_options"] = {
            "port": 9000,
            "grpc_servicer_functions": [
                "tritonclient.grpc.service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server"],
        }
    serve.start(http_options={"host": "0.0.0.0", "port": cfg.port}, proxy_location="HeadOnly", **kw)
    serve.run(app_builder(cfg.as_dict()), blocking=True, route_prefix="/")


def concurrency_args(cfg: ServeConfig, **over) -> dict:
    """Deployment options shared by every Ray stack: replicas or autoscaling, backpressure."""
    opts: dict = {
        "max_ongoing_requests": cfg.max_ongoing,
        "ray_actor_options": {"num_cpus": cfg.cpus_per_replica},
    }
    if cfg.autoscale_max > 0:
        opts["autoscaling_config"] = {
            "min_replicas": cfg.autoscale_min, "max_replicas": cfg.autoscale_max,
            "initial_replicas": cfg.autoscale_min,
            "target_ongoing_requests": cfg.autoscale_target,
            "upscale_delay_s": cfg.autoscale_upscale_delay_s,
            "downscale_delay_s": cfg.autoscale_downscale_delay_s,
            "metrics_interval_s": 1.0, "look_back_period_s": 5.0,
        }
        # replicas share the fixed core budget: size each for the max fleet
        opts["ray_actor_options"] = {"num_cpus": max(cfg.total_cpus / cfg.autoscale_max, 0.1)}
    else:
        opts["num_replicas"] = cfg.replicas
    if cfg.max_queued >= 0:
        opts["max_queued_requests"] = cfg.max_queued
    opts.update(over)
    return opts


def to_thread(fn, *a):
    return asyncio.to_thread(fn, *a)
