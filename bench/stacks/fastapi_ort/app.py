"""Vanilla FastAPI + ONNX Runtime. One session per uvicorn worker; sync endpoint runs in the
anyio threadpool (capped by FASTAPI_THREADS), which is how most people deploy it.

Family modes (FAMILY_MODE): "" single model | mux (N sessions in one process, eager or LAZY_LOAD)
| fanout (average K sessions with asyncio.gather) | pipeline (featurize -> model -> postprocess).
GET /metrics exposes handler time for the latency breakdown (no prometheus_client needed).
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request, Response

from bench.protocol.kserve_v2 import ProtocolError, decode_request, encode_response
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.pipeline import average, fanout_models, featurize, postprocess
from bench.stacks.common.runtime import make_runner

cfg = ServeConfig.from_env()
state: dict = {}
stats = {"sum": 0.0, "count": 0}


def _load(name: str):
    if not cfg.family_mode:
        return make_runner(cfg)
    from bench.stacks.common.ort_session import make_session, predict
    sess = make_session(f"{cfg.onnx_dir}/{name}.onnx", cfg.intra_threads, cfg.inter_threads, cfg.provider)
    return lambda d, s: predict(sess, d, s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    anyio.to_thread.current_default_thread_limiter().total_tokens = cfg.fastapi_threads
    state["names"] = cfg.family if cfg.family_mode else [cfg.model_name]
    state["runners"] = {}
    state["lock"] = asyncio.Lock()
    if not (cfg.family_mode and cfg.lazy_load):
        for n in state["names"]:
            state["runners"][n] = await anyio.to_thread.run_sync(_load, n)
    state["ready"] = True
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


async def runner(name: str):
    r = state["runners"].get(name)
    if r is None:
        if name not in state["names"]:
            raise HTTPException(404, f"model {name} not served")
        async with state["lock"]:
            r = state["runners"].get(name)
            if r is None:
                r = state["runners"][name] = await anyio.to_thread.run_sync(_load, name)
    return r


@app.get("/healthz")
def healthz() -> dict:
    if not state.get("ready"):
        raise HTTPException(503, "loading")
    return {"status": "ok", "model": cfg.model_name, "loaded": len(state["runners"])}


@app.get("/v2/models/{name}/ready")
def ready(name: str) -> Response:
    return Response(status_code=200 if name in state.get("names", []) and state.get("ready") else 404)


@app.get("/metrics")
def metrics() -> Response:
    body = (f"raybench_handler_seconds_sum {stats['sum']}\nraybench_handler_seconds_count {stats['count']}\n"
            f"raybench_loaded_models {len(state.get('runners', {}))}\n")
    return Response(body, media_type="text/plain")


async def _score(name: str, dense, sparse):
    return await anyio.to_thread.run_sync(await runner(name), dense, sparse)


@app.post("/v2/models/{name}/infer")
async def infer(name: str, request: Request) -> Response:
    t0 = time.perf_counter()
    if name not in state["names"]:
        raise HTTPException(404, f"model {name} not served")
    body = await request.body()
    try:
        dense, sparse = decode_request(body)
    except ProtocolError as e:
        raise HTTPException(400, str(e)) from e
    mode = cfg.family_mode
    if mode == "fanout":
        members = fanout_models(state["names"], name, cfg.fanout_k)
        prob = average(list(await asyncio.gather(*(_score(m, dense, sparse) for m in members))))
    elif mode == "pipeline":
        d, s = featurize(dense, sparse)
        prob = postprocess(await _score(name, d, s))
    else:
        prob = await _score(name, dense, sparse)
    out = Response(encode_response(name, prob), media_type="application/json")
    stats["sum"] += time.perf_counter() - t0
    stats["count"] += 1
    return out
