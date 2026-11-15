"""Vanilla FastAPI + ONNX Runtime. One session per uvicorn worker; sync endpoint runs in the
anyio threadpool (capped by FASTAPI_THREADS), which is how most people deploy it."""
from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request, Response

from bench.protocol.kserve_v2 import ProtocolError, decode_request, encode_response
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.runtime import make_runner

cfg = ServeConfig.from_env()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    anyio.to_thread.current_default_thread_limiter().total_tokens = cfg.fastapi_threads
    state["run"] = make_runner(cfg)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    if "run" not in state:
        raise HTTPException(503, "loading")
    return {"status": "ok", "model": cfg.model_name}


@app.get("/v2/models/{name}/ready")
def ready(name: str) -> Response:
    return Response(status_code=200 if name == cfg.model_name and "run" in state else 404)


@app.post("/v2/models/{name}/infer")
async def infer(name: str, request: Request) -> Response:
    if name != cfg.model_name:
        raise HTTPException(404, f"model {name} not served")
    body = await request.body()
    try:
        dense, sparse = decode_request(body)
    except ProtocolError as e:
        raise HTTPException(400, str(e)) from e
    prob = await anyio.to_thread.run_sync(state["run"], dense, sparse)
    return Response(encode_response(cfg.model_name, prob), media_type="application/json")
