"""Entrypoint: uvicorn with REPLICAS workers (each worker = one process with its own session)."""
from __future__ import annotations

import uvicorn

from bench.stacks.common.config import ServeConfig


def main() -> None:
    cfg = ServeConfig.from_env()
    uvicorn.run(
        "bench.stacks.fastapi_ort.app:app",
        host="0.0.0.0",
        port=cfg.port,
        workers=cfg.replicas,
        log_level="warning",
        access_log=False,
        http="httptools",
        loop="uvloop",
    )


if __name__ == "__main__":
    main()
