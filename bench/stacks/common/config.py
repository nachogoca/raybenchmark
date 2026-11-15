"""Serving configuration shared by every stack: env vars locally, Serve `args` on KubeRay."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.models.schema import MODELS, SIZES


def _cpus_available() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


@dataclass
class ServeConfig:
    arch: str = "deepfm"
    size: str = "small"
    onnx_dir: str = "/models/onnx"
    replicas: int = 1
    batching: bool = False
    batch_max: int = 64
    batch_wait_ms: float = 5.0
    intra_threads: int = 1
    inter_threads: int = 1
    provider: str = "CPUExecutionProvider"
    runtime: str = "ort"          # inference runtime: "ort" (ONNX Runtime) | "torch" (PyTorch eager)
    port: int = 8000
    total_cpus: int = field(default_factory=_cpus_available)
    triton_url: str = "triton:8001"
    model_repo: str = "/models/repo"
    fastapi_threads: int = 8      # anyio threadpool cap per uvicorn worker
    max_ongoing: int = 8          # Ray Serve max_ongoing_requests per replica
    num_cpus: float = 0.0         # explicit Ray num_cpus per replica (0 = total_cpus / replicas)

    def __post_init__(self) -> None:
        if self.arch not in MODELS or self.size not in SIZES:
            raise ValueError(f"unknown model {self.arch}/{self.size}")
        if self.runtime not in ("ort", "torch"):
            raise ValueError(f"unknown runtime {self.runtime}")

    @property
    def model_name(self) -> str:
        return f"{self.arch}_{self.size}"

    @property
    def onnx_file(self) -> Path:
        return Path(self.onnx_dir) / f"{self.model_name}.onnx"

    @property
    def cpus_per_replica(self) -> float:
        if self.num_cpus > 0:
            return self.num_cpus
        return max(self.total_cpus / self.replicas, 0.1)

    @property
    def infer_path(self) -> str:
        return f"/v2/models/{self.model_name}/infer"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_env(cls) -> ServeConfig:
        e = os.environ
        kw = {}
        mapping = {
            "arch": "MODEL_ARCH", "size": "MODEL_SIZE", "onnx_dir": "ONNX_DIR",
            "replicas": "REPLICAS", "batching": "BATCHING", "batch_max": "BATCH_MAX",
            "batch_wait_ms": "BATCH_WAIT_MS", "intra_threads": "ORT_INTRA_THREADS",
            "inter_threads": "ORT_INTER_THREADS", "provider": "ORT_PROVIDER", "port": "PORT",
            "total_cpus": "TOTAL_CPUS", "triton_url": "TRITON_URL", "model_repo": "MODEL_REPO",
            "fastapi_threads": "FASTAPI_THREADS", "max_ongoing": "MAX_ONGOING",
            "num_cpus": "NUM_CPUS", "runtime": "RUNTIME",
        }
        for k, env in mapping.items():
            if env in e:
                kw[k] = e[env]
        return cls.from_args(kw)

    @classmethod
    def from_args(cls, args: dict) -> ServeConfig:
        types = {f.name: f.type for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kw = {}
        for k, v in args.items():
            if k not in types:
                raise ValueError(f"unknown config key {k}")
            t = types[k]
            if t == "bool" or t is bool:
                kw[k] = str(v).lower() in ("1", "true", "yes", "on")
            elif t == "int" or t is int:
                kw[k] = int(v)
            elif t == "float" or t is float:
                kw[k] = float(v)
            else:
                kw[k] = str(v)
        return cls(**kw)
