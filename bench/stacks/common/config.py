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
    # --- family / composition (bench.stacks.ray_family, fastapi family mode, triton ensembles)
    family_mode: str = ""         # "" | mux | deployments | fanout | pipeline
    variants: int = 16            # family size (deepfm_small_v00 .. v{N-1})
    mux_models_per_replica: int = 4
    fanout_k: int = 4
    lazy_load: bool = False       # FastAPI family: load sessions on first request
    # --- Ray autoscaling / backpressure (0 = off)
    autoscale_max: int = 0        # >0 enables autoscaling between autoscale_min and autoscale_max
    autoscale_min: int = 1
    autoscale_target: float = 2.0  # target_ongoing_requests
    autoscale_upscale_delay_s: float = 0.0
    autoscale_downscale_delay_s: float = 30.0
    max_queued: int = -1          # Ray max_queued_requests (-1 = unlimited)
    uvicorn_limit_concurrency: int = 0   # FastAPI/uvicorn --limit-concurrency (0 = off)
    grpc: bool = False            # also expose the Ray gRPC ingress on :9000

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
    def family(self) -> list[str]:
        from bench.models.schema import variant_names
        return variant_names(self.arch, self.size, self.variants)

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
            "num_cpus": "NUM_CPUS", "runtime": "RUNTIME", "family_mode": "FAMILY_MODE",
            "variants": "VARIANTS", "mux_models_per_replica": "MUX_MODELS_PER_REPLICA",
            "fanout_k": "FANOUT_K", "lazy_load": "LAZY_LOAD", "autoscale_max": "AUTOSCALE_MAX",
            "autoscale_min": "AUTOSCALE_MIN", "autoscale_target": "AUTOSCALE_TARGET",
            "autoscale_upscale_delay_s": "AUTOSCALE_UPSCALE_DELAY_S",
            "autoscale_downscale_delay_s": "AUTOSCALE_DOWNSCALE_DELAY_S", "max_queued": "MAX_QUEUED",
            "uvicorn_limit_concurrency": "UVICORN_LIMIT_CONCURRENCY", "grpc": "GRPC",
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
