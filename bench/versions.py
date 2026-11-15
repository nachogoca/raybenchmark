"""Collect tool, library, image and host versions. Never raises; every value is a string."""



from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBS = ["ray", "onnxruntime", "tritonclient", "fastapi", "uvicorn", "numpy", "orjson"]
OPTIONAL_LIBS = ["torch", "onnx"]
IMAGES = [
    "raybench-py:local",
    "raybench-triton-ray:local",
    "nvcr.io/nvidia/tritonserver:26.10-py3",
]


def _run(cmd: list[str], timeout: float = 5) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True, cwd=ROOT
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _lib(name: str, missing: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return missing


def _kubectl() -> str:
    out = _run(["kubectl", "version", "--client", "-o", "json"])
    if not out:
        return "missing"
    try:
        return str(json.loads(out)["clientVersion"]["gitVersion"])
    except Exception:
        return "missing"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"       # macOS


def _lock_hash() -> str:
    try:
        return hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    except Exception:
        return "missing"


def _first_line(s: str | None) -> str:
    return s.splitlines()[0] if s else "missing"


def collect() -> dict[str, str]:
    d: dict[str, str] = {"python": platform.python_version()}
    for name in LIBS:
        d[name] = _lib(name, "missing")
    for name in OPTIONAL_LIBS:
        d[name] = _lib(name, "n/a")

    d["docker"] = _first_line(_run(["docker", "--version"]))
    d["docker compose"] = _first_line(_run(["docker", "compose", "version"]))
    d["k6"] = _first_line(_run(["k6", "version"]))
    d["kind"] = _first_line(_run(["kind", "version"]))
    d["kubectl"] = _kubectl()
    d["helm"] = _first_line(_run(["helm", "version", "--short"]))
    d["just"] = _first_line(_run(["just", "--version"]))

    for image in IMAGES:
        out = _run(
            ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image]
        )
        d[f"image {image}"] = out or "not built"

    d["git sha"] = _run(["git", "rev-parse", "--short", "HEAD"]) or "no-commit"
    d["uv.lock sha256"] = _lock_hash()
    d["kernel"] = _run(["uname", "-r"]) or "unknown"
    d["cpu"] = _cpu_model()
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    d["gpu"] = gpu.replace("\n", "; ") if gpu else "none"
    return {k: str(v) for k, v in d.items()}


def main() -> None:
    info = collect()
    width = max(len(k) for k in info)
    for k, v in info.items():
        print(f"{k:<{width}}  {v}")


if __name__ == "__main__":
    main()
