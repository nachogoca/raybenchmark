import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "artifacts" / "onnx"
# Ray writes a session folder per ray.init (tens to hundreds of MB each). Keep them off the
# RAM-backed /tmp (it filled up once). The path must stay short: AF_UNIX sockets cap at 107 bytes.
RAY_TMP = Path.home() / ".rbray"
os.environ.setdefault("RAY_TMPDIR", str(RAY_TMP))


@pytest.fixture(scope="session", autouse=True)
def _remove_ray_sessions():
    """Delete the Ray session folders this test run created, pass or fail."""
    RAY_TMP.mkdir(exist_ok=True)
    before = set((RAY_TMP / "ray").glob("session_*")) if (RAY_TMP / "ray").exists() else set()
    yield
    for d in (RAY_TMP / "ray").glob("session_*") if (RAY_TMP / "ray").exists() else []:
        if d not in before:
            shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def small_onnx_dir() -> Path:
    if not (ONNX_DIR / "deepfm_small.onnx").exists():
        pytest.skip("run `just models` first")
    return ONNX_DIR


@pytest.fixture
def serve_env(small_onnx_dir, monkeypatch):
    monkeypatch.setenv("MODEL_ARCH", "deepfm")
    monkeypatch.setenv("MODEL_SIZE", "small")
    monkeypatch.setenv("ONNX_DIR", str(small_onnx_dir))
    monkeypatch.setenv("TOTAL_CPUS", "2")
    return dict(os.environ)
