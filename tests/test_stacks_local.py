"""Correctness gate for the two stacks that run without Docker: FastAPI and Ray Serve + ORT
must return identical probabilities for the same payload, with batching on and off."""
import importlib

import numpy as np
import orjson
import pytest

from bench.protocol.kserve_v2 import decode_response, encode_request
from bench.stacks.common.ort_session import make_session, predict


def _payload(rows: int):
    rng = np.random.default_rng(7)
    dense = rng.lognormal(size=(rows, 13)).astype(np.float32)
    sparse = rng.integers(0, 10_000, (rows, 26), dtype=np.int64)
    return dense, sparse, encode_request(dense, sparse)


def test_fastapi_matches_ort(serve_env, small_onnx_dir):
    from fastapi.testclient import TestClient

    import bench.stacks.fastapi_ort.app as mod
    importlib.reload(mod)
    dense, sparse, body = _payload(5)
    ref = predict(make_session(small_onnx_dir / "deepfm_small.onnx"), dense, sparse)
    with TestClient(mod.app) as c:
        assert c.get("/healthz").status_code == 200
        r = c.post("/v2/models/deepfm_small/infer", content=body)
        assert r.status_code == 200
        np.testing.assert_allclose(decode_response(r.content), ref, atol=1e-6)
        assert c.post("/v2/models/nope/infer", content=body).status_code == 404
        assert c.post("/v2/models/deepfm_small/infer", content=b"{}").status_code == 400


@pytest.mark.slow
@pytest.mark.parametrize("batching", ["0", "1"])
def test_ray_ort_matches_ort(serve_env, small_onnx_dir, monkeypatch, batching):
    import httpx
    import ray
    from ray import serve

    from bench.stacks.common.config import ServeConfig
    from bench.stacks.ray_ort.app import app_builder

    monkeypatch.setenv("BATCHING", batching)
    monkeypatch.setenv("REPLICAS", "2")
    cfg = ServeConfig.from_env()
    ray.init(num_cpus=2, include_dashboard=False, logging_level="error", ignore_reinit_error=True)
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 18000})
        serve.run(app_builder(cfg.as_dict()), route_prefix="/")
        dense, sparse, body = _payload(9)
        ref = predict(make_session(small_onnx_dir / "deepfm_small.onnx"), dense, sparse)
        with httpx.Client(base_url="http://127.0.0.1:18000", timeout=30) as c:
            assert c.get("/healthz").status_code == 200
            # several concurrent-ish requests so the batcher has something to merge
            outs = [c.post("/v2/models/deepfm_small/infer", content=body) for _ in range(4)]
            for r in outs:
                assert r.status_code == 200, r.text
                np.testing.assert_allclose(decode_response(r.content), ref, atol=1e-6)
            assert orjson.loads(outs[0].content)["model_name"] == "deepfm_small"
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.mark.parametrize("arch", ["deepfm", "dcnv2"])
def test_torch_runtime_matches_ort(small_onnx_dir, monkeypatch, arch):
    """The PyTorch runtime serves the same seeded weights as the ONNX file: outputs must agree."""
    from bench.stacks.common.config import ServeConfig
    from bench.stacks.common.runtime import make_runner

    monkeypatch.setenv("ONNX_DIR", str(small_onnx_dir))
    monkeypatch.setenv("MODEL_ARCH", arch)
    monkeypatch.setenv("MODEL_SIZE", "small")
    outs = {}
    dense, sparse, _ = _payload(11)
    for runtime in ("ort", "torch"):
        monkeypatch.setenv("RUNTIME", runtime)
        outs[runtime] = make_runner(ServeConfig.from_env())(dense, sparse)
    assert outs["torch"].shape == outs["ort"].shape == (11, 1)
    np.testing.assert_allclose(outs["torch"], outs["ort"], atol=1e-5)


def test_unknown_runtime_rejected():
    from bench.stacks.common.config import ServeConfig

    with pytest.raises(ValueError, match="runtime"):
        ServeConfig.from_args({"runtime": "tensorflow"})


@pytest.mark.slow
@pytest.mark.parametrize("batching", ["0", "1"])
def test_ray_fastapi_matches_ort(serve_env, small_onnx_dir, monkeypatch, batching):
    """Ray Serve + FastAPI ingress returns the same probabilities and the same error codes."""
    import httpx
    import ray
    from ray import serve

    from bench.stacks.common.config import ServeConfig
    from bench.stacks.ray_fastapi.app import app_builder

    monkeypatch.setenv("BATCHING", batching)
    monkeypatch.setenv("REPLICAS", "1")
    cfg = ServeConfig.from_env()
    ray.init(num_cpus=2, include_dashboard=False, logging_level="error", ignore_reinit_error=True)
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 18001})
        serve.run(app_builder(cfg.as_dict()), route_prefix="/")
        dense, sparse, body = _payload(9)
        ref = predict(make_session(small_onnx_dir / "deepfm_small.onnx"), dense, sparse)
        with httpx.Client(base_url="http://127.0.0.1:18001", timeout=30) as c:
            assert c.get("/healthz").status_code == 200
            r = c.post("/v2/models/deepfm_small/infer", content=body)
            assert r.status_code == 200, r.text
            np.testing.assert_allclose(decode_response(r.content), ref, atol=1e-6)
            assert c.post("/v2/models/nope/infer", content=body).status_code == 404
            assert c.post("/v2/models/deepfm_small/infer", content=b"{}").status_code == 400
    finally:
        serve.shutdown()
        ray.shutdown()
