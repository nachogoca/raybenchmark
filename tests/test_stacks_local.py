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


# ---------------------------------------------------------------- model family (multiplex / fan-out / pipeline)
def _family_ref(onnx_dir, names, dense, sparse, mode, primary):
    from bench.stacks.common import pipeline as pl

    def run(n, d, s):
        return predict(make_session(onnx_dir / f"{n}.onnx"), d, s)
    if mode == "fanout":
        return pl.average([run(m, dense, sparse) for m in pl.fanout_models(names, primary, 3)])
    if mode == "pipeline":
        d, s = pl.featurize(dense, sparse)
        return pl.postprocess(run(primary, d, s))
    return run(primary, dense, sparse)


@pytest.fixture
def family_env(serve_env, small_onnx_dir, monkeypatch):
    if not (small_onnx_dir / "deepfm_small_v03.onnx").exists():
        pytest.skip("run `just models` first (family variants)")
    monkeypatch.setenv("VARIANTS", "4")
    monkeypatch.setenv("FANOUT_K", "3")
    return small_onnx_dir


@pytest.mark.parametrize("mode,lazy", [("mux", "0"), ("mux", "1"), ("fanout", "0"), ("pipeline", "0")])
def test_fastapi_family_modes_match_reference(family_env, monkeypatch, mode, lazy):
    from fastapi.testclient import TestClient

    import bench.stacks.fastapi_ort.app as mod
    monkeypatch.setenv("FAMILY_MODE", mode)
    monkeypatch.setenv("LAZY_LOAD", lazy)
    importlib.reload(mod)
    names = [f"deepfm_small_v0{i}" for i in range(4)]
    dense, sparse, body = _payload(6)
    with TestClient(mod.app) as c:
        assert c.get("/healthz").json()["loaded"] == (0 if lazy == "1" else 4)
        for primary in ("deepfm_small_v00", "deepfm_small_v03"):         # v03 wraps around in fan-out
            r = c.post(f"/v2/models/{primary}/infer", content=body)
            assert r.status_code == 200, r.text
            np.testing.assert_allclose(decode_response(r.content),
                                       _family_ref(family_env, names, dense, sparse, mode, primary), atol=1e-5)
        assert c.post("/v2/models/deepfm_small_v09/infer", content=body).status_code == 404
        if lazy == "1":
            assert c.get("/healthz").json()["loaded"] >= 1
        assert "raybench_handler_seconds_count 2" in c.get("/metrics").text


@pytest.mark.slow
def test_ray_family_modes_match_reference(family_env, monkeypatch):
    import httpx
    import ray
    from ray import serve

    from bench.stacks.common.config import ServeConfig
    from bench.stacks.ray_family.app import app_builder

    names = [f"deepfm_small_v0{i}" for i in range(4)]
    dense, sparse, body = _payload(6)
    monkeypatch.setenv("MUX_MODELS_PER_REPLICA", "2")        # smaller than the family: forces eviction
    ray.init(num_cpus=2, include_dashboard=False, logging_level="error", ignore_reinit_error=True)
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 18001})
        for mode in ("mux", "deployments", "fanout", "pipeline"):
            monkeypatch.setenv("FAMILY_MODE", mode)
            serve.run(app_builder(ServeConfig.from_env().as_dict()), route_prefix="/")
            with httpx.Client(base_url="http://127.0.0.1:18001", timeout=60) as c:
                assert c.get("/healthz").json()["mode"] == mode
                for primary in names:                          # 4 models through a 2-model LRU
                    r = c.post(f"/v2/models/{primary}/infer", content=body,
                               headers={"serve_multiplexed_model_id": primary})
                    assert r.status_code == 200, (mode, r.text)
                    np.testing.assert_allclose(decode_response(r.content),
                                               _family_ref(family_env, names, dense, sparse, mode, primary), atol=1e-5)
                assert c.post("/v2/models/deepfm_small_v09/infer", content=body).status_code == 404
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.mark.slow
def test_ray_grpc_ingress_matches_ort(serve_env, small_onnx_dir, monkeypatch):
    import ray
    import tritonclient.grpc as tgrpc
    from ray import serve

    from bench.stacks.common.config import ServeConfig
    from bench.stacks.common.ray_runner import enable_grpc_pickle_alias
    from bench.stacks.ray_ort.app import app_builder

    enable_grpc_pickle_alias()          # before ray.init, so every Ray process can pickle the request
    dense, sparse, _ = _payload(4)
    ref = predict(make_session(small_onnx_dir / "deepfm_small.onnx"), dense, sparse)
    ray.init(num_cpus=2, include_dashboard=False, logging_level="error", ignore_reinit_error=True)
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 18002}, grpc_options={
            "port": 19000, "grpc_servicer_functions": [
                "tritonclient.grpc.service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server"]})
        serve.run(app_builder(ServeConfig.from_env().as_dict()), route_prefix="/")
        client = tgrpc.InferenceServerClient("127.0.0.1:19000")
        ins = [tgrpc.InferInput("dense", list(dense.shape), "FP32"), tgrpc.InferInput("sparse", list(sparse.shape), "INT64")]
        ins[0].set_data_from_numpy(dense)
        ins[1].set_data_from_numpy(sparse)
        res = client.infer("deepfm_small", ins, headers={"application": "default"})
        np.testing.assert_allclose(res.as_numpy("prob"), ref, atol=1e-6)
    finally:
        serve.shutdown()
        ray.shutdown()
