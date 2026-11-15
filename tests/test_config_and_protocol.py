"""ServeConfig scenario knobs, Ray deployment options, gRPC codec, model variants, Triton repo."""
import numpy as np
import pytest

from bench.load.backends import K6Params, StackCfg
from bench.models.schema import load_spec, variant_names
from bench.protocol.kserve_grpc import decode_infer_request, encode_infer_response
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ray_runner import concurrency_args
from bench.stacks.triton import pybackend, repo


def test_config_reads_scenario_env(monkeypatch):
    for k, v in {"FAMILY_MODE": "fanout", "VARIANTS": "8", "FANOUT_K": "4", "LAZY_LOAD": "1", "AUTOSCALE_MAX": "4",
                 "MAX_QUEUED": "50", "UVICORN_LIMIT_CONCURRENCY": "30", "GRPC": "1", "TOTAL_CPUS": "6"}.items():
        monkeypatch.setenv(k, v)
    c = ServeConfig.from_env()
    assert (c.family_mode, c.variants, c.fanout_k, c.lazy_load, c.autoscale_max, c.max_queued, c.grpc) == \
        ("fanout", 8, 4, True, 4, 50, True)
    assert c.family == variant_names("deepfm", "small", 8) and c.family[3] == "deepfm_small_v03"


def test_concurrency_args_static_autoscale_backpressure():
    static = concurrency_args(ServeConfig(replicas=2, total_cpus=6))
    assert static["num_replicas"] == 2 and static["ray_actor_options"]["num_cpus"] == 3 and "max_queued_requests" not in static
    auto = concurrency_args(ServeConfig(autoscale_max=4, autoscale_min=1, autoscale_target=8, total_cpus=6, max_queued=50))
    ac = auto["autoscaling_config"]
    assert (ac["min_replicas"], ac["max_replicas"], ac["target_ongoing_requests"], ac["upscale_delay_s"]) == (1, 4, 8.0, 0.0)
    assert auto["ray_actor_options"]["num_cpus"] == 1.5 and "num_replicas" not in auto and auto["max_queued_requests"] == 50


def test_variant_specs_are_distinct_models():
    base, v0, v1 = load_spec("deepfm", "small"), load_spec("deepfm", "small", 0), load_spec("deepfm", "small", 1)
    assert (base.name, v0.name, v1.name) == ("deepfm_small", "deepfm_small_v00", "deepfm_small_v01")
    from bench.models.build import build
    w0 = next(build(v0).parameters()).detach().numpy()
    w1 = next(build(v1).parameters()).detach().numpy()
    assert not np.allclose(w0, w1)                                   # different seeds
    assert np.allclose(w0, next(build(v0).parameters()).detach().numpy())   # deterministic


def _req(dense, sparse, raw):
    from tritonclient.grpc import service_pb2 as pb
    req = pb.ModelInferRequest(model_name="m")
    for name, dt, arr in (("dense", "FP32", dense), ("sparse", "INT64", sparse)):
        t = req.inputs.add()
        t.name, t.datatype = name, dt
        t.shape.extend(arr.shape)
        if raw:
            req.raw_input_contents.append(arr.tobytes())
        elif dt == "FP32":
            t.contents.fp32_contents.extend(arr.ravel().tolist())
        else:
            t.contents.int64_contents.extend(arr.ravel().tolist())
    return req


@pytest.mark.parametrize("raw", [False, True])
def test_grpc_codec_roundtrip(raw):
    dense = np.random.default_rng(0).random((5, 13), dtype=np.float32)
    sparse = np.random.default_rng(1).integers(0, 100, (5, 26), dtype=np.int64)
    d, s = decode_infer_request(_req(dense, sparse, raw))
    np.testing.assert_array_equal(d, dense)
    np.testing.assert_array_equal(s, sparse)
    resp = encode_infer_response("m", np.array([[0.25], [0.5]], dtype=np.float32))
    assert resp.model_name == "m" and list(resp.outputs[0].shape) == [2, 1]
    assert np.frombuffer(resp.raw_output_contents[0], dtype=np.float32).tolist() == [0.25, 0.5]


def test_k6params_env_family_and_step():
    p = K6Params(model="deepfm_small", rows=200, mode="rate", rate=50, models="a,b", zipf_s=1.1,
                 shape="step", steps="30:40,300:90")
    e = p.env()
    assert e["MODELS"] == "a,b" and e["ZIPF_S"] == "1.1" and e["SHAPE"] == "step" and e["STEPS"] == "30:40,300:90"
    assert "MODELS" not in K6Params(model="m", rows=1, mode="rate").env()


def test_stackcfg_ports_and_urls():
    assert StackCfg("ray_ort", "deepfm", "small", False, 1, 1, 6).grpc_port == 9000
    assert StackCfg("triton", "deepfm", "small", False, 1, 1, 6).host_port == 8100
    assert StackCfg("ray_triton_grpc", "deepfm", "small", False, 1, 1, 6).prom_urls == [
        "http://localhost:8080/metrics", "http://localhost:8002/metrics"]
    assert StackCfg("ray_family", "deepfm", "small", False, 1, 1, 6, services=("x",)).profile == "x"


def test_triton_repo_family_ensembles_and_queue(tmp_path, monkeypatch):
    onnx = tmp_path / "onnx"
    onnx.mkdir()
    for n in variant_names("deepfm", "small", 4) + ["deepfm_small"]:
        (onnx / f"{n}.onnx").write_bytes(b"x")
    out = tmp_path / "repo"
    monkeypatch.setattr("sys.argv", ["repo", "--out", str(out), "--arch", "deepfm", "--size", "small", "--batching", "1",
                                     "--onnx-dir", str(onnx), "--family", "4", "--ensembles", "1", "--queue-size", "50"])
    repo.main()
    assert (out / "deepfm_small_v03" / "1" / "model.onnx").is_symlink()
    assert "max_queue_size: 50" in (out / "deepfm_small_v02" / "config.pbtxt").read_text()
    fan = (out / "fan4_deepfm_small_v02" / "config.pbtxt").read_text()
    assert 'platform: "ensemble"' in fan and "deepfm_small_v00" in fan           # wraps past the end of the family
    assert (out / "pipe_deepfm_small_v01" / "config.pbtxt").exists()
    for k in (4, 8):
        assert (out / f"average{k}" / "1" / "model.py").exists()
    assert (out / "featurize" / "config.pbtxt").read_text().startswith('name: "featurize"')
    assert pybackend.py_config("x", [("a", "TYPE_FP32", 1)], [("b", "TYPE_FP32", 1)]).count("data_type") == 2
