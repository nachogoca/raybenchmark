import numpy as np
import pytest

from bench.models.schema import load_spec
from bench.protocol.payloads import PAYLOAD_DIR
from bench.stacks.common.config import ServeConfig
from bench.stacks.common.ort_session import make_session, predict
from bench.stacks.triton.repo import config_pbtxt, render_repo


def test_spec_sizes():
    s = load_spec("deepfm", "small")
    assert s.total_rows == 260_000 and s.embed_dim == 16
    assert load_spec("dcnv2", "large").total_rows > 9_000_000


def test_ort_session_runs(small_onnx_dir):
    sess = make_session(small_onnx_dir / "dcnv2_small.onnx", intra=1, inter=1)
    dense = np.ones((3, 13), np.float32)
    sparse = np.zeros((3, 26), np.int64)
    out = predict(sess, dense, sparse)
    assert out.shape == (3, 1) and np.all((out > 0) & (out < 1))


def test_render_repo(small_onnx_dir, tmp_path):
    mdir = render_repo(tmp_path, "deepfm", "small", instances=2, batching=True,
                       intra_threads=3, onnx_dir=small_onnx_dir)
    cfg = (mdir / "config.pbtxt").read_text()
    assert 'name: "deepfm_small"' in cfg and "count: 2" in cfg
    assert "dynamic_batching" in cfg and 'string_value: "3"' in cfg
    assert (mdir / "1" / "model.onnx").resolve().exists()
    off = config_pbtxt("x", instances=1, batching=False, intra_threads=1)
    assert "dynamic_batching" not in off and "KIND_CPU" in off
    assert "KIND_GPU" in config_pbtxt("x", instances=1, batching=False, intra_threads=1, gpu=True)


def test_config_from_args_types():
    c = ServeConfig.from_args({"arch": "dcnv2", "size": "small", "replicas": "4",
                               "batching": 1, "batch_wait_ms": "2.5", "num_cpus": 0.5})
    assert c.replicas == 4 and c.batching is True and c.batch_wait_ms == 2.5
    assert c.cpus_per_replica == 0.5 and c.model_name == "dcnv2_small"
    with pytest.raises(ValueError):
        ServeConfig.from_args({"bogus": 1})


def test_payload_files_exist_or_skip():
    if not (PAYLOAD_DIR / "rows1.json").exists():
        pytest.skip("run `just payloads`")
    import orjson
    bodies = orjson.loads((PAYLOAD_DIR / "rows200.json").read_bytes())
    assert len(bodies) == 64 and bodies[0]["inputs"][0]["shape"] == [200, 13]
