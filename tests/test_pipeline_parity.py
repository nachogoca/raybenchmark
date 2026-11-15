"""The Triton python-backend math must equal the numpy stages the Ray and FastAPI stacks run."""
import numpy as np

from bench.stacks.common import pipeline as pl
from bench.stacks.triton import pybackend as pb


def _ns():
    ns: dict = {}
    exec(pb.PURE_MATH, ns)
    return ns


def test_featurize_postprocess_average_parity():
    rng = np.random.default_rng(1)
    dense = rng.lognormal(0, 1, (7, 13)).astype(np.float32)
    sparse = rng.integers(0, 10_000, (7, 26), dtype=np.int64)
    ns = _ns()
    d1, s1 = pl.featurize(dense, sparse)
    d2, s2 = ns["featurize"](dense, sparse)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(s1, s2)
    p = rng.uniform(0.01, 0.99, (7, 1)).astype(np.float32)
    np.testing.assert_allclose(pl.postprocess(p), ns["postprocess"](p), rtol=1e-6)
    np.testing.assert_allclose(pl.average([p, p * 0.5]), ns["average"]([p, p * 0.5]), rtol=1e-6)
    assert s1.min() >= 0 and s1.max() < 10_000


def test_fanout_members_wrap():
    fam = [f"m{i}" for i in range(4)]
    assert pl.fanout_models(fam, "m3", 3) == ["m3", "m0", "m1"]


def test_ensemble_configs_are_wired():
    cfg = pb.fanout_ensemble("fan4_x", ["a", "b", "c", "d"])
    assert cfg.count("model_name:") == 5 and 'model_name: "average4"' in cfg
    pipe = pb.pipeline_ensemble("pipe_x", "a")
    assert pipe.index("featurize") < pipe.index('"a"') < pipe.index("postprocess")
