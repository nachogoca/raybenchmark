"""Sweep logic against a fake backend whose latency is a function of offered rate."""
import argparse
import math

import pytest

from bench.load import sweep as sw
from bench.load.backends import StackCfg, UpInfo, parse_k6_stdout


class FakeBackend:
    """p99 grows with rate; capacity ~ 200 req/s at 100 ms, errors above 400."""
    def __init__(self):
        self.ups, self.downs, self.k6_calls = 0, 0, []

    def up(self, cfg):
        self.ups += 1
        return UpInfo(1.5, 100.0, 500.0, [])

    def down(self, quiet=False):
        self.downs += 1

    def reader(self):
        return lambda: {"cpu_usec": 0, "cpu_cores": 1.0, "mem_bytes": 1e6, "rss_bytes": 1e6,
                        "gpu_util": None, "vram_mb": None}

    def limits(self):
        return {"limit_cpu": 6, "limit_mem": "12g"}

    def run_k6(self, p, raw_path=None):
        load = p.rate if p.mode == "rate" else p.vus * 50
        self.k6_calls.append((p.mode, load, p.duration_s))
        p99 = 4.0 * math.exp(load / 60)          # 4ms at 0, 20ms@97, 50ms@152, 100ms@193
        err = 0 if load <= 400 else int(load)
        return {"ok": load * p.duration_s, "err": err, "dropped_iterations": 0, "rows_total": 0,
                "achieved_rps": load, "latency_ms": {"avg": p99 / 2, "p50": p99 / 2, "p90": p99 * 0.8,
                                                     "p99": p99, "max": p99 * 2}}


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "RESULTS", tmp_path)
    a = argparse.Namespace(profile="core", env="laptop-cpu", stacks="ray_ort,fastapi_ort", models="deepfm",
                           rows="1", dry_run=False, no_resume=False, tag="local", registry="")
    monkeypatch.setattr(sw, "ComposeBackend", lambda env_cfg, log: FakeBackend())
    s = sw.Sweep(a)
    monkeypatch.setattr("bench.versions.collect", lambda: {"fake": "1"})
    return s


def test_search_and_confirm_find_capacity(sweep):
    cfg = StackCfg("ray_ort", "deepfm", "small", True, 1, 6, 6)
    caps = sweep.search(cfg, 1)
    assert caps[100] >= 150 and caps[100] <= 193, caps
    assert caps[20] < caps[50] < caps[100]
    confirmed = sweep.confirm(cfg, 1, caps)
    assert all(confirmed[s] > 0 for s in confirmed)
    assert all(confirmed[s] <= caps[s] for s in confirmed)
    long_steps = [c for c in sweep.backend.k6_calls if c[2] == sweep.p["measure_s"]]
    assert len(long_steps) == 3 * sweep.p["reps"]    # one confirm per SLO, reps each


def test_full_run_writes_rows_and_resumes(sweep):
    sweep.run()
    rows = [ln for ln in sweep.out.read_text().splitlines() if ln]
    import orjson
    recs = [orjson.loads(ln) for ln in rows]
    kinds = {r["step_kind"] for r in recs}
    assert {"coldstart", "search", "confirm", "overload", "closed", "done"} <= kinds
    # ray_ort x batching(2) x replicas(2) + fastapi x replicas(2) = 6 configs
    assert sum(r["step_kind"] == "done" for r in recs) == 6
    assert sweep.backend.ups == 6 and sweep.backend.downs == 6
    # resume: nothing left to do
    a = argparse.Namespace(profile="core", env="laptop-cpu", stacks="ray_ort,fastapi_ort", models="deepfm",
                           rows="1", dry_run=False, no_resume=False, tag="local", registry="")
    s2 = sw.Sweep(a)
    assert len([c for c in s2.configs() if s2.config_key(c) not in s2.done_keys]) == 0


def test_fastapi_never_batches(sweep):
    cfgs = [c for c in sweep.configs() if c.stack == "fastapi_ort"]
    assert cfgs and all(not c.batching for c in cfgs)
    assert all(c.intra_threads == 6 // c.replicas for c in cfgs)


def test_parse_k6_stdout():
    out = "noise\nK6_SUMMARY_JSON:{\"ok\": 1}\n"
    assert parse_k6_stdout(out) == {"ok": 1}
    with pytest.raises(RuntimeError):
        parse_k6_stdout("nothing")
