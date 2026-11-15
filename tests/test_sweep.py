"""Sweep + scenario logic against a fake backend whose latency is a function of offered rate."""
import argparse
import math

import orjson
import pytest

from bench.load import sweep as sw
from bench.load.backends import StackCfg, UpInfo, parse_k6_stdout
from bench.load.scenarios import grid, mk


class FakeBackend:
    """p99 grows with rate; capacity ~ 190 req/s at 100 ms, errors above 400."""

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
        p99 = 4.0 * math.exp(load / 60)          # 4 ms at 0, 20 ms @97, 50 ms @152, 100 ms @193
        err = 0 if load <= 400 else int(load)
        return {"ok": load * p.duration_s, "err": err, "dropped_iterations": 0, "rows_total": 0,
                "achieved_rps": load, "latency_ms": {"avg": p99 / 2, "p50": p99 / 2, "p90": p99 * 0.8,
                                                     "p99": p99, "max": p99 * 2}}


def _make(tmp_path, monkeypatch, profile="smoke", scenarios=None):
    monkeypatch.setattr(sw, "RESULTS", tmp_path)
    monkeypatch.setattr(sw, "ComposeBackend", lambda env_cfg, log: FakeBackend())
    monkeypatch.setattr("bench.versions.collect", lambda: {"fake": "1"})
    a = argparse.Namespace(profile=profile, env="laptop", scenarios=scenarios, dry_run=False,
                           no_resume=False, tag="local", registry="")
    return sw.Sweep(a)


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    return _make(tmp_path, monkeypatch)


def test_search_finds_capacity_per_slo(sweep):
    cfg = mk(sweep, "ray_ort", "T")
    caps = sweep.search(cfg, 200, start=25, factor=1.25)
    assert 150 <= caps[100] <= 193, caps
    assert caps[20] < caps[50] < caps[100]


def test_mst_rule_rejects_errors_and_drops():
    ok = {"k6": {"ok": 1000, "err": 0, "dropped_iterations": 0, "latency_ms": {"p99": 40.0}}}
    assert sw.Sweep.passes(ok, 50)
    assert not sw.Sweep.passes(ok, 20)
    assert not sw.Sweep.passes({"k6": {**ok["k6"], "err": 1}}, 50)
    assert not sw.Sweep.passes({"k6": {**ok["k6"], "dropped_iterations": 5}}, 50)       # > 0.1 %
    assert sw.Sweep.passes({"k6": {**ok["k6"], "dropped_iterations": 1}}, 50)           # <= 0.1 %


def test_grid_stops_after_consecutive_failures(sweep):
    sweep.p["quick"] = False
    sweep.m["grid"].update(rps=[50, 100, 250, 300, 350, 400], stop_after_fail=2)
    rows = grid(sweep, mk(sweep, "ray_ort", "T"), 200)
    assert [r["rate"] for r in rows] == [50, 100, 250, 300]       # 250 and 300 both miss 100 ms


def test_smoke_scenario_writes_rows_and_resumes(tmp_path, monkeypatch):
    s = _make(tmp_path, monkeypatch, scenarios="S1,A1")
    s.run()
    rows = [orjson.loads(line) for line in (tmp_path / "laptop.jsonl").read_text().splitlines()]
    kinds = {r["step_kind"] for r in rows}
    assert {"coldstart", "grid", "closed", "burst", "done"} <= kinds
    assert all(r["valid"] for r in rows if "k6" in r)
    assert (tmp_path / "runs" / f"{s.run_id}.json").exists()
    n = len(rows)
    s2 = _make(tmp_path, monkeypatch, scenarios="S1,A1")
    s2.run()
    assert len((tmp_path / "laptop.jsonl").read_text().splitlines()) == n      # everything resumed


def test_parse_k6_stdout():
    out = 'noise\nK6_SUMMARY_JSON:{"ok": 5}\n'
    assert parse_k6_stdout(out) == {"ok": 5}
    with pytest.raises(RuntimeError):
        parse_k6_stdout("nothing")


def test_stackcfg_family_helpers():
    cfg = StackCfg("triton", "deepfm", "small", False, 1, 1, 6, scenario="F1")
    assert cfg.grpc_port == 8001 and cfg.profile == "triton" and cfg.prom_urls == ["http://localhost:8002/metrics"]
