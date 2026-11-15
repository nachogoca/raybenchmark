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
        self.ups, self.downs, self.k6_calls, self.kills, self.params = 0, 0, [], [], []

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

    def restart_count(self):
        return 0

    def kill_matching(self, pattern, n=1, sig=9):
        self.kills.append(("proc", pattern))
        return [1]

    def kill_container(self, service, restart_after_s=1.0):
        self.kills.append(("container", service))

    def run_k6(self, p, raw_path=None):
        load = p.rate if p.mode == "rate" else p.vus * 50
        self.k6_calls.append((p.mode, load, p.duration_s))
        self.params.append(p)
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
    s = sw.Sweep(a)
    s.m["guards"]["min_free_gb"] = 0          # tmp dirs live on a small RAM disk; the guard has its own test
    return s


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
    rows = [orjson.loads(line) for line in (tmp_path / "laptop-smoke.jsonl").read_text().splitlines()]
    kinds = {r["step_kind"] for r in rows}
    assert {"coldstart", "grid", "closed", "burst", "done"} <= kinds
    assert all(r["valid"] for r in rows if "k6" in r)
    assert (tmp_path / "runs" / f"{s.run_id}.json").exists()
    n = len(rows)
    s2 = _make(tmp_path, monkeypatch, scenarios="S1,A1")
    s2.run()
    assert len((tmp_path / "laptop-smoke.jsonl").read_text().splitlines()) == n      # everything resumed


def test_parse_k6_stdout():
    out = 'noise\nK6_SUMMARY_JSON:{"ok": 5}\n'
    assert parse_k6_stdout(out) == {"ok": 5}
    with pytest.raises(RuntimeError):
        parse_k6_stdout("nothing")


def test_stackcfg_family_helpers():
    cfg = StackCfg("triton", "deepfm", "small", False, 1, 1, 6, scenario="F1")
    assert cfg.grpc_port == 8001 and cfg.profile == "triton" and cfg.prom_urls == ["http://localhost:8002/metrics"]


class _ImmediateTimer:
    """threading.Timer stand-in: fires the hook as soon as the step starts."""

    def __init__(self, delay, fn):
        self.fn, self.daemon, self.cancelled = fn, False, False

    def start(self):
        self.fn()

    def cancel(self):
        self.cancelled = True


def test_every_scenario_runs_on_fake_backend(tmp_path, monkeypatch):
    from bench.load import scenarios as sc

    monkeypatch.setattr(sw.threading, "Timer", _ImmediateTimer)
    monkeypatch.setattr(sc, "_first_cold", lambda url, body, model, headers=None: (120.0, 3.0))
    monkeypatch.setattr(sc.httpx, "post", lambda *a, **k: type("R", (), {"raise_for_status": lambda self: None})())
    s = _make(tmp_path, monkeypatch, scenarios=",".join(sc.SCENARIOS))
    s.run()
    rows = [orjson.loads(line) for line in (tmp_path / "laptop-smoke.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["step_kind"] == "error"], [r.get("error") for r in rows if r["step_kind"] == "error"]
    seen = {(r["scenario"], r["step_kind"]) for r in rows}
    for expect in [("S1", "grid"), ("S1", "closed"), ("S1_scale", "search"), ("S1_batch", "search"), ("I1", "grid"),
                   ("F1", "family_fixed"), ("F2", "cold_variant"), ("F3", "fanout"), ("F5", "grid"),
                   ("A1", "burst"), ("R1", "overload"), ("R2", "failure")]:
        assert expect in seen, expect
    # family traffic carries the model list and Zipf skew; gRPC variants flag the grpc script
    fam = [p for p in s.backend.params if p.models]
    assert fam and all(p.zipf_s == 1.1 for p in fam)
    assert any(p.grpc for p in s.backend.params)
    assert any(p.shape == "step" and p.steps.count(",") == 2 for p in s.backend.params)       # A1 burst
    assert s.backend.kills, "R2 must kill something"
    assert all(r["valid"] for r in rows if "k6" in r)


def test_r2_hook_kills_the_configured_target(tmp_path, monkeypatch):
    monkeypatch.setattr(sw.threading, "Timer", _ImmediateTimer)
    s = _make(tmp_path, monkeypatch, scenarios="R2")
    s.p["quick"] = False
    s.m["scenarios"]["R2"].update(reps=1, measure_s=20)
    s.run()
    assert ("proc", "ray::ServeReplica") in s.backend.kills
    assert ("container", "triton") in s.backend.kills


def test_invalid_when_loadgen_saturated(tmp_path, monkeypatch):
    s = _make(tmp_path, monkeypatch)
    orig = s.backend.run_k6
    s.backend.run_k6 = lambda p, raw=None: {**orig(p, raw), "k6_cpu_frac": 0.95}
    row = s.step(mk(s, "ray_ort", "T"), 200, kind="grid", rate=50, long=True)
    assert row["valid"] is False and "loadgen_cpu" in row["invalid_reasons"][0]


def test_dry_run_estimates_without_side_effects(tmp_path, monkeypatch):
    a = argparse.Namespace(profile="full", env="laptop", scenarios=None, dry_run=True, no_resume=False, tag="local", registry="")
    monkeypatch.setattr(sw, "RESULTS", tmp_path)
    s = sw.Sweep(a)
    s.run()
    assert 6 < s.est_s / 3600 < 12, s.est_s / 3600          # overnight budget
    assert not (tmp_path / "laptop-smoke.jsonl").exists()


def test_matrix_keys_are_strings():
    """YAML 1.1 turns bare on/off/yes/no keys into booleans, which silently renames variants."""
    import yaml
    def walk(d, path=""):
        for k, v in d.items():
            assert isinstance(k, str), f"non-string key {k!r} under {path}"
            if isinstance(v, dict):
                walk(v, f"{path}/{k}")
    walk(yaml.safe_load(sw.MATRIX.read_text()))


def test_profiles_never_share_results_or_resume_keys(tmp_path, monkeypatch):
    """A full run must not skip configs a smoke run finished, nor write into the smoke file."""
    smoke = _make(tmp_path, monkeypatch, scenarios="S1")
    smoke.run()
    assert (tmp_path / "laptop-smoke.jsonl").exists() and not (tmp_path / "laptop.jsonl").exists()
    a = argparse.Namespace(profile="full", env="laptop", scenarios="S1", dry_run=True, no_resume=False, tag="x", registry="")
    full = sw.Sweep(a)
    assert full.out.name == "laptop.jsonl" and not full.done_keys
