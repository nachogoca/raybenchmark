import argparse

import yaml

from bench import monitor
from bench.load import sweep as sw

LOG = """[22:32:54] run r1: profile=smoke env=laptop scenarios=['S1', 'I1'] quick=True
[22:32:54] [S1] ray_ort variant=rep0 replicas=1 batching=0
[22:33:21]     grid     rows=200 rate=50     ok=301    err=0   drop=0    p50=    3.4 p99=    6.5 cpu=3.4 rss=1010.9
[22:33:31]     grid     rows=200 rate=150    ok=901    err=0   drop=0    p50=    3.7 p99=   16.9 cpu=4.0 rss=1006.0  INVALID ['x']
[22:34:20] == S1 done in 2.6 min (est. so far 0.07 h)
[22:34:23] [I1] triton variant=triton_http replicas=1 batching=0
"""


def test_parse_log_progress():
    i = monitor.parse_log(LOG)
    assert i["scenarios"] == ["S1", "I1"] and i["done"] == ["S1"] and i["steps"] == 2 and i["invalid"] == 1
    assert i["config"] == {"test": "I1", "stack": "triton", "variant": "triton_http"} and not i["finished"]
    assert monitor.parse_log(LOG + "[23:00:00] == I1 done in 1.0 min (x)\n")["finished"]
    assert monitor.parse_log("[1] DISK LOW: 3 GB")["stopped"]


def _fake(monkeypatch, tmp_path, text, proc, free=100.0, age=10, cont=()):
    log = tmp_path / "r.log"
    log.write_text(text)
    import os
    import time
    os.utime(log, (time.time() - age, time.time() - age))
    monkeypatch.setattr(monitor, "latest_log", lambda: log)
    monkeypatch.setattr(monitor, "find_sweep", lambda: proc)
    monkeypatch.setattr(monitor, "containers", lambda: list(cont))
    monkeypatch.setattr(monitor.disk, "snapshot", lambda: {"free_gb": free, "used_pct": 50, "results_mb": 1, "docker": {}})


class _Proc:
    pid = 1

    def create_time(self):
        import time
        return time.time() - 600


def test_alerts_for_dead_stuck_disk_and_containers(monkeypatch, tmp_path):
    _fake(monkeypatch, tmp_path, LOG, _Proc())
    assert monitor.assess()[1] == ["1 invalid step(s) so far (see the log)"]
    _fake(monkeypatch, tmp_path, LOG, None)
    assert any("gone" in a for a in monitor.assess()[1])
    _fake(monkeypatch, tmp_path, LOG, _Proc(), age=60 * 30)
    assert any("silent" in a for a in monitor.assess()[1])
    _fake(monkeypatch, tmp_path, LOG, _Proc(), free=12)
    assert any("disk low" in a for a in monitor.assess()[1])
    _fake(monkeypatch, tmp_path, LOG, _Proc(), cont=[("raybench-ray_ort-1", "Restarting (1) 3 s ago")])
    assert any("Restarting" in a for a in monitor.assess()[1])
    _fake(monkeypatch, tmp_path, LOG + "[23:00:00] == I1 done in 1.0 min (x)\n", None)
    lines, alerts = monitor.assess()
    assert "FINISHED" in lines[0] and not [a for a in alerts if "gone" in a]


def test_run_stops_cleanly_when_disk_is_low(tmp_path, monkeypatch):
    import orjson
    import pytest
    monkeypatch.setattr(sw, "RESULTS", tmp_path)
    (tmp_path / "runs").mkdir()
    downs = []

    class B:
        def down(self, quiet=False):
            downs.append(1)
    monkeypatch.setattr(sw, "ComposeBackend", lambda env_cfg, log: B())
    s = sw.Sweep(argparse.Namespace(profile="smoke", env="laptop", scenarios="S1", dry_run=False, no_resume=False, tag="x", registry=""))
    monkeypatch.setattr(sw, "free_gb", lambda p: 5.0)
    with pytest.raises(SystemExit) as e:
        s.check_disk()
    assert e.value.code == 3 and downs
    row = orjson.loads((tmp_path / "laptop-smoke.jsonl").read_text().splitlines()[-1])
    assert row["step_kind"] == "stopped" and "DISK LOW" in row["reason"]
    monkeypatch.setattr(sw, "free_gb", lambda p: 500.0)
    s.check_disk()                                    # plenty of space: no exit


def test_every_container_log_is_capped():
    c = yaml.safe_load(open(monitor.ROOT / "deploy/docker/compose.yaml").read())
    for name, svc in c["services"].items():
        if "cpuset" in svc:
            assert svc["logging"]["options"]["max-size"] == "20m", name
    assert monitor.GUARDS["min_free_gb"] >= 10


def test_latest_log_ignores_the_monitor_own_log(monkeypatch, tmp_path):
    import os
    import time
    monkeypatch.setattr(monitor, "RUNS", tmp_path)
    (tmp_path / "run.log").write_text("x")
    (tmp_path / "monitor.log").write_text("y")
    os.utime(tmp_path / "run.log", (time.time() - 100, time.time() - 100))
    assert monitor.latest_log().name == "run.log"
