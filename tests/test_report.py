import orjson

from bench.report import analyze as an
from bench.report import report


def _step(stack, rate, p99, scen="S1", kind="grid", rep=0, variant="rep0", **kw):
    dist = {"pcts": {"50": p99 / 2, "75": p99 * .7, "90": p99 * .8, "95": p99 * .9, "99": p99, "99.9": p99 * 1.5,
                     "99.99": p99 * 2, "100": p99 * 3}, "mean_ms": p99 / 2,
            "hist": {"lo_ms": .05, "hi_ms": 20000, "buckets": 240, "counts": [0] * 100 + [10] + [0] * 139},
            "series": [{"t": t, "rps": rate, "ok": rate, "err": 0, "p50": p99 / 2, "p99": p99} for t in range(10)],
            "error_codes": {}, "n": 100, "n_err": 0, "t0": 0.0}
    return {"run_id": "r", "scenario": scen, "variant": variant, "stack": stack, "step_kind": kind, "rate": rate,
            "rep": rep, "rows": 200, "replicas": 1, "batching": False, "total_cpus": 6, "valid": True,
            "duration_s": 30,
            "k6": {"ok": rate * 30, "err": 0, "dropped_iterations": 0, "achieved_rps": rate,
                   "latency_ms": {"avg": p99 / 2, "p50": p99 / 2, "p90": p99 * .8, "p99": p99, "max": p99 * 3}, "dist": dist},
            "res": {"cpu_cores_mean": rate / 100, "rss_peak_mb": 500, "cpu_by_component": {"uvicorn": rate / 100}, "cpu_core_seconds": 30},
            "prom": {}, **kw}


def _rows():
    rows = []
    for stack, scale in (("fastapi_ort", 1.0), ("ray_ort", 1.5)):
        for rep in range(3):
            for rate in (50, 100, 200, 300):
                rows.append(_step(stack, rate, 4 * np_exp(rate * scale / 100), rep=rep, variant=f"rep{rep}"))
        rows.append({"step_kind": "coldstart", "scenario": "S1", "variant": "rep0", "stack": stack, "replicas": 1,
                     "coldstart": {"start_to_ready_s": 3.0, "rss_idle_mb": 80.0, "image_size_mb": 700.0}})
    return rows


def np_exp(x):
    import math
    return math.exp(x)


def test_capacity_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "RESULTS", tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "laptop.jsonl").write_bytes(b"\n".join(orjson.dumps(r) for r in _rows()))
    df = an.flatten(an.load_rows("laptop"))
    cap = an.s1_capacity(df, [20, 50, 100]).set_index(["stk", "slo"])
    # fastapi p99 = 4*exp(rate/100): 50ms -> rate<=252 -> grid 200; ray_ort slower -> 100
    assert cap.loc[("fastapi_ort", 50), "capacity"] == 200
    assert cap.loc[("ray_ort", 50), "capacity"] == 100
    assert cap.loc[("fastapi_ort", 50), "lo"] <= 200 <= cap.loc[("fastapi_ort", 50), "hi"]
    out = report.build("laptop", tmp_path / "r.html", tmp_path / "REPORT.md")
    text = out.read_text()
    assert "Executive summary" in text and "FastAPI + ORT" in text and "plotly" in text.lower()
    assert "No data" in text                       # scenarios that did not run are labelled, not hidden
    assert "| FastAPI + ORT | 200 |" in (tmp_path / "REPORT.md").read_text()


def test_recovery_seconds():
    series = [{"t": t, "p99": 500 if t < 5 else 20, "err": 0} for t in range(20)]
    assert an.recovery_s(series, 0, 50, hold=3) == 5.0
    assert an.recovery_s(series, 0, 10, hold=3) is None
