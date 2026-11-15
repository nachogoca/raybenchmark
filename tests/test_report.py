import itertools

import pandas as pd

from bench.report import build, charts


def _write(tmp_path):
    cap, lad, ov = [], [], []
    for stack, rows, rep, bat, slo in itertools.product(
        ["fastapi_ort", "ray_ort"], [1, 200], [1, 4], [False, True], [20, 50, 100]
    ):
        c = 100 * rep * slo / 20 / (1 + rows / 50) * (1.5 if bat else 1)
        cap.append(
            dict(env="laptop-cpu", stack=stack, arch="deepfm", size="small", model="deepfm",
                 rows=rows, batching=bat, replicas=rep, pod_cpu=1, slo_ms=slo, capacity_rps=c,
                 confirmed=slo != 20, reps=3, rows_per_s=c * rows, p50_ms=3, p99_ms=slo * 0.9,
                 cpu_cores=rep * 0.8, cpu_util_pct=80, rss_peak_mb=500,
                 usd_per_1m_req=0.05 if stack == "ray_ort" else None, usd_per_1b_rows=None)
        )
        for rate in (10, 50, 200):
            lad.append(
                dict(env="laptop-cpu", stack=stack, arch="deepfm", size="small", model="deepfm",
                     rows=rows, batching=bat, replicas=rep, pod_cpu=1, run_id="r1",
                     step_kind="search", rate=rate, p99_ms=rate / 4 + 5)
            )
    for stack in ("fastapi_ort", "ray_ort"):
        for f in (1.5, 3.0):
            ov.append(
                dict(env="laptop-cpu", stack=stack, arch="deepfm", size="small", model="deepfm",
                     rows=1, batching=True, replicas=4, pod_cpu=1, factor=f, rate=100,
                     achieved_rps=90, p50_ms=5, p99_ms=40 * f, max_ms=90, err_rate_pct=f,
                     dropped=0, cpu_cores_mean=3)
            )
    cold = [
        dict(env="laptop-cpu", stack=s, arch="deepfm", size="small", model="deepfm", replicas=4,
             batching=True, start_to_ready_s=t, rss_idle_mb=200, image_size_mb=900, n=1)
        for s, t in (("fastapi_ort", 3), ("ray_ort", 9))
    ]
    for name, rows in (("capacity", cap), ("ladder", lad), ("overload", ov), ("coldstart", cold)):
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False)


def test_report(tmp_path):
    _write(tmp_path)
    figs = tmp_path / "figures"
    out = charts.make_all(tmp_path, figs)
    names = {p.name for p in out}
    assert names == {
        "latency_vs_rate_laptop-cpu.svg",
        "throughput_vs_replicas_laptop-cpu.svg",
        "cost_per_slo_laptop-cpu.svg",
    }
    assert all(p.exists() for p in out)
    md, app = tmp_path / "REPORT.md", tmp_path / "APPENDIX.md"
    build.build(tmp_path, figs, md, app)
    text = md.read_text()
    assert "| Stack | Best config (batching, replicas) | Capacity req/s |" in text
    for n in names:
        assert f"figures/{n})" in text
    assert "Recommendation" in text
    assert len(text.splitlines()) < 80
    assert "## Cold start" in app.read_text()


def test_empty_inputs(tmp_path):
    assert charts.make_all(tmp_path, tmp_path / "f") == []
