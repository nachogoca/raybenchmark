import gzip

import numpy as np

from bench.collect import k6parse as kp


def test_aggregate_percentiles_and_errors():
    rng = np.random.default_rng(0)
    lat = rng.lognormal(1.0, 0.5, 5000)
    ts = np.linspace(1000.0, 1010.0, 5000)
    status = np.full(5000, 200)
    status[:50] = 503
    out = kp.aggregate(ts, lat, status)
    assert out["n"] == 5000 and out["n_err"] == 50 and out["error_codes"] == {"503": 50}
    ok = lat[50:]
    assert abs(out["pcts"]["99"] - np.percentile(ok, 99)) < 1e-9
    assert len(out["series"]) == 11 and sum(s["rps"] for s in out["series"]) == 5000
    counts = np.array(out["hist"]["counts"])
    assert counts.sum() == len(ok)
    est = kp.hist_percentile(out["hist"]["counts"], 99)
    assert abs(est / out["pcts"]["99"] - 1) < 0.05            # bucket width ~3.4 %


def test_parse_csv_gz_keeps_only_measure_metric(tmp_path):
    p = tmp_path / "k6.csv.gz"
    lines = ["metric_name,timestamp,metric_value,status"]
    lines += [f"measure_latency,{100 + i / 10},{5 + i % 7},200" for i in range(100)]
    lines += ["http_req_duration,100,999,200", "measure_latency,105,50,503"]
    p.write_bytes(gzip.compress("\n".join(lines).encode()))
    out = kp.parse_csv(p)
    assert out["n"] == 101 and out["n_err"] == 1 and out["error_codes"] == {"503": 1}
    assert max(out["pcts"].values()) < 999


def test_empty():
    out = kp.aggregate(np.array([]), np.array([]), np.array([], dtype=int))
    assert out["n"] == 0 and out["series"] == [] and out["hist"] is None
