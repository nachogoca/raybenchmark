"""Turn a k6 per-request CSV (`--out csv=...`, optionally .gz) into step aggregates.

Only points of the custom `measure_latency` metric are used: the k6 script emits them for the
measure scenario only (warm-up excluded), tagged with the HTTP status.

Output (all JSON-serialisable, small enough for a jsonl row):
  n, n_err, error_codes   request counts and a status -> count map for non-200 responses
  pcts                    percentile spectrum of successful requests (ms)
  hist                    log-spaced histogram {lo_ms, hi_ms, buckets, counts}
  series                  per-second [{t, rps, ok, err, p50, p99}] relative to the first sample
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np

PCTS = (50, 75, 90, 95, 99, 99.9, 99.99, 100)
HIST_LO_MS, HIST_HI_MS, HIST_BUCKETS = 0.05, 20000.0, 240   # ~3.4 % bucket width


def hist_edges() -> np.ndarray:
    return np.geomspace(HIST_LO_MS, HIST_HI_MS, HIST_BUCKETS + 1)


def aggregate(ts: np.ndarray, lat_ms: np.ndarray, status: np.ndarray) -> dict:
    """ts: seconds (float); lat_ms: latency; status: int HTTP status (0 = connection error)."""
    n = int(len(lat_ms))
    ok_mask = status == 200
    ok = lat_ms[ok_mask]
    codes: dict[str, int] = {}
    for code in np.unique(status[~ok_mask]):
        codes[str(int(code))] = int((status == code).sum())
    out: dict = {"n": n, "n_err": int((~ok_mask).sum()), "error_codes": codes,
                 "t0": float(ts.min()) if n else None}
    if len(ok):
        out["pcts"] = {str(p): float(np.percentile(ok, p)) for p in PCTS}
        out["mean_ms"] = float(ok.mean())
        out["stdev_ms"] = float(ok.std())
        counts, _ = np.histogram(np.clip(ok, HIST_LO_MS, HIST_HI_MS), bins=hist_edges())
        out["hist"] = {"lo_ms": HIST_LO_MS, "hi_ms": HIST_HI_MS, "buckets": HIST_BUCKETS,
                       "counts": counts.astype(int).tolist()}
    else:
        out["pcts"], out["hist"] = {}, None
    series = []
    if n:
        t0 = float(ts.min())
        sec = np.floor(ts - t0).astype(int)
        for s in range(int(sec.max()) + 1):
            m = sec == s
            if not m.any():
                series.append({"t": s, "rps": 0, "ok": 0, "err": 0, "p50": None, "p99": None})
                continue
            okm = m & ok_mask
            lat = lat_ms[okm]
            series.append({
                "t": s, "rps": int(m.sum()), "ok": int(okm.sum()), "err": int((m & ~ok_mask).sum()),
                "p50": float(np.percentile(lat, 50)) if len(lat) else None,
                "p99": float(np.percentile(lat, 99)) if len(lat) else None,
            })
    out["series"] = series
    return out


def _open(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return path.open()


def parse_csv(path: str | Path) -> dict:
    """Read k6's CSV output. Columns used: metric_name, timestamp, metric_value, status."""
    import pandas as pd

    path = Path(path)
    with _open(path) as f:
        df = pd.read_csv(f, usecols=["metric_name", "timestamp", "metric_value", "status"],
                         dtype={"status": "string"})
    df = df[df["metric_name"] == "measure_latency"]
    status = pd.to_numeric(df["status"], errors="coerce").fillna(0).astype(int).to_numpy()
    return aggregate(df["timestamp"].to_numpy(dtype=float),
                     df["metric_value"].to_numpy(dtype=float), status)


def hist_percentile(counts: list[int], q: float) -> float:
    """Percentile estimate from the stored histogram (upper bucket edge); used to merge reps."""
    c = np.asarray(counts, dtype=float)
    if c.sum() == 0:
        return float("nan")
    idx = int(np.searchsorted(np.cumsum(c), q / 100.0 * c.sum()))
    return float(hist_edges()[min(idx + 1, HIST_BUCKETS)])
