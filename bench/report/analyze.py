"""Turn bench/results/*.jsonl into four CSVs under bench/results/summary/ and print the headline
table. Deterministic: same jsonl in, same CSV out.

  ladder.csv    every k6 step (search/confirm/overload/closed)
  capacity.csv  one row per (env, stack, model, rows, batching, replicas, slo): capacity + cost
  coldstart.csv one row per (env, stack, model, replicas): start_to_ready_s, rss_idle_mb, image_size_mb
  overload.csv  p99 / error rate at 1.5x and 3x capacity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
SUMMARY = RESULTS / "summary"
PRICES = ROOT / "config" / "prices.yaml"
GROUP = ["env", "stack", "arch", "size", "model", "rows", "batching", "replicas", "pod_cpu"]


def load_rows(files: list[Path]) -> list[dict]:
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def flatten(rows: list[dict]) -> pd.DataFrame:
    recs = []
    for r in rows:
        k, res, lat = r.get("k6") or {}, r.get("res") or {}, (r.get("k6") or {}).get("latency_ms") or {}
        recs.append({
            **{g: r.get(g) for g in GROUP}, "run_id": r.get("run_id"), "step_kind": r.get("step_kind"),
            "mode": r.get("mode"), "rate": r.get("rate"), "vus": r.get("vus"), "rep": r.get("rep"),
            "slo_target": r.get("slo_target"), "factor": r.get("factor"), "attempt": r.get("attempt"),
            "intra_threads": r.get("intra_threads"), "total_cpus": r.get("total_cpus"),
            "ok": k.get("ok"), "err": k.get("err"), "dropped": k.get("dropped_iterations"),
            "achieved_rps": k.get("achieved_rps"), "rows_total": k.get("rows_total"),
            "p50_ms": lat.get("p50"), "p90_ms": lat.get("p90"), "p99_ms": lat.get("p99"),
            "max_ms": lat.get("max"), "avg_ms": lat.get("avg"),
            "cpu_cores_mean": res.get("cpu_cores_mean"), "cpu_cores_peak": res.get("cpu_cores_peak"),
            "rss_mean_mb": res.get("rss_mean_mb"), "rss_peak_mb": res.get("rss_peak_mb"),
            "mem_peak_mb": res.get("mem_peak_mb"), "gpu_util_mean": res.get("gpu_util_mean"),
            "vram_peak_mb": res.get("vram_peak_mb"), "limit_cpu": res.get("limit_cpu"),
            "coldstart": r.get("coldstart"), "error": r.get("error"),
        })
    df = pd.DataFrame(recs)
    # A non-ORT runtime is a distinct "stack" in every grouping/chart: ray_ort[torch]. Old rows
    # have no runtime field and stay plain ORT stacks.
    rt = pd.Series([r.get("runtime") or "ort" for r in rows], index=df.index)
    df["runtime"] = rt
    df["stack"] = df["stack"].where(rt == "ort", df["stack"] + "[" + rt + "]")
    df["batching"] = df["batching"].astype("boolean")
    return df


def prices() -> dict:
    return yaml.safe_load(PRICES.read_text()) if PRICES.exists() else {}


def capacity_table(df: pd.DataFrame, slos: list[int]) -> pd.DataFrame:
    steps = df[df.step_kind.isin(["search", "confirm"])].copy()
    steps["clean"] = (steps.err.fillna(1) == 0) & (steps.dropped.fillna(1) == 0)
    out = []
    pr = prices()
    for key, g in steps.groupby(GROUP, dropna=False):
        env = key[0]
        usd_h = (pr.get(env) or {}).get("usd_per_hour")
        for slo in slos:
            best = None
            for kind in ("confirm", "search"):
                sub = g[g.step_kind == kind]
                # median across reps at the same rate
                agg = (sub.groupby("rate").agg(p50=("p50_ms", "median"), p99=("p99_ms", "median"),
                                               clean=("clean", "all"), cpu=("cpu_cores_mean", "mean"),
                                               rss=("rss_peak_mb", "max"), rps=("achieved_rps", "mean"),
                                               limit_cpu=("limit_cpu", "first"), n=("rate", "size"))
                       .reset_index())
                ok = agg[agg.clean & (agg.p99 <= slo)]
                if len(ok):
                    best = ok.loc[ok.rate.idxmax()]
                    best_kind = kind
                    break
            row = dict(zip(GROUP, key, strict=True))
            row.update(slo_ms=slo)
            if best is None:
                row.update(capacity_rps=0, confirmed=False)
            else:
                cap = float(best.rps if best_kind == "confirm" else best.rate)
                cpu_util = (best.cpu / best.limit_cpu * 100) if best.limit_cpu and best.cpu == best.cpu else None
                row.update(capacity_rps=round(cap, 1), confirmed=best_kind == "confirm", reps=int(best.n),
                           rows_per_s=round(cap * row["rows"], 0), p50_ms=round(best.p50, 2),
                           p99_ms=round(best.p99, 2), cpu_cores=round(best.cpu, 2) if best.cpu == best.cpu else None,
                           cpu_util_pct=round(cpu_util, 1) if cpu_util is not None else None,
                           rss_peak_mb=best.rss,
                           usd_per_1m_req=round(usd_h / (cap * 3600) * 1e6, 4) if usd_h and cap else None,
                           usd_per_1b_rows=round(usd_h / (cap * row["rows"] * 3600) * 1e9, 4) if usd_h and cap else None)
            out.append(row)
    return pd.DataFrame(out)


def coldstart_table(df: pd.DataFrame) -> pd.DataFrame:
    cs = df[df.step_kind == "coldstart"].copy()
    if cs.empty:
        return pd.DataFrame()
    cs["start_to_ready_s"] = cs.coldstart.map(lambda c: c.get("start_to_ready_s"))
    cs["rss_idle_mb"] = cs.coldstart.map(lambda c: c.get("rss_idle_mb"))
    cs["image_size_mb"] = cs.coldstart.map(lambda c: c.get("image_size_mb"))
    return (cs.groupby(["env", "stack", "arch", "size", "model", "replicas", "batching"], dropna=False)
              .agg(start_to_ready_s=("start_to_ready_s", "median"), rss_idle_mb=("rss_idle_mb", "median"),
                   image_size_mb=("image_size_mb", "first"), n=("start_to_ready_s", "size")).reset_index())


def overload_table(df: pd.DataFrame) -> pd.DataFrame:
    ov = df[df.step_kind == "overload"].copy()
    if ov.empty:
        return pd.DataFrame()
    ov["err_rate_pct"] = ov.err / (ov.ok + ov.err).replace(0, 1) * 100
    ov["dropped_pct"] = ov.dropped / (ov.rate * ov.ok.where(ov.ok > 0, 1) / ov.achieved_rps.replace(0, 1)).replace(0, 1) * 100
    cols = [*GROUP, "factor", "rate", "achieved_rps", "p50_ms", "p99_ms", "max_ms", "err_rate_pct", "dropped", "cpu_cores_mean"]
    return ov[cols].sort_values(GROUP + ["factor"]).reset_index(drop=True)


def headline(cap: pd.DataFrame, slo: int = 50) -> pd.DataFrame:
    """Best configuration per (env, stack, model, rows) at one SLO."""
    c = cap[(cap.slo_ms == slo) & (cap.capacity_rps > 0)]
    if c.empty:
        return c
    idx = c.groupby(["env", "stack", "model", "rows"]).capacity_rps.idxmax()
    cols = ["env", "stack", "model", "rows", "batching", "replicas", "capacity_rps", "p99_ms",
            "cpu_util_pct", "rss_peak_mb", "usd_per_1m_req", "confirmed"]
    return c.loc[idx, cols].sort_values(["env", "model", "rows", "capacity_rps"], ascending=[True, True, True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", help="jsonl files (default: bench/results/*.jsonl)")
    ap.add_argument("--slo", type=int, default=50, help="SLO for the printed headline table")
    ap.add_argument("--out", help="summary dir (default: bench/results/summary)")
    a = ap.parse_args()
    global SUMMARY
    if a.out:
        SUMMARY = Path(a.out)
    files = [Path(f) for f in a.results] if a.results else sorted(RESULTS.glob("*.jsonl"))
    if not files:
        print("no results yet")
        return
    slos = yaml.safe_load((ROOT / "config" / "matrix.yaml").read_text())["slos_ms"]
    df = flatten(load_rows(files))
    SUMMARY.mkdir(exist_ok=True)
    ladder = df[df.step_kind.isin(["search", "confirm", "overload", "closed"])].drop(columns=["coldstart", "error"])
    ladder.to_csv(SUMMARY / "ladder.csv", index=False)
    cap = capacity_table(df, slos)
    cap.to_csv(SUMMARY / "capacity.csv", index=False)
    coldstart_table(df).to_csv(SUMMARY / "coldstart.csv", index=False)
    overload_table(df).to_csv(SUMMARY / "overload.csv", index=False)
    errors = df[df.step_kind == "error"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(f"rows: {len(df)}  steps: {len(ladder)}  configs with errors: {len(errors)}")
    print(f"\nheadline @ p99<={a.slo}ms (best config per stack/model/rows):")
    print(headline(cap, a.slo).to_string(index=False))
    if len(errors):
        print("\nerrors:")
        print(errors[["env", "stack", "model", "batching", "replicas", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
