"""Turn bench/results/<env>.jsonl into tidy tables and the derived numbers the report shows.

Capacity (MST, max sustainable throughput at an SLO) = highest offered rate on the S1 grid where
the median-over-reps p99 <= SLO, with zero errors and <= 0.1 % dropped arrivals. Its 95 % CI comes
from bootstrapping the repetitions at every rate. Steps flagged invalid (load generator CPU-bound,
CPU throttled, restarts) are excluded from capacity and listed in the appendix.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results"
CONFIG = ROOT / "config"


def load_rows(env: str = "laptop") -> list[dict]:
    path = RESULTS / f"{env}.jsonl"
    return [orjson.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def matrix() -> dict:
    return yaml.safe_load((CONFIG / "matrix.yaml").read_text())


def prices() -> dict:
    p = yaml.safe_load((CONFIG / "prices.yaml").read_text())
    return {"name": p["default"], **p["reference"][p["default"]]}


def latest_run_meta() -> dict:
    files = sorted((RESULTS / "runs").glob("*.json"))
    return json.loads(files[-1].read_text()) if files else {}


def flatten(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        if "k6" not in r:
            continue
        k, res = r["k6"], r.get("res") or {}
        lat, dist = k["latency_ms"], k.get("dist") or {}
        pc = dist.get("pcts") or {}
        out.append({
            "scenario": r.get("scenario", ""), "variant": r.get("variant", ""), "stk": r["stack"],
            "kind": r["step_kind"], "rate": r.get("rate") or 0, "vus": r.get("vus") or 0,
            "rep": r.get("rep", 0), "rows": r.get("rows"), "replicas": r.get("replicas", 1),
            "batching": r.get("batching", False), "valid": r.get("valid", True),
            "invalid": "; ".join(r.get("invalid_reasons") or []),
            "p50": lat.get("p50"), "p90": lat.get("p90"), "p99": lat.get("p99"), "pmax": lat.get("max"),
            "avg": lat.get("avg"), "p999": pc.get("99.9"), "err": k["err"], "drop": k["dropped_iterations"],
            "ok": k["ok"], "rps": k.get("achieved_rps") or 0.0, "duration_s": r.get("duration_s"),
            "cpu": res.get("cpu_cores_mean"), "cpu_peak": res.get("cpu_cores_peak"),
            "core_s": res.get("cpu_core_seconds"), "rss": res.get("rss_peak_mb"),
            "mem": res.get("mem_peak_mb"), "comp": res.get("cpu_by_component") or {},
            "k6_frac": k.get("k6_cpu_frac"), "dist": dist, "prom": r.get("prom") or {},
            "host": r.get("host") or {}, "protocol": r.get("protocol", "http"),
            "n_models": r.get("n_models", 1), "k": r.get("k"), "factor": r.get("factor"),
            "policy": r.get("policy"), "base_capacity": r.get("base_capacity"),
            "hook_at": r.get("hook_at_unix"), "t_measure": r.get("t_measure_unix"),
            "steps": r.get("steps"), "error_codes": dist.get("error_codes") or {},
            "slo_target": r.get("slo_target"), "total_cpus": r.get("total_cpus"),
        })
    return pd.DataFrame(out)


def coldstarts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "scenario": r.get("scenario", ""), "variant": r.get("variant", ""), "stk": r["stack"],
        "replicas": r.get("replicas"), "ready_s": r["coldstart"]["start_to_ready_s"],
        "idle_rss_mb": r["coldstart"].get("rss_idle_mb"), "image_mb": r["coldstart"].get("image_size_mb"),
    } for r in rows if r.get("step_kind") == "coldstart"])


def cold_variants(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: r.get(k if k != "stk" else "stack") for k in ("variant", "stk", "trial", "model", "kind", "first_ms",
                                                "steady_ms", "load_ms")}
                         for r in rows if r.get("step_kind") == "cold_variant"])


# ------------------------------------------------------------------ capacity
def _points(df: pd.DataFrame) -> dict[int, dict]:
    pts: dict[int, dict] = {}
    for rate, g in df.groupby("rate"):
        pts[int(rate)] = {"p99": [float(x) for x in g["p99"] if x is not None and not math.isnan(x)],
                          "clean": bool(((g["err"] == 0) & (g["drop"] <= 0.001 * (g["ok"] + g["drop"] + g["err"]).clip(lower=1))).all())}
    return pts


def mst(points: dict[int, dict], slo: float, rng: np.random.Generator | None = None) -> int:
    best = 0
    for rate, p in sorted(points.items()):
        vals = p["p99"]
        if not vals or not p["clean"]:
            continue
        if rng is not None:
            vals = list(rng.choice(vals, size=len(vals), replace=True))
        if float(np.median(vals)) <= slo:
            best = rate
    return best


def capacity_ci(points: dict[int, dict], slo: float, n: int = 400) -> tuple[int, int, int]:
    rng = np.random.default_rng(7)
    est = mst(points, slo)
    boots = sorted(mst(points, slo, rng) for _ in range(n))
    return est, boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def s1_points(df: pd.DataFrame, stack: str, rows: int = 200) -> dict[int, dict]:
    sel = df[(df.scenario == "S1") & (df.stk == stack) & (df.rows == rows) & df.kind.isin(["grid", "refine"]) & df.valid]
    return _points(sel)


def s1_capacity(df: pd.DataFrame, slos: list[int]) -> pd.DataFrame:
    out = []
    for stack in sorted(df[df.scenario == "S1"].stk.unique()):
        pts = s1_points(df, stack)
        for slo in slos:
            est, lo, hi = capacity_ci(pts, slo)
            out.append({"stk": stack, "slo": slo, "capacity": est, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def search_capacity(df: pd.DataFrame, scenario: str, slo: float) -> pd.DataFrame:
    """Capacity from short search steps (secondary scenarios): per variant, highest passing rate."""
    sel = df[(df.scenario == scenario) & (df.kind == "search")]
    out = []
    for (variant, stack), g in sel.groupby(["variant", "stk"]):
        ok = g[(g.err == 0) & (g.drop <= 0.001 * (g.ok + g.drop + g.err).clip(lower=1)) & (g.p99 <= slo)]
        out.append({"variant": variant, "stk": stack, "capacity": int(ok.rate.max()) if len(ok) else 0,
                    "replicas": int(g.replicas.iloc[0])})
    return pd.DataFrame(out)


def knee(points: dict[int, dict]) -> int | None:
    """First rate where median p99 exceeds 2x the lowest-rate median p99."""
    items = [(r, float(np.median(p["p99"]))) for r, p in sorted(points.items()) if p["p99"]]
    if len(items) < 2:
        return None
    base = items[0][1]
    return next((r for r, v in items[1:] if v > 2 * base), None)


# ------------------------------------------------------------------ cost and resources
def cost_per_million(rps: float, vcpus: float, mem_gb: float, pr: dict) -> float | None:
    if not rps:
        return None
    return (vcpus * pr["usd_per_vcpu_hour"] + mem_gb * pr["usd_per_gb_hour"]) / (rps * 3600) * 1e6


def step_at(df: pd.DataFrame, stack: str, rate: int, scenario: str = "S1", rows: int = 200) -> pd.DataFrame:
    return df[(df.scenario == scenario) & (df.stk == stack) & (df.rows == rows)
              & (df.rate == rate) & df.kind.isin(["grid", "refine"]) & df.valid]


def scorecard(df: pd.DataFrame, cold: pd.DataFrame, slo: int = 50, ref_rate: int = 100) -> pd.DataFrame:
    pr, total = prices(), None
    cap = s1_capacity(df, [slo]).set_index("stk")
    out = []
    for stack, c in cap.iterrows():
        at = step_at(df, stack, int(c.capacity)) if c.capacity else df.iloc[0:0]
        ref = step_at(df, stack, ref_rate)
        total = int(df[(df.stk == stack) & (df.scenario == "S1")].total_cpus.iloc[0])
        rss = float(at.rss.median()) if len(at) and at.rss.notna().any() else None
        cons = float(at.cpu.median()) if len(at) and at.cpu.notna().any() else None
        cs = cold[(cold.scenario == "S1") & (cold.stk == stack)]
        out.append({
            "stk": stack, "capacity": int(c.capacity), "ci": f"{int(c.lo)}-{int(c.hi)}",
            "p99_at_ref": float(ref.p99.median()) if len(ref) else None,
            "cores_at_cap": cons, "core_ms_per_req": (cons / c.capacity * 1000) if cons and c.capacity else None,
            "rps_per_vcpu": c.capacity / total if c.capacity else None, "rss_mb": rss,
            "cold_s": float(cs.ready_s.median()) if len(cs) else None,
            "usd_per_m_alloc": cost_per_million(c.capacity, total, (rss or 0) / 1024, pr),
            "usd_per_m_used": cost_per_million(c.capacity, cons or 0, (rss or 0) / 1024, pr) if cons else None,
        })
    return pd.DataFrame(out).sort_values("capacity", ascending=False)


# ------------------------------------------------------------------ time-line scenarios
def recovery_s(series: list[dict], start: int, slo: float, hold: int = 5) -> float | None:
    """Seconds after `start` until p99 stays <= slo for `hold` consecutive seconds."""
    ok = 0
    for s in series:
        if s["t"] < start:
            continue
        ok = ok + 1 if (s["p99"] is not None and s["p99"] <= slo and s["err"] == 0) else 0
        if ok >= hold:
            return float(s["t"] - hold + 1 - start)
    return None
