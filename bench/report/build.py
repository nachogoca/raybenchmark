"""Write REPORT.md (one page) and bench/report/APPENDIX.md from the summary CSVs."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import pandas as pd

from bench.report.charts import (
    DEFAULT_OUT,
    DEFAULT_SUMMARY,
    HERE,
    best_configs,
    main_model,
    read_csv,
    stacks_in,
)

ROOT = HERE.parents[1]


def md_table(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (tabulate is not installed)."""

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _int(v) -> str:
    return "n/a" if pd.isna(v) else f"{round(v):d}"


def _cfg(r) -> str:
    return f"batch {'on' if r['batching'] else 'off'}, {int(r['replicas'])}x"


def _cold(cold: pd.DataFrame, r) -> float:
    if cold.empty:
        return float("nan")
    m = cold[(cold["env"] == r["env"]) & (cold["stack"] == r["stack"])]
    exact = m[(m["replicas"] == r["replicas"]) & (m["batching"] == r["batching"])]
    m = exact if not exact.empty else m
    return m["start_to_ready_s"].mean() if not m.empty else float("nan")


def headline_rows(best: pd.DataFrame) -> list[int]:
    vals = sorted(best["rows"].unique())
    pick = [r for r in (1, 200) if r in vals]
    return pick or ([vals[0], vals[-1]] if len(vals) > 1 else vals)


def summary_table(best: pd.DataFrame, cold: pd.DataFrame, rows: int) -> tuple[str, bool]:
    out, star = [], False
    b = best[best["rows"] == rows]
    for stack in stacks_in(b):
        r = b[b["stack"] == stack].iloc[0]
        mark = "" if r["confirmed"] else "*"
        star |= bool(mark)
        usd = "n/a" if pd.isna(r["usd_per_1m_req"]) else f"{r['usd_per_1m_req']:.4f}"
        out.append(
            {
                "Stack": stack,
                "Best config (batching, replicas)": _cfg(r),
                "Capacity req/s": _int(r["capacity_rps"]) + mark,
                "p99 ms at capacity": _int(r["p99_ms"]) + mark,
                "CPU util %": _int(r["cpu_util_pct"]),
                "Peak RSS MB": _int(r["rss_peak_mb"]),
                "Cold start s": _int(_cold(cold, r)),
                "$/1M req": usd,
            }
        )
    return md_table(pd.DataFrame(out)), star


def overload_lines(best: pd.DataFrame, ov: pd.DataFrame, rows_list: list[int]) -> list[str]:
    if ov.empty:
        return ["No overload data."]
    lines = []
    worst = []
    for rows in rows_list:
        parts = []
        for _, b in best[best["rows"] == rows].iterrows():
            o = ov
            for k in ("env", "stack", "rows", "batching", "replicas"):
                o = o[o[k] == b[k]]
            cells = []
            for f in (1.5, 3.0):
                x = o[o["factor"] == f]
                if x.empty:
                    cells.append("n/a")
                    continue
                x = x.iloc[0]
                cells.append(f"p99 {_int(x['p99_ms'])} ms, err {x['err_rate_pct']:.1f}%")
                if f == 3.0:
                    worst.append((x["err_rate_pct"], x["p99_ms"], b["stack"], rows))
            parts.append(f"{b['stack']} {cells[0]} / {cells[1]}")
        if parts:
            lines.append(f"rows={rows}, at 1.5x / 3x capacity@100ms: " + "; ".join(parts) + ".")
    if worst:
        e, p, s, r = min(worst)
        lines.append(f"Lowest error rate at 3x: {s} (rows={r}), {e:.1f}% errors, p99 {_int(p)} ms.")
    return lines or ["No overload data."]


def recommendations(best: pd.DataFrame, cold: pd.DataFrame, rows_list: list[int]) -> list[str]:
    b = best.copy()
    b["cold"] = [_cold(cold, r) for _, r in b.iterrows()]
    out = []

    def pick(col, rows, low=True):
        x = b[(b["rows"] == rows) & b[col].notna()]
        if x.empty:
            return None
        return x.loc[x[col].idxmin() if low else x[col].idxmax()]

    def each(col, low, fmt):
        parts = []
        for rows in rows_list:
            r = pick(col, rows, low)
            if r is not None:
                parts.append(f"{r['stack']} at rows={rows} ({fmt(r[col])})")
        return ", ".join(parts)

    t = each("capacity_rps", False, lambda v: f"{_int(v)} req/s")
    out.append(f"- Highest capacity at 50 ms: {t}.")
    t = each("usd_per_1m_req", True, lambda v: f"${v:.4f}/1M")
    out.append(f"- Lowest $/1M requests: {t}." if t else "- Lowest $/1M requests: n/a (no price).")
    m = each("rss_peak_mb", True, lambda v: f"{_int(v)} MB")
    out.append(f"- Smallest peak RSS: {m}." if m else "- Smallest peak RSS: n/a.")
    t = each("cold", True, lambda v: f"{_int(v)} s")
    out.append(f"- Fastest cold start: {t}." if t else "- Fastest cold start: n/a.")
    r1 = pick("capacity_rps", rows_list[0], False)
    rn = pick("capacity_rps", rows_list[-1], False)
    if r1 is not None and rn is not None:
        out.append(
            f"- Pick {r1['stack']} for rows={rows_list[0]} traffic and {rn['stack']} for "
            f"rows={rows_list[-1]}; pick the cheapest or smallest stack above when the SLO is "
            "met by several."
        )
    return out


def _run_ids(ladder: pd.DataFrame) -> pd.DataFrame:
    if ladder.empty:
        return pd.DataFrame({"run_id": []})
    g = ladder.groupby(["env", "run_id"]).size().reset_index(name="steps")
    return g


def build(summary_dir: Path, figures_dir: Path, out_md: Path, appendix_md: Path,
          env: str | None = None, slo: int = 50) -> Path:
    summary_dir, figures_dir = Path(summary_dir), Path(figures_dir)
    out_md, appendix_md = Path(out_md), Path(appendix_md)
    cap_all = read_csv(summary_dir, "capacity.csv")
    cold = read_csv(summary_dir, "coldstart.csv")
    ov = read_csv(summary_dir, "overload.csv")
    ladder = read_csv(summary_dir, "ladder.csv")
    cap = main_model(cap_all)
    if env:
        cap = cap[cap["env"] == env]
    envs = sorted(cap["env"].unique()) if not cap.empty else []
    best = best_configs(cap, slo) if not cap.empty else cap
    primary = envs[0] if envs else (env or "n/a")
    if envs and len(envs) > 1:
        counts = best.groupby("env").size()
        primary = counts.idxmax() if not counts.empty else primary

    rel = os.path.relpath(figures_dir, out_md.parent).replace(os.sep, "/")
    L = [
        "# Serving benchmark: FastAPI vs Ray Serve vs Triton",
        "",
        "We compared five ways to serve the same DeepFM ONNX model: FastAPI + ORT, Ray Serve + "
        "ORT, Ray Serve + Triton (gRPC and in-process), and Triton alone.",
        "Method: open-loop k6 load, same ONNX file, same ORT settings, capacity is the highest "
        "rate with p99 under the SLO (20, 50 or 100 ms) and no errors.",
        f"Date: {dt.date.today().isoformat()}. Environment: {', '.join(envs) or 'n/a'}.",
        "",
        f"## Summary (SLO p99 <= {slo} ms, best config per stack)",
        "",
    ]
    rows_list: list[int] = []
    if not best.empty:
        rows_list = headline_rows(best)
        star_any = False
        for e in envs:
            be = best[best["env"] == e]
            for rows in rows_list:
                if be[be["rows"] == rows].empty:
                    continue
                tbl, star = summary_table(be, cold, rows)
                star_any |= star
                L += [f"**{e}, rows = {rows}**", "", tbl, ""]
        if star_any:
            L += ["`*` = number comes from search steps only (not confirmed with full reps).", ""]
    else:
        L += [f"No capacity met the {slo} ms SLO.", ""]

    L += ["## Charts", ""]
    for name, alt in (
        ("latency_vs_rate", "p99 latency vs offered rate"),
        ("throughput_vs_replicas", "capacity vs replicas"),
        ("cost_per_slo", "cost per SLO"),
    ):
        L += [f"![{alt}, {primary}]({rel}/{name}_{primary}.svg)", ""]

    L += ["## Overload", ""]
    pb = best[best["env"] == primary] if not best.empty else best
    pov = ov[ov["env"] == primary] if not ov.empty else ov
    if rows_list:
        L += overload_lines(pb, pov, rows_list)
    else:
        L += ["No overload data."]
    L += ["", "## Recommendation", ""]
    if rows_list:
        L += recommendations(pb, cold[cold["env"] == primary] if not cold.empty else cold,
                             rows_list)
    else:
        L += ["- Not enough data."]
    L += ["", "Details: bench/report/APPENDIX.md"]
    out_md.write_text("\n".join(L) + "\n")

    A = ["# Appendix", "", "## Capacity (all SLOs, all configs)", ""]
    A += [md_table(cap_all) if not cap_all.empty else "No data.", "", "## Cold start", ""]
    A += [md_table(cold) if not cold.empty else "No data.", "", "## Overload", ""]
    A += [md_table(ov) if not ov.empty else "No data.", "", "## Runs", ""]
    A += [md_table(_run_ids(ladder)), ""]
    appendix_md.write_text("\n".join(A))
    return out_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--slo", type=int, default=50)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--figures", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=ROOT / "REPORT.md")
    ap.add_argument("--appendix", type=Path, default=HERE / "APPENDIX.md")
    a = ap.parse_args()
    print(build(a.summary, a.figures, a.out, a.appendix, a.env, a.slo))


if __name__ == "__main__":
    main()
