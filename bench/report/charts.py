"""Three charts for the report, written as SVG. Reads the CSVs made by bench.report.analyze."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE.parent / "results" / "summary"
DEFAULT_OUT = HERE / "figures"

# Okabe-Ito, one color per stack, used in every chart.
STACK_COLORS = {
    "fastapi_ort": "#0072B2",
    "ray_ort": "#E69F00",
    "ray_fastapi": "#56B4E9",
    "ray_triton_grpc": "#009E73",
    "ray_triton_inproc": "#CC79A7",
    "triton": "#D55E00",
}
SLOS = (20, 50, 100)
KEY = ["env", "stack", "arch", "size", "model"]
CFG = ["batching", "replicas"]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "raybenchmark",
    }
)


def read_csv(summary_dir: Path, name: str) -> pd.DataFrame:
    p = Path(summary_dir) / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:  # e.g. overload.csv under the smoke profile
        return pd.DataFrame()


def main_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one (model, arch, size): the most common one (headline model)."""
    if df.empty or "model" not in df:
        return df
    cols = [c for c in ("model", "arch", "size") if c in df]
    top = df.groupby(cols).size().sort_values(ascending=False).index[0]
    top = top if isinstance(top, tuple) else (top,)
    mask = pd.Series(True, index=df.index)
    for c, v in zip(cols, top, strict=True):
        mask &= df[c] == v
    return df[mask]


def best_configs(cap: pd.DataFrame, slo: int = 50) -> pd.DataFrame:
    """Per (env, stack, rows): the capacity row with the highest capacity_rps at `slo`."""
    c = cap[(cap["slo_ms"] == slo) & (cap["capacity_rps"] > 0)]
    if c.empty:
        return c
    idx = c.groupby(["env", "stack", "rows"])["capacity_rps"].idxmax()
    return c.loc[idx].reset_index(drop=True)


def stacks_in(df: pd.DataFrame) -> list[str]:
    present = set(df["stack"])
    return [s for s in STACK_COLORS if s in present] + sorted(present - set(STACK_COLORS))


def _color(stack: str) -> str:
    base = stack.split("[")[0]
    return STACK_COLORS.get(base, "#555555")


def _panels(n: int, height: float = 3.4):
    fig, axes = plt.subplots(1, n, figsize=(9, height), squeeze=False)
    return fig, axes[0]


def _legend(fig, axes) -> None:
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(
            list(handles.values()),
            list(handles),
            loc="lower center",
            ncol=min(len(handles), 5),
            frameon=False,
        )


def _finish(fig, path: Path) -> Path:
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return path


def latency_vs_rate(cap, ladder, env: str, out_dir: Path) -> Path | None:
    best = best_configs(cap[cap["env"] == env])
    lad = ladder[(ladder["env"] == env) & ladder["step_kind"].isin(["search", "confirm"])]
    if best.empty or lad.empty:
        print(f"note: no data for latency_vs_rate_{env}")
        return None
    rows_vals = sorted(best["rows"].unique())
    fig, axes = _panels(len(rows_vals))
    drew = False
    for ax, rows in zip(axes, rows_vals, strict=True):
        for _, b in best[best["rows"] == rows].iterrows():
            sel = lad[(lad["rows"] == rows) & (lad["stack"] == b["stack"])]
            for k in ("model", "arch", "size", "batching", "replicas"):
                sel = sel[sel[k] == b[k]]
            sel = sel[sel["rate"] > 0].groupby("rate", as_index=False)["p99_ms"].mean()
            if sel.empty:
                continue
            drew = True
            ax.plot(sel["rate"], sel["p99_ms"], marker="o", ms=3, lw=1.5,
                    color=_color(b["stack"]), label=b["stack"])
        for slo in SLOS:
            ax.axhline(slo, color="#888888", ls="--", lw=0.8)
            ax.text(1.0, slo, f"{slo} ms", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color="#666666")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"rows = {rows}")
        ax.set_xlabel("offered rate (req/s)")
        ax.set_ylabel("p99 latency (ms)")
    if not drew:
        plt.close(fig)
        print(f"note: no ladder rows matched best configs for {env}")
        return None
    fig.suptitle(f"p99 latency vs offered load, {env}", fontsize=10)
    _legend(fig, axes)
    return _finish(fig, out_dir / f"latency_vs_rate_{env}.svg")


def throughput_vs_replicas(cap, env: str, out_dir: Path) -> Path | None:
    c = cap[(cap["env"] == env) & (cap["slo_ms"] == 50)]
    if c.empty:
        print(f"note: no data for throughput_vs_replicas_{env}")
        return None
    rows_vals = sorted(c["rows"].unique())
    fig, axes = _panels(len(rows_vals))
    for ax, rows in zip(axes, rows_vals, strict=True):
        for (stack, batching), g in c[c["rows"] == rows].groupby(["stack", "batching"]):
            g = g.groupby("replicas", as_index=False)["capacity_rps"].max().sort_values("replicas")
            ax.plot(g["replicas"], g["capacity_rps"], marker="o", ms=4, lw=1.5,
                    ls="-" if batching else ":", color=_color(stack),
                    label=stack if batching else "_nolegend_")
        ax.set_title(f"rows = {rows}")
        ax.set_xlabel("replicas")
        ax.set_ylabel("capacity at p99 <= 50 ms (req/s)")
        ax.set_xticks(sorted(c["replicas"].unique()))
    fig.suptitle(f"Capacity at 50 ms vs replicas, {env} (solid: batching on, dotted: off)",
                 fontsize=10)
    _legend(fig, axes)
    return _finish(fig, out_dir / f"throughput_vs_replicas_{env}.svg")


def cost_per_slo(cap, env: str, out_dir: Path) -> Path | None:
    c = cap[cap["env"] == env]
    best = best_configs(c)
    if best.empty:
        print(f"note: no data for cost_per_slo_{env}")
        return None
    use_usd = c["usd_per_1m_req"].notna().any()
    c = c.assign(per_core=c["rows_per_s"] / c["cpu_cores"].where(c["cpu_cores"] > 0))
    col = "usd_per_1m_req" if use_usd else "per_core"
    ylabel = "USD per 1M requests" if use_usd else "rows/s per CPU core"
    rows_vals = [r for r in (1, 200) if r in set(best["rows"])] or sorted(best["rows"].unique())[:2]
    fig, axes = _panels(len(rows_vals))
    stacks = stacks_in(best)
    width = 0.8 / max(len(stacks), 1)
    for ax, rows in zip(axes, rows_vals, strict=True):
        for i, stack in enumerate(stacks):
            b = best[(best["rows"] == rows) & (best["stack"] == stack)]
            if b.empty:
                continue
            b = b.iloc[0]
            vals = []
            for slo in SLOS:
                m = c[(c["rows"] == rows) & (c["stack"] == stack) & (c["slo_ms"] == slo)]
                for k in ("model", "arch", "size", "batching", "replicas"):
                    m = m[m[k] == b[k]]
                v = m[col].iloc[0] if not m.empty else float("nan")
                vals.append(0.0 if pd.isna(v) else v)
            xs = [j + (i - (len(stacks) - 1) / 2) * width for j in range(len(SLOS))]
            ax.bar(xs, vals, width=width * 0.9, color=_color(stack), label=stack)
        ax.set_xticks(range(len(SLOS)), [f"{s} ms" for s in SLOS])
        ax.set_xlabel("p99 SLO")
        ax.set_ylabel(ylabel)
        ax.set_title(f"rows = {rows}")
    what = "Cost per 1M requests" if use_usd else "Throughput per CPU core (no price)"
    fig.suptitle(f"{what} by SLO, best config per stack, {env}", fontsize=10)
    _legend(fig, axes)
    return _finish(fig, out_dir / f"cost_per_slo_{env}.svg")


def make_all(summary_dir: Path, out_dir: Path, env: str | None = None) -> list[Path]:
    summary_dir, out_dir = Path(summary_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = main_model(read_csv(summary_dir, "capacity.csv"))
    ladder = read_csv(summary_dir, "ladder.csv")
    if cap.empty:
        print("note: capacity.csv empty or missing; no charts")
        return []
    if not ladder.empty:
        ladder = ladder.merge(
            cap[["model", "arch", "size"]].drop_duplicates(), on=["model", "arch", "size"]
        )
    envs = [env] if env else sorted(cap["env"].unique())
    written: list[Path] = []
    for e in envs:
        for p in (
            latency_vs_rate(cap, ladder, e, out_dir) if not ladder.empty else None,
            throughput_vs_replicas(cap, e, out_dir),
            cost_per_slo(cap, e, out_dir),
        ):
            if p:
                written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    for p in make_all(a.summary, a.out, a.env):
        print(p)


if __name__ == "__main__":
    main()
