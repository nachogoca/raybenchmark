"""Single-file interactive HTML report (Plotly inlined, works offline) + REPORT.md headline tables.

Usage: uv run python -m bench.report.report [--env laptop]
Writes bench/report/report.html and REPORT.md from bench/results/<env>.jsonl.
"""
from __future__ import annotations

import argparse
import html
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from bench.collect.k6parse import hist_edges
from bench.report import analyze as an

ROOT = Path(__file__).resolve().parents[2]
PALETTE = ["#2f6fb3", "#d9822b", "#3a9d6e", "#b5473a", "#7a5fb0", "#8c7b2e", "#c05a98", "#4d8f9c", "#6b6b6b",
           "#a3562b", "#2f8f5b", "#5367c7"]
STACK_LABEL = {
    "fastapi_ort": "FastAPI + ORT", "triton": "Triton", "ray_ort": "Ray + ORT", "ray_fastapi": "Ray + FastAPI",
    "ray_triton_grpc": "Ray → Triton (gRPC)", "ray_triton_inproc": "Ray + in-process Triton",
    "ray_family": "Ray family",
}
SLOS = [20, 50, 100]
_colors: dict[str, str] = {}


def color(key: str) -> str:
    return _colors.setdefault(key, PALETTE[len(_colors) % len(PALETTE)])


def label(stack: str) -> str:
    return STACK_LABEL.get(stack, stack)


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}{suffix}"


def layout(fig: go.Figure, title: str = "", xt: str = "", yt: str = "", h: int = 420, logy=False, logx=False) -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.01, font=dict(size=14)), height=h,
                      margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.2),
                      hovermode="x unified", font=dict(family="system-ui, sans-serif", size=12))
    fig.update_xaxes(title=xt, type="log" if logx else None)
    fig.update_yaxes(title=yt, type="log" if logy else None)
    return fig


def fig_html(fig: go.Figure) -> str:
    return f'<div class="fig">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</div>'


def table(rows: list[dict], cols: list[tuple[str, str]], cls: str = "") -> str:
    if not rows:
        return '<p class="muted">No data.</p>'
    head = "".join(f"<th>{esc(h)}</th>" for _, h in cols)
    body = "".join("<tr>" + "".join(f"<td>{r.get(k, '')}</td>" for k, _ in cols) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def nodata(what: str) -> str:
    return f'<p class="muted">No data: {esc(what)} did not run or produced no valid steps.</p>'


def section(sid: str, title: str, intro: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{esc(title)}</h2><p class="intro">{intro}</p>{body}</section>'


# ------------------------------------------------------------------ S1: latency / capacity
def s1(df, stack_order):
    return df[(df.scenario == "S1") & (df.rows == 200) & df.kind.isin(["grid", "refine"]) & df.valid]


def latency_section(df, order) -> str:
    d = s1(df, order)
    if d.empty:
        return section("latency", "Latency vs offered load", "", nodata("S1"))
    figs = []
    for col, name in (("p50", "p50"), ("p99", "p99"), ("p999", "p99.9")):
        fig = go.Figure()
        for st in order:
            g = d[d.stk == st].groupby("rate")[col].agg(["median", "min", "max"]).reset_index()
            if g.empty:
                continue
            c = color(st)
            fig.add_trace(go.Scatter(x=g.rate, y=g["max"], mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=st))
            fig.add_trace(go.Scatter(x=g.rate, y=g["min"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=c + "26", showlegend=False, hoverinfo="skip", legendgroup=st))
            fig.add_trace(go.Scatter(x=g.rate, y=g["median"], mode="lines+markers", name=label(st), legendgroup=st,
                                     line=dict(color=c, width=2), marker=dict(size=6)))
        if col == "p99":
            for s in SLOS:
                fig.add_hline(y=s, line=dict(dash="dot", color="#999"), annotation_text=f"SLO {s} ms", annotation_position="top left")
        figs.append(fig_html(layout(fig, f"{name} latency vs offered rate (median of reps, band = min–max)",
                                    "offered rate (req/s)", f"{name} (ms)", logy=True)))
    fig = go.Figure()
    for st in order:
        g = d[d.stk == st].groupby("rate").rps.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.rps, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    m = float(d.rate.max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="ideal", line=dict(dash="dot", color="#999")))
    figs.append(fig_html(layout(fig, "Achieved vs offered throughput (saturation)", "offered (req/s)", "achieved (req/s)")))
    return section("latency", "Latency vs offered load",
                   "Open-loop k6 at the same offered rates for every stack (rows = 200 per request, 3 repetitions, stack order "
                   "shuffled per repetition). Each stack stops two points after it misses the loosest SLO.", "".join(figs))


def spectrum_section(df, order) -> str:
    d = s1(df, order)
    rates = [r for r in (50, 100, 200) if d[d.rate == r].stk.nunique() >= 2]
    if not rates:
        return section("spectrum", "Percentile spectrum and distribution", "", nodata("S1 grid"))
    body = []
    keys = ["50", "75", "90", "95", "99", "99.9", "99.99"]
    xs = [-math.log10(1 - float(k) / 100) for k in keys]
    for r in rates:
        fig = go.Figure()
        for st in order:
            g = d[(d.stk == st) & (d.rate == r)]
            if g.empty:
                continue
            ys = [float(np.median([x["pcts"][k] for x in g.dist if x.get("pcts") and k in x["pcts"]] or [np.nan])) for k in keys]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label(st), line=dict(color=color(st)),
                                     text=[f"p{k}" for k in keys], hovertemplate="%{text}: %{y:.2f} ms"))
        fig.update_xaxes(tickvals=xs, ticktext=[f"p{k}" for k in keys])
        body.append(fig_html(layout(fig, f"Latency by percentile at {r} req/s (HDR-style spectrum)", "percentile", "latency (ms)", logy=True)))
        edges = hist_edges()
        mid = np.sqrt(edges[:-1] * edges[1:])
        fig = go.Figure()
        for st in order:
            g = d[(d.stk == st) & (d.rate == r)]
            counts = sum((np.array(x["hist"]["counts"]) for x in g.dist if x.get("hist")), np.zeros(len(mid)))
            if counts.sum() == 0:
                continue
            fig.add_trace(go.Scatter(x=mid, y=np.cumsum(counts) / counts.sum(), mode="lines", name=label(st),
                                     line=dict(color=color(st), shape="hv")))
        body.append(fig_html(layout(fig, f"Latency CDF at {r} req/s (all repetitions pooled)", "latency (ms)", "fraction of requests", logx=True)))
    return section("spectrum", "Percentile spectrum and distribution",
                   "Tail behaviour at fixed rates: the spectrum shows p50 … p99.99 on a log-of-nines axis, the CDF the whole distribution.",
                   "".join(body))


def capacity_section(df, order, cap) -> str:
    if cap.empty:
        return section("capacity", "Throughput and capacity", "", nodata("S1"))
    fig = go.Figure()
    for slo in SLOS:
        c = cap[cap.slo == slo].set_index("stk").reindex([s for s in order if s in set(cap.stk)])
        fig.add_trace(go.Bar(x=[label(s) for s in c.index], y=c.capacity, name=f"p99 ≤ {slo} ms",
                             error_y=dict(type="data", symmetric=False, array=(c.hi - c.capacity).clip(lower=0),
                                          arrayminus=(c.capacity - c.lo).clip(lower=0))))
    body = [fig_html(layout(fig, "Max sustainable throughput (MST) per SLO, 95 % bootstrap CI", "", "req/s", h=400))]
    d = df[(df.scenario == "S1") & (df.kind == "closed")]
    if not d.empty:
        g = d.groupby("stk").rps.median().reindex([s for s in order if s in set(d.stk)])
        fig = go.Figure(go.Bar(x=[label(s) for s in g.index], y=g.values, marker_color=[color(s) for s in g.index]))
        body.append(fig_html(layout(fig, "Closed-loop maximum throughput (4 virtual users per replica)", "", "req/s", h=340)))
    rows = []
    for st in order:
        pts = an.s1_points(df, st)
        if pts:
            rows.append({"stack": label(st), "knee": fmt(an.knee(pts), 0, " req/s") if an.knee(pts) else "–"})
    body.append("<h3>Knee</h3><p class='intro'>Lowest offered rate where the median p99 exceeds twice its lowest-rate value.</p>"
                + table(rows, [("stack", "Stack"), ("knee", "Latency knee")]))
    return section("capacity", "Throughput and capacity",
                   "MST = highest grid rate whose median p99 meets the SLO with zero errors and ≤ 0.1 % dropped arrivals. "
                   "Whiskers: 95 % bootstrap interval over repetitions.", "".join(body))


def fleet_section(df, order, sc) -> str:
    d = an.search_capacity(df, "S1_scale", 100)
    if d.empty:
        return section("fleet", "Fleet capacity and scaling", "", nodata("S1_scale"))
    fig = go.Figure()
    eff = []
    for st in d.stk.unique():
        g = d[d.stk == st].sort_values("replicas")
        fig.add_trace(go.Scatter(x=g.replicas, y=g.capacity, mode="lines+markers", name=label(st), line=dict(color=color(st))))
        base = float(g[g.replicas == 1].capacity.iloc[0]) if (g.replicas == 1).any() else None
        if base:
            fig.add_trace(go.Scatter(x=g.replicas, y=base * g.replicas, mode="lines", showlegend=False,
                                     line=dict(dash="dot", color=color(st), width=1)))
        for _, r in g.iterrows():
            eff.append({"stack": label(st), "replicas": int(r.replicas), "cap": int(r.capacity),
                        "eff": fmt(100 * r.capacity / (base * r.replicas), 0, " %") if base else "–",
                        "per_core": fmt(r.capacity / 6, 1)})
    body = [fig_html(layout(fig, "Capacity at p99 ≤ 100 ms vs replicas (same 6 cores split; dotted = linear)", "replicas", "req/s"))]
    body.append(table(eff, [("stack", "Stack"), ("replicas", "Replicas"), ("cap", "Capacity (req/s)"),
                            ("eff", "Scaling efficiency"), ("per_core", "req/s per core")]))
    if not sc.empty:
        rows = []
        pr = an.prices()
        for _, r in sc.iterrows():
            if not r.rps_per_vcpu:
                continue
            row = {"stack": label(r.stk)}
            for t in (1_000, 10_000, 100_000):
                v = t / r.rps_per_vcpu
                row[str(t)] = f"{v:,.0f} vCPU · ${v * pr['usd_per_vcpu_hour'] * 730:,.0f}/mo"
            rows.append(row)
        body.append("<h3>Fleet size for a target load (extrapolated linearly from the S1 SLO 50 ms capacity per vCPU)</h3>"
                    + table(rows, [("stack", "Stack"), ("1000", "1k req/s"), ("10000", "10k req/s"), ("100000", "100k req/s")]))
    return section("fleet", "Fleet capacity and scaling",
                   "The laptop 'fleet' is one box: replicas share the same 6 cores, so this measures how each stack's per-replica "
                   "overhead scales, not horizontal scale-out. Cluster fleet tests are in docs/PLAN.md.", "".join(body))


def breakdown_section(df, order, ref=100) -> str:
    d = s1(df, order)
    d = d[d.rate == ref]
    if d.empty:
        return section("breakdown", "Latency breakdown", "", nodata(f"S1 at {ref} req/s"))
    comp, queue, over, names = [], [], [], []
    for st in order:
        g = d[d.stk == st]
        if g.empty:
            continue
        client = float(np.median([x["mean_ms"] for x in g.dist if x.get("mean_ms")] or [np.nan]))
        pv = [p.get("derived") or {} for p in g.prom if p]

        def med(key, pv=pv):
            v = [x[key] for x in pv if x.get(key) is not None]
            return float(np.median(v)) if v else None
        if st == "triton":
            q, c = med("triton_queue_ms") or 0, sum(med(k) or 0 for k in ("triton_input_ms", "triton_compute_ms", "triton_output_ms"))
        elif st == "fastapi_ort":
            q, c = 0.0, med("fastapi_handler_ms") or 0
        elif st == "ray_triton_grpc":
            q, c = med("triton_queue_ms") or 0, sum(med(k) or 0 for k in ("triton_input_ms", "triton_compute_ms", "triton_output_ms"))
        else:
            q, c = 0.0, med("ray_replica_ms") or 0
        names.append(label(st))
        queue.append(q)
        comp.append(c)
        over.append(max(client - q - c, 0))
    fig = go.Figure()
    for nm, vals, col in (("queue wait", queue, "#d9822b"), ("server compute / handler", comp, "#2f6fb3"),
                          ("proxy + network + client", over, "#9aa5b1")):
        fig.add_trace(go.Bar(x=names, y=vals, name=nm, marker_color=col))
    fig.update_layout(barmode="stack")
    return section("breakdown", "Latency breakdown",
                   f"Mean latency at {ref} req/s split with server-side Prometheus metrics: Triton queue/compute counters, the FastAPI "
                   "handler timer, Ray's replica processing latency. The remainder is HTTP proxy, network and k6. "
                   "For Ray → Triton the compute segment is Triton's own; Ray's hop lands in the remainder.",
                   fig_html(layout(fig, "Mean latency decomposition (ms)", "", "ms", h=380)))


def resources_section(df, order, cold, ref=100) -> str:
    d = s1(df, order)
    if d.empty:
        return section("resources", "Resource utilisation", "", nodata("S1"))
    body = []
    g = d[d.rate == ref]
    comps = sorted({c for x in g.comp for c in x})
    if comps:
        fig = go.Figure()
        for c in comps:
            ys = [float(np.median([x.get(c, 0) for x in g[g.stk == st].comp] or [0])) for st in order if st in set(g.stk)]
            fig.add_trace(go.Bar(x=[label(s) for s in order if s in set(g.stk)], y=ys, name=c))
        fig.update_layout(barmode="stack")
        body.append(fig_html(layout(fig, f"CPU cores busy by component at {ref} req/s", "", "cores", h=400)))
    fig = go.Figure()
    for st in order:
        x = d[(d.stk == st) & d.cpu.notna() & (d.rps > 0)].groupby("rate").apply(
            lambda z: float(np.median(z.cpu / z.rps * 1000)), include_groups=False).reset_index(name="v")
        fig.add_trace(go.Scatter(x=x.rate, y=x.v, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    body.append(fig_html(layout(fig, "CPU cost per request (core-milliseconds) vs offered rate", "offered rate (req/s)", "core-ms / request", logy=True)))
    cs = cold[cold.scenario == "S1"].groupby("stk").agg(ready=("ready_s", "median"), idle=("idle_rss_mb", "median"),
                                                          image=("image_mb", "median")).reindex([s for s in order if s in set(cold[cold.scenario == "S1"].stk)])
    peak = d.groupby("stk").rss.max()
    rows = [{"stack": label(s), "ready": fmt(r.ready, 1, " s"), "idle": fmt(r.idle, 0, " MB"),
             "peak": fmt(peak.get(s), 0, " MB"), "image": fmt(r.image, 0, " MB")} for s, r in cs.iterrows()]
    body.append(table(rows, [("stack", "Stack"), ("ready", "Cold start"), ("idle", "Idle RSS"), ("peak", "Peak RSS under load"), ("image", "Image")]))
    return section("resources", "Resource utilisation",
                   "Per-process CPU (cgroup + /proc) split by role: HTTP proxy / HAProxy, Ray replicas, raylet/GCS/controller, Triton, uvicorn.",
                   "".join(body))


def cost_section(sc) -> str:
    if sc.empty:
        return section("cost", "Cost efficiency", "", nodata("S1"))
    pr = an.prices()
    sc = sc.dropna(subset=["usd_per_m_alloc"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_alloc, name="allocated (6 vCPU reserved)"))
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_used, name="consumed (measured busy cores)"))
    body = [fig_html(layout(fig, "$ per 1M requests at the p99 ≤ 50 ms capacity", "", "USD / 1M req", h=380)),
            ]
    fig = go.Figure()
    for _, r in sc.iterrows():
        if r.p99_at_ref:
            fig.add_trace(go.Scatter(x=[r.usd_per_m_alloc], y=[r.p99_at_ref], mode="markers+text", text=[label(r.stk)],
                                     textposition="top center", marker=dict(size=14, color=color(r.stk)), showlegend=False))
    body.append(fig_html(layout(fig, "Cost vs latency (lower-left is better)", "USD / 1M req (allocated)", "p99 at 100 req/s (ms)")))
    note = (f"Rates: {esc(pr['name'])}, ${pr['usd_per_vcpu_hour']}/vCPU-h + ${pr['usd_per_gb_hour']}/GB-h, as of {esc(pr['as_of'])}, "
            f"<b>{'verified' if pr['verified'] else 'NOT verified against the price page'}</b> (<code>config/prices.yaml</code>). "
            "Laptop CPU time is mapped to those rates; absolute dollars are indicative, the ranking is what matters.")
    return section("cost", "Cost efficiency", note, "".join(body))


# ------------------------------------------------------------------ Ray Serve feature scenarios
def ingress_section(df) -> str:
    d = df[(df.scenario == "I1") & (df.rows == 200)]
    if d.empty:
        return "<h3>Ingress (I1)</h3>" + nodata("I1")
    cap = an.search_capacity(df, "I1", 100).set_index("variant").capacity
    cap50 = an.search_capacity(df, "I1", 50).set_index("variant").capacity
    fig = go.Figure()
    for v in d.variant.unique():
        g = d[(d.variant == v) & (d.kind == "grid")].groupby("rate").p99.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
    out = [fig_html(layout(fig, "p99 at fixed rates by ingress", "offered rate (req/s)", "p99 (ms)", logy=True, h=360))]
    fig = go.Figure()
    for slo, c in ((50, cap50), (100, cap)):
        fig.add_trace(go.Bar(x=list(c.index), y=list(c.values), name=f"p99 ≤ {slo} ms"))
    out.append(fig_html(layout(fig, "Capacity by ingress (short search steps)", "", "req/s", h=340)))
    return "<h3>Ingress (I1): Python proxy vs HAProxy vs gRPC</h3>" + "".join(out)


def family_section(df, cold_v) -> str:
    out = []
    d = df[df.scenario == "F1"]
    if d.empty:
        out.append("<h3>Multiplexing and model families (F1, F2)</h3>" + nodata("F1"))
    else:
        c50, c100 = an.search_capacity(df, "F1", 50), an.search_capacity(df, "F1", 100)
        fx = d[d.kind == "family_fixed"]
        rows = []
        for v in c100.variant:
            f = fx[fx.variant == v]
            dv = (f.prom.iloc[0].get("derived") if len(f) and f.prom.iloc[0] else None) or {}
            miss = (dv["mux_loads"] / dv["mux_requests"]) if dv.get("mux_requests") else None
            rows.append({"variant": v, "stack": label(c100[c100.variant == v].stk.iloc[0]),
                         "c50": int(c50[c50.variant == v].capacity.iloc[0]) if (c50.variant == v).any() else "–",
                         "c100": int(c100[c100.variant == v].capacity.iloc[0]),
                         "p99": fmt(f.p99.median() if len(f) else None, 1, " ms"),
                         "rss": fmt(f.rss.median() if len(f) else None, 0, " MB"),
                         "cpu": fmt(f.cpu.median() if len(f) else None, 2),
                         "miss": fmt(100 * miss, 1, " %") if miss is not None else "–"})
        fig = go.Figure(go.Bar(x=c100.variant, y=c100.capacity, name="p99 ≤ 100 ms"))
        fig.add_trace(go.Bar(x=c50.variant, y=c50.capacity, name="p99 ≤ 50 ms"))
        out.append("<h3>Multiplexing and model families (F1, F2)</h3>"
                   + fig_html(layout(fig, "Family capacity under Zipf(1.1) traffic", "", "req/s", h=360))
                   + table(rows, [("variant", "Config"), ("stack", "Stack"), ("c50", "Capacity @50 ms"), ("c100", "Capacity @100 ms"),
                                  ("p99", "p99 at 60 req/s"), ("rss", "Peak RSS"), ("cpu", "Cores busy"), ("miss", "Model load rate")]))
    if not cold_v.empty:
        fig = go.Figure()
        for (v, kind), g in cold_v.groupby(["variant", "kind"]):
            fig.add_trace(go.Box(y=g.first_ms, name=f"{v} ({kind})", boxpoints="all"))
        fig.add_trace(go.Scatter(x=[f"{v} ({k})" for (v, k), g in cold_v.groupby(["variant", "kind"])],
                                 y=[g.steady_ms.median() for _, g in cold_v.groupby(["variant", "kind"])],
                                 mode="markers", name="steady-state", marker=dict(symbol="line-ew", size=18, color="#333", line=dict(width=2))))
        out.append(fig_html(layout(fig, "First-request latency of a never-used (cold) or evicted variant", "", "ms", logy=True, h=380)))
    return "".join(out)


def composition_section(df) -> str:
    out = []
    d = df[(df.scenario == "F3") & (df.kind == "fanout")]
    if d.empty:
        out.append("<h3>Fan-out (F3)</h3>" + nodata("F3"))
    else:
        d = d.assign(system=d.variant.str.replace(r"_k\d+$", "", regex=True))
        f1, f2 = go.Figure(), go.Figure()
        for s, g in d.groupby("system"):
            a = g.groupby("k").p99.median().reset_index()
            f1.add_trace(go.Scatter(x=a.k, y=a.p99, mode="lines+markers", name=s, line=dict(color=color(s))))
            b = g.assign(v=g.cpu / g.rps * 1000).groupby("k").v.median().reset_index()
            f2.add_trace(go.Scatter(x=b.k, y=b.v, mode="lines+markers", name=s, line=dict(color=color(s))))
        out.append("<h3>Fan-out (F3): one request scored by K variants</h3>"
                   + fig_html(layout(f1, "p99 vs K at 40 req/s", "K variants per request", "p99 (ms)", logy=True, h=340))
                   + fig_html(layout(f2, "CPU per request vs K", "K variants per request", "core-ms / request", h=340)))
    p = df[(df.scenario == "F5") & (df.kind == "grid")]
    if p.empty:
        out.append("<h3>Pipeline (F5)</h3>" + nodata("F5"))
    else:
        fig = go.Figure()
        for v, g in p.groupby("variant"):
            a = g.groupby("rate").p99.median().reset_index()
            fig.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
        cap = an.search_capacity(df, "F5", 100)
        out.append("<h3>Pipeline (F5): featurize → routed model → postprocess</h3>"
                   + fig_html(layout(fig, "Pipeline p99 vs offered rate", "req/s", "p99 (ms)", logy=True, h=340))
                   + table([{"v": r.variant, "c": int(r.capacity)} for r in cap.itertuples()], [("v", "Stack"), ("c", "Capacity @100 ms (req/s)")]))
    return "".join(out)


def _rel(series_t0, t_measure):  # seconds offset of k6 series start vs t_measure
    return (series_t0 - t_measure) if series_t0 and t_measure else 0.0


def autoscale_section(df, slo=50) -> str:
    d = df[(df.scenario == "A1") & (df.kind == "burst")]
    if d.empty:
        return "<h3>Autoscaling under a 10× burst (A1)</h3>" + nodata("A1")
    f_lat, f_rep, f_err = go.Figure(), go.Figure(), go.Figure()
    rows = []
    for pol, g in d.groupby("policy"):
        c = color(pol)
        pre = int(g.iloc[0].steps.split(",")[0].split(":")[1])
        rec, errs, p99b, cores, maxr = [], [], [], [], []
        for i, r in enumerate(g.itertuples()):
            ser = (r.dist or {}).get("series") or []
            rec.append(an.recovery_s(ser, pre, slo))
            errs.append(sum(s["err"] for s in ser))
            p99b.append(np.median([s["p99"] for s in ser if s["t"] >= pre and s["p99"] is not None] or [np.nan]))
            cores.append(r.core_s)
            gauge = (r.prom.get("gauges") or {}).get("ray_serve_deployment_replica_healthy") or []
            maxr.append(max(gauge) if gauge else None)
            if i == 0 and ser:
                t = [s["t"] for s in ser]
                f_lat.add_trace(go.Scatter(x=t, y=[s["p99"] for s in ser], name=pol, mode="lines", line=dict(color=c)))
                f_err.add_trace(go.Scatter(x=t, y=[s["err"] for s in ser], name=pol, mode="lines", line=dict(color=c)))
                pt = r.prom.get("t") or []
                if gauge and pt:
                    off = (r.t_measure or pt[0])
                    f_rep.add_trace(go.Scatter(x=[x - off for x in pt], y=gauge, name=pol, mode="lines", line=dict(color=c, shape="hv")))
        rows.append({"p": pol, "rec": fmt(np.nanmedian([x for x in rec if x is not None] or [np.nan]), 0, " s") if any(x is not None for x in rec) else "never",
                     "err": fmt(np.median(errs), 0), "p99": fmt(np.nanmedian(p99b), 1, " ms"),
                     "core": fmt(np.nanmedian([x for x in cores if x] or [np.nan]), 0), "max": fmt(np.nanmax([x for x in maxr if x] or [np.nan]), 0)})
    for f in (f_lat, f_rep, f_err):
        f.add_vrect(x0=pre, x1=pre + int(df[df.scenario == "A1"].steps.iloc[0].split(",")[1].split(":")[1]),
                    fillcolor="#b5473a", opacity=0.07, line_width=0, annotation_text="10× burst", annotation_position="top left")
    return ("<h3>Autoscaling under a 10× burst (A1)</h3>"
            + fig_html(layout(f_lat, "p99 per second during the burst (first repetition)", "seconds into measurement", "p99 (ms)", logy=True, h=360))
            + fig_html(layout(f_rep, "Healthy replicas over time", "seconds (0 = start of measurement)", "replicas", h=300))
            + fig_html(layout(f_err, "Errors per second", "seconds into measurement", "errors", h=300))
            + table(rows, [("p", "Policy"), ("rec", f"Time to p99 ≤ {slo} ms after burst start"), ("err", "Errors"),
                           ("p99", "Median p99 after burst start"), ("core", "Core-seconds"), ("max", "Max replicas")]))


def backpressure_section(df) -> str:
    d = df[(df.scenario == "R1") & df.kind.isin(["search", "overload"])]
    if d.empty:
        return "<h3>Backpressure (R1)</h3>" + nodata("R1")
    f1, f2 = go.Figure(), go.Figure()
    rows = []
    for v, g in d.groupby("variant"):
        a = g.groupby("rate").agg(ok=("ok", "median"), dur=("duration_s", "median"), p99=("p99", "median")).reset_index()
        f1.add_trace(go.Scatter(x=a.rate, y=a.ok / a.dur, mode="lines+markers", name=v, line=dict(color=color(v))))
        f2.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
    mx = float(d.rate.max())
    f1.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="ideal", line=dict(dash="dot", color="#999")))
    for r in d[d.kind == "overload"].itertuples():
        codes = ", ".join(f"{k}: {v}" for k, v in r.error_codes.items()) or "none"
        rows.append({"v": r.variant, "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0),
                     "p99": fmt(r.p99, 1, " ms"), "codes": codes})
    return ("<h3>Backpressure beyond capacity (R1)</h3>"
            + fig_html(layout(f1, "Goodput (successful req/s) vs offered rate", "offered (req/s)", "goodput (req/s)", h=360))
            + fig_html(layout(f2, "p99 of successful requests vs offered rate", "offered (req/s)", "p99 (ms)", logy=True, h=340))
            + table(rows, [("v", "Config"), ("f", "× capacity"), ("rate", "Offered"), ("good", "Goodput"), ("p99", "p99 (ok)"), ("codes", "Non-200 responses")]))


def failure_section(df) -> str:
    d = df[(df.scenario == "R2") & (df.kind == "failure")]
    if d.empty:
        return "<h3>Failure recovery (R2)</h3>" + nodata("R2")
    fig, fig2 = go.Figure(), go.Figure()
    rows = []
    for pol, g in d.groupby("policy"):
        recs, errs, win = [], [], []
        for i, r in enumerate(g.itertuples()):
            ser = (r.dist or {}).get("series") or []
            if not ser:
                continue
            kill = int((r.hook_at or 0) - (r.t_measure or 0))
            pre = [s["p99"] for s in ser if s["t"] < kill and s["p99"] is not None]
            thr = 2 * float(np.median(pre)) if pre else 100.0
            recs.append(an.recovery_s(ser, kill, thr, hold=3))
            errs.append(sum(s["err"] for s in ser))
            win.append(sum(1 for s in ser if s["err"] > 0))
            if i == 0:
                t = [s["t"] - kill for s in ser]
                fig.add_trace(go.Scatter(x=t, y=[s["err"] for s in ser], name=pol, line=dict(color=color(pol))))
                fig2.add_trace(go.Scatter(x=t, y=[s["p99"] for s in ser], name=pol, line=dict(color=color(pol))))
        rows.append({"p": pol, "err": fmt(np.median(errs) if errs else None, 0), "win": fmt(np.median(win) if win else None, 0, " s"),
                     "rec": fmt(np.nanmedian([x for x in recs if x is not None] or [np.nan]), 0, " s")})
    for f in (fig, fig2):
        f.add_vline(x=0, line=dict(color="#b5473a", dash="dash"), annotation_text="kill")
    return ("<h3>Failure recovery (R2): kill one serving process under steady load</h3>"
            + fig_html(layout(fig, "Errors per second around the kill (first repetition)", "seconds relative to kill", "errors", h=320))
            + fig_html(layout(fig2, "p99 per second around the kill", "seconds relative to kill", "p99 (ms)", logy=True, h=320))
            + table(rows, [("p", "Stack"), ("err", "Errors (median)"), ("win", "Seconds with errors"), ("rec", "Seconds to p99 < 2× baseline")]))


def overload_section(df, order, cold) -> str:
    d = df[(df.scenario == "S1") & (df.kind == "overload")]
    rows = [{"s": label(r.stk), "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0), "p99": fmt(r.p99, 0, " ms"),
             "err": r.err, "drop": r.drop} for r in d.sort_values(["stk", "factor"]).itertuples()]
    return section("overload", "Overload behaviour",
                   "Offered rate at 1.5× and 3× the stack's p99 ≤ 100 ms capacity (first repetition). Goodput = successful requests per second.",
                   table(rows, [("s", "Stack"), ("f", "× capacity"), ("rate", "Offered"), ("good", "Goodput"), ("p99", "p99"),
                                ("err", "Errors"), ("drop", "Dropped arrivals")]))


# ------------------------------------------------------------------ summary, methodology, appendix
def summary_section(df, sc, cap, cold_v) -> str:
    cards = []

    def card(title, value, sub):
        cards.append(f'<div class="card"><div class="ct">{esc(title)}</div><div class="cv">{value}</div><div class="cs">{sub}</div></div>')
    if not sc.empty:
        b = sc.iloc[0]
        card("Highest capacity (p99 ≤ 50 ms)", esc(label(b.stk)), f"{b.capacity} req/s (95 % CI {b.ci})")
        c = sc.dropna(subset=["usd_per_m_alloc"]).sort_values("usd_per_m_alloc")
        if len(c):
            card("Lowest $ / 1M requests", esc(label(c.iloc[0].stk)), f"${c.iloc[0].usd_per_m_alloc:.3f} allocated")
        e = sc.dropna(subset=["core_ms_per_req"]).sort_values("core_ms_per_req")
        if len(e):
            card("Most CPU-efficient", esc(label(e.iloc[0].stk)), f"{e.iloc[0].core_ms_per_req:.2f} core-ms/request")
        f = sc.dropna(subset=["cold_s"]).sort_values("cold_s")
        if len(f):
            card("Fastest cold start", esc(label(f.iloc[0].stk)), f"{f.iloc[0].cold_s:.0f} s")
    i1 = an.search_capacity(df, "I1", 50)
    if len(i1):
        top = i1.sort_values("capacity", ascending=False).iloc[0]
        card("Best Ray ingress (I1)", esc(top.variant), f"{top.capacity} req/s at 50 ms")
    f1 = an.search_capacity(df, "F1", 100)
    if len(f1):
        top = f1.sort_values("capacity", ascending=False).iloc[0]
        card("Best family layout (F1)", esc(top.variant), f"{top.capacity} req/s at 100 ms")
    table_html = ""
    if not sc.empty:
        rows = [{"stack": label(r.stk), "cap": r.capacity, "ci": r.ci, "p99": fmt(r.p99_at_ref, 1, " ms"), "cores": fmt(r.cores_at_cap, 2),
                 "cms": fmt(r.core_ms_per_req, 2), "rpv": fmt(r.rps_per_vcpu, 1), "rss": fmt(r.rss_mb, 0, " MB"),
                 "cold": fmt(r.cold_s, 0, " s"), "ca": fmt(r.usd_per_m_alloc, 3), "cu": fmt(r.usd_per_m_used, 3)} for r in sc.itertuples()]
        table_html = table(rows, [("stack", "Stack"), ("cap", "Capacity @50 ms (req/s)"), ("ci", "95 % CI"), ("p99", "p99 @100 req/s"),
                                  ("cores", "Cores busy @cap"), ("cms", "core-ms / req"), ("rpv", "req/s / vCPU"), ("rss", "Peak RSS"),
                                  ("cold", "Cold start"), ("ca", "$/1M (alloc)"), ("cu", "$/1M (used)")], cls="score")
    return section("summary", "Executive summary", "DeepFM small (ONNX), 200 rows per request, KServe v2 JSON, k6 open-loop. "
                   "Single laptop, 6 serving cores. Numbers are relative: use them to rank stacks, not to size a cloud fleet.",
                   f'<div class="cards">{"".join(cards)}</div>{table_html}')


def methodology_section(df, meta, rows) -> str:
    v = meta.get("versions", {})
    m = an.matrix()
    flat = "".join(f"<tr><td>{esc(k)}</td><td>{esc(val)}</td></tr>" for k, val in sorted(
        (v if isinstance(v, dict) else {}).items()) if not isinstance(val, (dict, list)))
    pf = "".join(f"<li>{'✔' if p['ok'] else '✘'} {esc(p['check'])}</li>" for p in meta.get("preflight", []))
    inval = int((~df.valid).sum()) if len(df) else 0
    return section("method", "Methodology and environment",
                   "How the numbers were produced; read this before trusting a comparison.",
                   f"""<ul>
<li><b>Load:</b> k6 open-loop (constant-arrival-rate), pinned to cores 6-7; serving containers pinned to cores 0-5, identical memory limit.</li>
<li><b>Capacity (MST):</b> highest grid rate with median-of-3 p99 ≤ SLO ({', '.join(map(str, m['slos_ms']))} ms), zero errors, ≤ 0.1 % dropped arrivals; CI by bootstrapping repetitions.</li>
<li><b>Order:</b> stack order shuffled per repetition so thermal drift does not favour one stack. Warm-up excluded from every statistic.</li>
<li><b>Validity:</b> a step is invalid (excluded from capacity, listed in the appendix) when k6 used &gt; 80 % of its cores, the CPU throttled, or a container restarted. {inval} of {len(df)} steps invalid.</li>
<li><b>Server metrics:</b> Ray, Triton and FastAPI Prometheus counters sampled at 1 Hz; per-process CPU from cgroups and /proc.</li>
<li><b>Model:</b> seeded random weights, ONNX Runtime CPU, same graph optimisation level in every stack; Triton config mirrors the ORT settings.</li>
<li><b>Single machine:</b> serving and load generator share one CPU package and memory bus; absolute numbers are not cloud numbers.</li></ul>
<h3>Preflight (at run start)</h3><ul>{pf or '<li>not recorded</li>'}</ul>
<h3>Versions</h3><div class="tw"><table><tbody>{flat}</tbody></table></div>""")


def appendix_section(df, rows) -> str:
    cov = df.groupby(["scenario", "stk"]).agg(steps=("valid", "size"), invalid=("valid", lambda s: int((~s).sum()))).reset_index()
    cov_rows = [{"s": r.scenario, "st": label(r.stk), "n": r.steps, "i": r.invalid} for r in cov.itertuples()]
    bad = [{"s": r.scenario, "st": label(r.stk), "v": r.variant, "rate": r.rate, "why": esc(r.invalid)} for r in df[~df.valid].itertuples()]
    allrows = [{"s": r.scenario, "v": r.variant, "st": r.stk, "k": r.kind, "rate": r.rate or r.vus, "rep": r.rep,
                "p50": fmt(r.p50, 2), "p99": fmt(r.p99, 2), "ok": r.ok, "err": r.err, "drop": r.drop, "cpu": fmt(r.cpu, 2),
                "rss": fmt(r.rss, 0), "valid": "yes" if r.valid else "NO"} for r in df.itertuples()]
    return section("appendix", "Appendix", "Coverage, invalid steps and every measured step (click a header to sort).",
                   "<h3>Coverage</h3>" + table(cov_rows, [("s", "Scenario"), ("st", "Stack"), ("n", "Steps"), ("i", "Invalid")])
                   + "<h3>Invalid steps</h3>" + table(bad, [("s", "Scenario"), ("st", "Stack"), ("v", "Variant"), ("rate", "Rate"), ("why", "Reason")])
                   + "<h3>All steps</h3>" + table(allrows, [("s", "Scenario"), ("v", "Variant"), ("st", "Stack"), ("k", "Kind"), ("rate", "Rate/VUs"),
                                                          ("rep", "Rep"), ("p50", "p50 ms"), ("p99", "p99 ms"), ("ok", "OK"), ("err", "Err"),
                                                          ("drop", "Dropped"), ("cpu", "Cores"), ("rss", "RSS MB"), ("valid", "Valid")], cls="sortable")
                   + "<h3>Reproduce</h3><pre>just models payloads build\njust preflight\njust overnight   # or: just smoke\njust report</pre>")


CSS = """
:root{--bg:#fff;--fg:#1c2530;--mut:#66727f;--line:#e3e8ee;--card:#f6f8fa;--acc:#2f6fb3}
@media (prefers-color-scheme:dark){:root{--bg:#14181d;--fg:#e6ebf0;--mut:#98a4b0;--line:#2a323b;--card:#1b2127;--acc:#6fa8e0}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{padding:28px 32px 8px;border-bottom:1px solid var(--line)}header h1{margin:0 0 4px;font-size:24px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 32px;z-index:5;display:flex;flex-wrap:wrap;gap:4px 16px}
nav a{color:var(--acc);text-decoration:none;font-size:13px}main{max-width:1180px;margin:0 auto;padding:0 32px 80px}
section{padding-top:28px}h2{border-bottom:1px solid var(--line);padding-bottom:6px}h3{margin:26px 0 6px}
.intro,.muted{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.ct{font-size:12px;color:var(--mut)}.cv{font-size:18px;font-weight:600;margin:2px 0}.cs{font-size:12px;color:var(--mut)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:13px;min-width:60%}
th,td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left;white-space:nowrap}th{background:var(--card)}
.sortable th{cursor:pointer}.fig{margin:8px 0}pre{background:var(--card);padding:10px;border-radius:6px;overflow-x:auto}
@media (max-width:640px){main,header,nav{padding-left:16px;padding-right:16px}}
"""
SORT_JS = """document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{
const b=t.tBodies[0],r=[...b.rows],d=th.dataset.d=th.dataset.d==='1'?'-1':'1';
r.sort((x,y)=>{const a=x.cells[i].textContent.replace(/[, ]/g,''),c=y.cells[i].textContent.replace(/[, ]/g,'');
const n=parseFloat(a),m=parseFloat(c);return (isNaN(n)||isNaN(m)?a.localeCompare(c):n-m)*d});r.forEach(x=>b.appendChild(x))}));"""


def to_markdown(sc: pd.DataFrame, df, env: str) -> str:
    lines = ["# Serving benchmark: results", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from `bench/results/{env}.jsonl`. "
             "Full interactive report: `bench/report/report.html` (open locally).", ""]
    if not sc.empty:
        pr = an.prices()
        lines += ["## Scorecard (SLO p99 ≤ 50 ms, DeepFM small, 200 rows/request)", "",
                  "| Stack | Capacity req/s | 95 % CI | p99 @100 req/s | core-ms/req | req/s/vCPU | Peak RSS | Cold start | $/1M alloc | $/1M used |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sc.itertuples():
            lines.append(f"| {label(r.stk)} | {r.capacity} | {r.ci} | {fmt(r.p99_at_ref, 1, ' ms')} | {fmt(r.core_ms_per_req, 2)} | "
                         f"{fmt(r.rps_per_vcpu, 1)} | {fmt(r.rss_mb, 0, ' MB')} | {fmt(r.cold_s, 0, ' s')} | {fmt(r.usd_per_m_alloc, 3)} | {fmt(r.usd_per_m_used, 3)} |")
        lines += ["", f"Prices: {pr['name']} ${pr['usd_per_vcpu_hour']}/vCPU-h, ${pr['usd_per_gb_hour']}/GB-h "
                      f"({'verified' if pr['verified'] else 'NOT verified'}, as of {pr['as_of']}).", ""]
    for sid, title in (("I1", "Ray ingress (capacity at p99 ≤ 50 ms, short search)"), ("F1", "Model family (capacity at p99 ≤ 100 ms)"),
                       ("R1", "Backpressure (capacity at p99 ≤ 100 ms)")):
        c = an.search_capacity(df, sid, 50 if sid == "I1" else 100)
        if len(c):
            lines += [f"## {title}", "", "| Config | Capacity req/s |", "|---|---|"] + [f"| {r.variant} | {r.capacity} |" for r in c.itertuples()] + [""]
    return "\n".join(lines)


def build(env: str = "laptop", out_html: Path | None = None, out_md: Path | None = None) -> Path:
    rows = an.load_rows(env)
    df = an.flatten(rows)
    if df.empty:
        raise SystemExit(f"no step rows in bench/results/{env}.jsonl; run `just smoke` or `just overnight` first")
    cold, cold_v, meta = an.coldstarts(rows), an.cold_variants(rows), an.latest_run_meta()
    order = [s for s in ("fastapi_ort", "triton", "ray_ort", "ray_fastapi", "ray_triton_grpc", "ray_triton_inproc")
             if s in set(df[df.scenario == "S1"].stk)]
    for s in order:
        color(s)
    cap = an.s1_capacity(df, SLOS) if order else pd.DataFrame()
    sc = an.scorecard(df, cold) if order else pd.DataFrame()
    ray_features = section("ray", "Ray Serve features",
                           "Capabilities beyond a plain HTTP replica: ingress choices, multiplexing, composition, autoscaling, backpressure, fault tolerance.",
                           ingress_section(df) + family_section(df, cold_v) + composition_section(df) + autoscale_section(df)
                           + backpressure_section(df) + failure_section(df))
    parts = [summary_section(df, sc, cap, cold_v), methodology_section(df, meta, rows), latency_section(df, order),
             spectrum_section(df, order), capacity_section(df, order, cap), fleet_section(df, order, sc),
             breakdown_section(df, order), resources_section(df, order, cold), cost_section(sc), ray_features,
             overload_section(df, order, cold), appendix_section(df, rows)]
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("summary", "Summary"), ("method", "Method"), ("latency", "Latency"), ("spectrum", "Percentiles"),
                  ("capacity", "Capacity"), ("fleet", "Fleet"), ("breakdown", "Breakdown"), ("resources", "Resources"), ("cost", "Cost"),
                  ("ray", "Ray Serve features"), ("overload", "Overload"), ("appendix", "Appendix")))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>Serving benchmark report</title><style>{CSS}</style><script>{get_plotlyjs()}</script></head><body>"
           f"<header><h1>Serving benchmark: FastAPI vs Ray Serve vs Triton</h1><div class='muted'>env {esc(env)} · run {esc(meta.get('run_id', '–'))} · "
           f"{datetime.now():%Y-%m-%d} · {len(df)} measured steps</div></header><nav>{nav}</nav><main>{''.join(parts)}</main>"
           f"<script>{SORT_JS}</script></body></html>")
    out = out_html or ROOT / "bench" / "report" / "report.html"
    out.write_text(doc)
    (out_md or ROOT / "REPORT.md").write_text(to_markdown(sc, df, env))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="laptop")
    out = build(ap.parse_args().env)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and REPORT.md")


if __name__ == "__main__":
    main()
