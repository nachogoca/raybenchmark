"""Single-file interactive HTML report (Plotly inlined, works offline) + REPORT.md headline tables.

Usage: uv run python -m bench.report.report [--env laptop]
Writes bench/report/report.html and REPORT.md from bench/results/<env>.jsonl.
"""
from __future__ import annotations

import argparse
import html
import math
import warnings
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
warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
_colors: dict[str, str] = {}

# Plain names and explanations of every test. The ids are only data keys; readers never see them.
TESTS = {
    "S1": ("Main speed test",
           "How fast is each stack, and how much load can it take?",
           "We send more and more requests per second to all stacks. Each load is tried 3 times, in a shuffled order. "
           "Then we send 1.5× and 3× the capacity, and we run a top speed test.",
           "Latency at each load, the load where p99 breaks the limit, successful requests when overloaded, top speed."),
    "S1_scale": ("More replicas",
                 "Does capacity grow when we add copies of the server?",
                 "We run Ray, FastAPI and Triton with 1, 2 and 4 replicas. All replicas share the same 6 CPUs.",
                 "Capacity for each number of replicas. This shows the cost of each extra replica, not real scale-out."),
    "S1_batch": ("Batching",
                 "Does grouping requests help?",
                 "Ray and Triton, with batching off and on. Ray waits 2 or 5 ms to fill a group. We try small requests (1 row) and big ones (200 rows).",
                 "Capacity for each setting."),
    "I1": ("Ingress",
           "How much time does the entry point cost?",
           "We compare the three Ray ingress options: the Python proxy, HAProxy and gRPC. We also compare Triton over HTTP and over gRPC.",
           "p99 at fixed loads, and capacity."),
    "F1": ("Many models in one service",
           "How do we serve 16 models at once, when a few get most of the requests?",
           "Ray with a small model cache, Ray with a big cache, one Ray deployment per model, Triton with 16 models, FastAPI with 16 models. "
           "We also grow the Ray family from 4 to 64 models.",
           "Capacity, memory, and how often a model must be loaded."),
    "F2": ("First request to a new model",
           "How slow is the first request to a model that is not loaded yet?",
           "We ask for a model that was never used, 5 times per stack. We compare the first reply with the later ones. "
           "For Ray we also ask for a model that was pushed out of the cache.",
           "First reply time vs normal reply time."),
    "F3": ("One request, many models",
           "What does it cost to ask 1, 4 or 8 models for one answer and average them?",
           "Ray, FastAPI and a Triton ensemble, at a fixed load of 40 requests per second.",
           "p99 and CPU time per request for each number of models."),
    "F5": ("Steps in a row",
           "What does a chain of steps cost?",
           "Step 1 prepares the data. Step 2 calls the right model. Step 3 fixes the score. We compare Ray, FastAPI and a Triton ensemble.",
           "p99 as load goes up, and capacity."),
    "A1": ("Auto-scaling when load jumps 10×",
           "How fast does Ray add replicas when the load jumps?",
           "The load goes from 30 to 300 requests per second for 90 seconds, then back. We compare fixed fleets (1 and 4 replicas) "
           "with Ray auto-scaling under three settings. Each setting starts from a fresh service, 3 times.",
           "Seconds until latency is normal again, errors, CPU used, number of replicas over time."),
    "R1": ("Too much load",
           "What does each stack do when we send more than it can take?",
           "We send 1.5× and 3× its capacity. We compare 'no limit' with 'a limit of 50 waiting requests' in Ray, FastAPI and Triton.",
           "Successful requests per second, rejected requests, and p99 of the successful ones."),
    "R2": ("A process dies",
           "What happens if a server process dies during steady load?",
           "We send 100 requests per second and kill one Ray replica, one FastAPI worker, or the Triton container. 3 times each.",
           "Errors, and seconds until latency is normal again."),
}
PRICE_NAMES = {"gke_n2": "Google Cloud n2 machines", "eks_m6i": "AWS m6i machines"}
VERSION_KEYS = ("python", "ray", "onnxruntime", "tritonclient", "fastapi", "uvicorn", "numpy", "k6",
                "image nvcr.io/nvidia/tritonserver:26.10-py3")


def tname(key: str) -> str:
    return TESTS.get(key, (key,))[0]




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
    key = what.split(" ")[0]
    if key in TESTS:
        what = tname(key) + what[len(key):]
    return f'<p class="muted">No data: the test "{esc(what)}" did not run, or it had no good steps.</p>'


def section(sid: str, title: str, intro: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{esc(title)}</h2><p class="intro">{intro}</p>{body}</section>'


# ------------------------------------------------------------------ S1: latency / capacity
def s1(df, stack_order):
    return df[(df.scenario == "S1") & (df.rows == 200) & df.kind.isin(["grid", "refine"]) & df.valid]


def latency_section(df, order) -> str:
    d = s1(df, order)
    if d.empty:
        return section("latency", "Latency at each load", "", nodata("S1"))
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
                fig.add_hline(y=s, line=dict(dash="dot", color="#999"), annotation_text=f"limit {s} ms", annotation_position="top left")
        figs.append(fig_html(layout(fig, f"{name} latency as load goes up (line = middle of 3 runs, shade = lowest to highest)",
                                    "load (requests per second)", f"{name} (ms)", logy=True)))
    fig = go.Figure()
    for st in order:
        g = d[d.stk == st].groupby("rate").rps.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.rps, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    m = float(d.rate.max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="perfect", line=dict(dash="dot", color="#999")))
    figs.append(fig_html(layout(fig, "Load sent vs load served", "load sent (req/s)", "load served (req/s)")))
    return section("latency", "Latency at each load",
                   "We send the same number of requests per second to every stack (200 rows per request, 3 runs each). "
                   "We shuffle the stack order each run. A stack stops after 2 steps that are too slow.", "".join(figs))


def spectrum_section(df, order) -> str:
    d = s1(df, order)
    rates = [r for r in (50, 100, 200) if d[d.rate == r].stk.nunique() >= 2]
    if not rates:
        return section("spectrum", "Slow requests", "", nodata("S1 grid"))
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
        body.append(fig_html(layout(fig, f"Latency by percentile at {r} req/s", "percentile", "latency (ms)", logy=True)))
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
        body.append(fig_html(layout(fig, f"Share of requests faster than a given time, at {r} req/s", "latency (ms)", "share of requests", logx=True)))
    return section("spectrum", "Slow requests",
                   "Averages hide the slow requests. These charts show them. p99.9 means 1 request in 1,000 is slower than this.",
                   "".join(body))


def capacity_section(df, order, cap) -> str:
    if cap.empty:
        return section("capacity", "Capacity", "", nodata("S1"))
    fig = go.Figure()
    for slo in SLOS:
        c = cap[cap.slo == slo].set_index("stk").reindex([s for s in order if s in set(cap.stk)])
        fig.add_trace(go.Bar(x=[label(s) for s in c.index], y=c.capacity, name=f"p99 under {slo} ms",
                             error_y=dict(type="data", symmetric=False, array=(c.hi - c.capacity).clip(lower=0),
                                          arrayminus=(c.capacity - c.lo).clip(lower=0))))
    body = [fig_html(layout(fig, "Capacity: most requests per second while p99 stays under the limit", "", "req/s", h=400))]
    d = df[(df.scenario == "S1") & (df.kind == "closed")]
    if not d.empty:
        g = d.groupby("stk").rps.median().reindex([s for s in order if s in set(d.stk)])
        fig = go.Figure(go.Bar(x=[label(s) for s in g.index], y=g.values, marker_color=[color(s) for s in g.index]))
        body.append(fig_html(layout(fig, "Top speed test: each user sends the next request as soon as the last reply is back", "", "req/s", h=340)))
    rows = []
    for st in order:
        pts = an.s1_points(df, st)
        if pts:
            rows.append({"stack": label(st), "knee": fmt(an.knee(pts), 0, " req/s") if an.knee(pts) else "–"})
    body.append("<h3>Where latency starts to jump</h3><p class='intro'>The first load where p99 is more than 2× the p99 at the lowest load.</p>"
                + table(rows, [("stack", "Stack"), ("knee", "Load where latency jumps")]))
    return section("capacity", "Capacity",
                   "Capacity = the highest load where p99 stays under the limit, nothing fails, and k6 sends every request on time. "
                   "The thin lines on the bars show the likely range (95 % sure).", "".join(body))


def fleet_section(df, order, sc) -> str:
    d = an.search_capacity(df, "S1_scale", 100)
    if d.empty:
        return section("fleet", "More replicas", "", nodata("S1_scale"))
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
    body = [fig_html(layout(fig, "Capacity with more replicas (the same 6 CPUs are shared; dotted line = perfect scaling)", "replicas", "req/s"))]
    body.append(table(eff, [("stack", "Stack"), ("replicas", "Replicas"), ("cap", "Capacity (req/s)"),
                            ("eff", "How close to perfect"), ("per_core", "req/s per CPU core")]))
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
        body.append("<h3>CPUs needed for a target load</h3><p class='intro'>Simple math: target load ÷ capacity per CPU (p99 under 50 ms).</p>"
                    + table(rows, [("stack", "Stack"), ("1000", "1k req/s"), ("10000", "10k req/s"), ("100000", "100k req/s")]))
    return section("fleet", "More replicas",
                   "All replicas share the same 6 CPUs. So this shows the cost of each extra replica. "
                   "It does not show real scale-out over many computers.", "".join(body))


def breakdown_section(df, order, ref=100) -> str:
    d = s1(df, order)
    d = d[d.rate == ref]
    if d.empty:
        return section("breakdown", "Where the time goes", "", nodata(f"S1 at {ref} req/s"))
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
    for nm, vals, col in (("waiting in line", queue, "#d9822b"), ("model work", comp, "#2f6fb3"),
                          ("network and ingress", over, "#9aa5b1")):
        fig.add_trace(go.Bar(x=names, y=vals, name=nm, marker_color=col))
    fig.update_layout(barmode="stack")
    return section("breakdown", "Where the time goes",
                   f"Average latency at {ref} req/s, split in three parts. The numbers come from the servers' own counters "
                   "(Triton, FastAPI, Ray). What is left over is the network and the ingress (proxy). "
                   "For Ray → Triton, 'model work' is Triton's time. Ray's extra hop is in the left-over part.",
                   fig_html(layout(fig, "Average latency, split in parts (ms)", "", "ms", h=380)))


def resources_section(df, order, cold, ref=100) -> str:
    d = s1(df, order)
    if d.empty:
        return section("resources", "CPU and memory", "", nodata("S1"))
    body = []
    g = d[d.rate == ref]
    comps = sorted({c for x in g.comp for c in x})
    if comps:
        fig = go.Figure()
        for c in comps:
            ys = [float(np.median([x.get(c, 0) for x in g[g.stk == st].comp] or [0])) for st in order if st in set(g.stk)]
            fig.add_trace(go.Bar(x=[label(s) for s in order if s in set(g.stk)], y=ys, name=c))
        fig.update_layout(barmode="stack")
        body.append(fig_html(layout(fig, f"CPU used by each part at {ref} req/s", "", "CPU cores", h=400)))
    fig = go.Figure()
    for st in order:
        g = d[(d.stk == st) & d.cpu.notna() & (d.rps > 0)]
        x = g.assign(v=g.cpu / g.rps * 1000).groupby("rate").v.median().reset_index()
        fig.add_trace(go.Scatter(x=x.rate, y=x.v, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    body.append(fig_html(layout(fig, "CPU time per request", "load (requests per second)", "CPU ms per request", logy=True)))
    cs = cold[cold.scenario == "S1"].groupby("stk").agg(ready=("ready_s", "median"), idle=("idle_rss_mb", "median"),
                                                          image=("image_mb", "median")).reindex([s for s in order if s in set(cold[cold.scenario == "S1"].stk)])
    peak = d.groupby("stk").rss.max()
    rows = [{"stack": label(s), "ready": fmt(r.ready, 1, " s"), "idle": fmt(r.idle, 0, " MB"),
             "peak": fmt(peak.get(s), 0, " MB"), "image": fmt(r.image, 0, " MB")} for s, r in cs.iterrows()]
    body.append(table(rows, [("stack", "Stack"), ("ready", "Start-up time"), ("idle", "Memory when idle"), ("peak", "Peak memory under load"), ("image", "Image size")]))
    return section("resources", "CPU and memory",
                   "How much CPU each part of a stack uses: the ingress (proxy or HAProxy), Ray replicas, Ray helpers, Triton, uvicorn.",
                   "".join(body))


def cost_section(sc) -> str:
    if sc.empty:
        return section("cost", "Cost", "", nodata("S1"))
    pr = an.prices()
    sc = sc.dropna(subset=["usd_per_m_alloc"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_alloc, name="all 6 CPUs reserved"))
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_used, name="CPUs actually used"))
    body = [fig_html(layout(fig, "Dollars per 1 million requests (at the capacity with p99 under 50 ms)", "", "dollars per 1M requests", h=380)),
            ]
    fig = go.Figure()
    for _, r in sc.iterrows():
        if r.p99_at_ref:
            fig.add_trace(go.Scatter(x=[r.usd_per_m_alloc], y=[r.p99_at_ref], mode="markers+text", text=[label(r.stk)],
                                     textposition="top center", marker=dict(size=14, color=color(r.stk)), showlegend=False))
    body.append(fig_html(layout(fig, "Cost and latency (bottom-left is best)", "dollars per 1M requests (reserved CPUs)", "p99 at 100 req/s (ms)")))
    note = (f"Prices used: {esc(PRICE_NAMES.get(pr['name'], pr['name']))}, ${pr['usd_per_vcpu_hour']}/vCPU-h + ${pr['usd_per_gb_hour']}/GB-h, as of {esc(pr['as_of'])}, "
            f"<b>{'verified' if pr['verified'] else 'NOT checked against the price page'}</b>. "
            "We turn CPU time into dollars with these prices. The dollar amounts are a guide. The order of the stacks is what counts.")
    return section("cost", "Cost", note, "".join(body))


# ------------------------------------------------------------------ Ray Serve feature scenarios
def ingress_section(df) -> str:
    d = df[(df.scenario == "I1") & (df.rows == 200)]
    if d.empty:
        return "<h3>Ingress</h3>" + nodata("I1")
    cap = an.search_capacity(df, "I1", 100).set_index("variant").capacity
    cap50 = an.search_capacity(df, "I1", 50).set_index("variant").capacity
    fig = go.Figure()
    for v in d.variant.unique():
        g = d[(d.variant == v) & (d.kind == "grid")].groupby("rate").p99.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
    out = [fig_html(layout(fig, "p99 by ingress", "load (requests per second)", "p99 (ms)", logy=True, h=360))]
    fig = go.Figure()
    for slo, c in ((50, cap50), (100, cap)):
        fig.add_trace(go.Bar(x=list(c.index), y=list(c.values), name=f"p99 under {slo} ms"))
    out.append(fig_html(layout(fig, "Capacity by ingress (short test)", "", "req/s", h=340)))
    return "<h3>Ingress: Python proxy vs HAProxy vs gRPC</h3>" + "".join(out)


def family_section(df, cold_v) -> str:
    out = []
    d = df[df.scenario == "F1"]
    if d.empty:
        out.append("<h3>Many models in one service</h3>" + nodata("F1"))
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
        fig = go.Figure(go.Bar(x=c100.variant, y=c100.capacity, name="p99 under 100 ms"))
        fig.add_trace(go.Bar(x=c50.variant, y=c50.capacity, name="p99 under 50 ms"))
        out.append("<h3>Many models in one service</h3>"
                   + fig_html(layout(fig, "Capacity with 16 models (a few models get most requests)", "", "req/s", h=360))
                   + table(rows, [("variant", "Config"), ("stack", "Stack"), ("c50", "Capacity (p99 under 50 ms)"), ("c100", "Capacity (p99 under 100 ms)"),
                                  ("p99", "p99 at 60 req/s"), ("rss", "Peak memory"), ("cpu", "CPU cores used"), ("miss", "How often a model must be loaded")]))
    if not cold_v.empty:
        fig = go.Figure()
        for (v, kind), g in cold_v.groupby(["variant", "kind"]):
            fig.add_trace(go.Box(y=g.first_ms, name=f"{v} ({kind})", boxpoints="all"))
        fig.add_trace(go.Scatter(x=[f"{v} ({k})" for (v, k), g in cold_v.groupby(["variant", "kind"])],
                                 y=[g.steady_ms.median() for _, g in cold_v.groupby(["variant", "kind"])],
                                 mode="markers", name="steady-state", marker=dict(symbol="line-ew", size=18, color="#333", line=dict(width=2))))
        out.append(fig_html(layout(fig, "First request to a model that is not loaded yet (ms)", "", "ms", logy=True, h=380)))
    return "".join(out)


def composition_section(df) -> str:
    out = []
    d = df[(df.scenario == "F3") & (df.kind == "fanout")]
    if d.empty:
        out.append("<h3>One request, many models</h3>" + nodata("F3"))
    else:
        d = d.assign(system=d.variant.str.replace(r"_k\d+$", "", regex=True))
        f1, f2 = go.Figure(), go.Figure()
        for s, g in d.groupby("system"):
            a = g.groupby("k").p99.median().reset_index()
            f1.add_trace(go.Scatter(x=a.k, y=a.p99, mode="lines+markers", name=s, line=dict(color=color(s))))
            b = g.assign(v=g.cpu / g.rps * 1000).groupby("k").v.median().reset_index()
            f2.add_trace(go.Scatter(x=b.k, y=b.v, mode="lines+markers", name=s, line=dict(color=color(s))))
        out.append("<h3>One request, many models</h3>"
                   + fig_html(layout(f1, "p99 when each request uses K models (40 req/s)", "models per request (K)", "p99 (ms)", logy=True, h=340))
                   + fig_html(layout(f2, "CPU time per request", "models per request (K)", "CPU ms per request", h=340)))
    p = df[(df.scenario == "F5") & (df.kind == "grid")]
    if p.empty:
        out.append("<h3>Steps in a row</h3>" + nodata("F5"))
    else:
        fig = go.Figure()
        for v, g in p.groupby("variant"):
            a = g.groupby("rate").p99.median().reset_index()
            fig.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
        cap = an.search_capacity(df, "F5", 100)
        out.append("<h3>Steps in a row: prepare data → model → fix the score</h3>"
                   + fig_html(layout(fig, "p99 as load goes up", "req/s", "p99 (ms)", logy=True, h=340))
                   + table([{"v": r.variant, "c": int(r.capacity)} for r in cap.itertuples()], [("v", "Stack"), ("c", "Capacity (p99 under 100 ms)")]))
    return "".join(out)


def _rel(series_t0, t_measure):  # seconds offset of k6 series start vs t_measure
    return (series_t0 - t_measure) if series_t0 and t_measure else 0.0


def autoscale_section(df, slo=50) -> str:
    d = df[(df.scenario == "A1") & (df.kind == "burst")]
    if d.empty:
        return "<h3>Auto-scaling when load jumps 10×</h3>" + nodata("A1")
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
                    fillcolor="#b5473a", opacity=0.07, line_width=0, annotation_text="load jumps 10×", annotation_position="top left")
    return ("<h3>Auto-scaling when load jumps 10×</h3>"
            + fig_html(layout(f_lat, "p99 each second (first run). The red part is the jump.", "seconds since the test started", "p99 (ms)", logy=True, h=360))
            + fig_html(layout(f_rep, "Replicas running", "seconds since the test started", "replicas", h=300))
            + fig_html(layout(f_err, "Errors each second", "seconds since the test started", "errors", h=300))
            + table(rows, [("p", "Setting"), ("rec", f"Seconds until p99 is under {slo} ms again"), ("err", "Errors"),
                           ("p99", "Typical p99 after the jump"), ("core", "CPU seconds used"), ("max", "Most replicas")]))


def backpressure_section(df) -> str:
    d = df[(df.scenario == "R1") & df.kind.isin(["search", "overload"])]
    if d.empty:
        return "<h3>Too much load</h3>" + nodata("R1")
    f1, f2 = go.Figure(), go.Figure()
    rows = []
    for v, g in d.groupby("variant"):
        a = g.groupby("rate").agg(ok=("ok", "median"), dur=("duration_s", "median"), p99=("p99", "median")).reset_index()
        f1.add_trace(go.Scatter(x=a.rate, y=a.ok / a.dur, mode="lines+markers", name=v, line=dict(color=color(v))))
        f2.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=v, line=dict(color=color(v))))
    mx = float(d.rate.max())
    f1.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="perfect", line=dict(dash="dot", color="#999")))
    for r in d[d.kind == "overload"].itertuples():
        codes = ", ".join(f"{k}: {v}" for k, v in r.error_codes.items()) or "none"
        rows.append({"v": r.variant, "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0),
                     "p99": fmt(r.p99, 1, " ms"), "codes": codes})
    return ("<h3>Too much load: what each stack does</h3>"
            + fig_html(layout(f1, "Successful requests per second vs load sent", "load sent (req/s)", "successful (req/s)", h=360))
            + fig_html(layout(f2, "p99 of successful requests vs load sent", "load sent (req/s)", "p99 (ms)", logy=True, h=340))
            + table(rows, [("v", "Config"), ("f", "Times capacity"), ("rate", "Load sent"), ("good", "Successful req/s"), ("p99", "p99 (successful)"), ("codes", "Failed replies")]))


def failure_section(df) -> str:
    d = df[(df.scenario == "R2") & (df.kind == "failure")]
    if d.empty:
        return "<h3>A process dies</h3>" + nodata("R2")
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
    return ("<h3>A process dies: we kill one server process during steady load</h3>"
            + fig_html(layout(fig, "Errors each second around the kill (first run)", "seconds before / after the kill", "errors", h=320))
            + fig_html(layout(fig2, "p99 each second around the kill", "seconds before / after the kill", "p99 (ms)", logy=True, h=320))
            + table(rows, [("p", "Stack"), ("err", "Errors (typical)"), ("win", "Seconds with errors"), ("rec", "Seconds until p99 is normal again")]))


def overload_section(df, order, cold) -> str:
    d = df[(df.scenario == "S1") & (df.kind == "overload")]
    rows = [{"s": label(r.stk), "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0), "p99": fmt(r.p99, 0, " ms"),
             "err": r.err, "dropped": r.dropped} for r in d.sort_values(["stk", "factor"]).itertuples()]
    return section("overload", "Too much load",
                   "We send 1.5× and 3× the load each stack can take with p99 under 100 ms (first run). 'Successful' means the reply was OK.",
                   table(rows, [("s", "Stack"), ("f", "Times capacity"), ("rate", "Load sent"), ("good", "Successful req/s"), ("p99", "p99"),
                                ("err", "Errors"), ("dropped", "Requests not sent on time")]))


# ------------------------------------------------------------------ summary, methodology, appendix
def summary_section(df, sc, cap, cold_v) -> str:
    cards = []

    def card(title, value, sub):
        cards.append(f'<div class="card"><div class="ct">{esc(title)}</div><div class="cv">{value}</div><div class="cs">{sub}</div></div>')
    if not sc.empty:
        b = sc.iloc[0]
        card("Takes the most load (p99 under 50 ms)", esc(label(b.stk)), f"{b.capacity} req/s (likely range {b.ci})")
        c = sc.dropna(subset=["usd_per_m_alloc"]).sort_values("usd_per_m_alloc")
        if len(c):
            card("Cheapest per 1M requests", esc(label(c.iloc[0].stk)), f"${c.iloc[0].usd_per_m_alloc:.3f} with reserved CPUs")
        e = sc.dropna(subset=["core_ms_per_req"]).sort_values("core_ms_per_req")
        if len(e):
            card("Uses the least CPU", esc(label(e.iloc[0].stk)), f"{e.iloc[0].core_ms_per_req:.2f} CPU ms per request")
        f = sc.dropna(subset=["cold_s"]).sort_values("cold_s")
        if len(f):
            card("Starts fastest", esc(label(f.iloc[0].stk)), f"{f.iloc[0].cold_s:.0f} s")
    i1 = an.search_capacity(df, "I1", 50)
    if len(i1):
        top = i1.sort_values("capacity", ascending=False).iloc[0]
        card("Best Ray ingress", esc(top.variant), f"{top.capacity} req/s at 50 ms")
    f1 = an.search_capacity(df, "F1", 100)
    if len(f1):
        top = f1.sort_values("capacity", ascending=False).iloc[0]
        card("Best way to serve many models", esc(top.variant), f"{top.capacity} req/s at 100 ms")
    table_html = ""
    if not sc.empty:
        rows = [{"stack": label(r.stk), "cap": r.capacity, "ci": r.ci, "p99": fmt(r.p99_at_ref, 1, " ms"), "cores": fmt(r.cores_at_cap, 2),
                 "cms": fmt(r.core_ms_per_req, 2), "rpv": fmt(r.rps_per_vcpu, 1), "rss": fmt(r.rss_mb, 0, " MB"),
                 "cold": fmt(r.cold_s, 0, " s"), "ca": fmt(r.usd_per_m_alloc, 3), "cu": fmt(r.usd_per_m_used, 3)} for r in sc.itertuples()]
        table_html = table(rows, [("stack", "Stack"), ("cap", "Capacity (p99 under 50 ms)"), ("ci", "Likely range"), ("p99", "p99 at 100 req/s"),
                                  ("cores", "CPU cores used at capacity"), ("cms", "CPU ms per request"), ("rpv", "req/s per CPU"), ("rss", "Peak memory"),
                                  ("cold", "Start-up time"), ("ca", "$ per 1M (reserved CPUs)"), ("cu", "$ per 1M (CPUs used)")], cls="score")
    return section("summary", "Summary", "Model: DeepFM small. 200 rows per request. One computer: 6 CPUs for the servers and 2 for the load tool. "
                   "The numbers are good to compare stacks. They do not size a cloud fleet. "
                   "<br><b>Words used here.</b> <b>req/s</b>: requests per second. <b>Latency</b>: time to get the answer, in milliseconds (ms). "
                   "<b>p50</b>: half of the requests are faster than this. <b>p99</b>: 99 of 100 requests are faster than this. It shows the slow ones. "
                   "<b>Capacity</b>: the most req/s a stack takes while p99 stays under a limit and nothing fails. "
                   "<b>Replica</b>: one copy of the model server. "
                   "<b>Stack</b>: one way to serve a model (for example FastAPI, Triton or Ray Serve). "
                   "The tests are explained in the next section.",
                   f'<div class="cards">{"".join(cards)}</div>{table_html}')


def tests_section(df) -> str:
    ran = set(df.scenario)
    cards = []
    for key, (name, question, what, measure) in TESTS.items():
        state = "" if key in ran else ' <span class="muted">(no data in this report)</span>'
        cards.append(f'<div class="test"><h3>{esc(name)}{state}</h3><p><b>Question.</b> {esc(question)}</p>'
                     f'<p><b>What we do.</b> {esc(what)}</p><p><b>What we measure.</b> {esc(measure)}</p></div>')
    return section("tests", "The tests", "Eleven tests. Each one answers one question.", "".join(cards))


def methodology_section(df, meta, rows) -> str:
    v = meta.get("versions", {})
    m = an.matrix()
    flat = "".join(f"<tr><td>{esc(k)}</td><td>{esc(val)}</td></tr>" for k, val in sorted(
        (v if isinstance(v, dict) else {}).items()) if k in VERSION_KEYS and not isinstance(val, (dict, list)))
    inval = int((~df.valid).sum()) if len(df) else 0
    return section("method", "How we measured",
                   "How we got the numbers. Read this before you trust a comparison.",
                   f"""<ul>
<li><b>Load:</b> k6 sends requests on a fixed schedule. The load tool and the servers use separate CPUs (2 and 6), and the servers get the same memory limit.</li>
<li><b>Capacity:</b> the highest load where the middle p99 of 3 runs is under the limit ({', '.join(map(str, m['slos_ms']))} ms), nothing fails, and at most 0.1 % of requests are sent late. The likely range comes from re-sampling the 3 runs.</li>
<li><b>Order:</b> we shuffle the stack order each run, so a hot computer does not favour one stack. Warm-up time is not counted.</li>
<li><b>Bad steps:</b> a step is marked bad when k6 used over 80 % of its CPUs, the CPU slowed down from heat, or a container restarted. Bad steps do not count for capacity. They are listed at the bottom. {inval} of {len(df)} steps are bad.</li>
<li><b>Server numbers:</b> we read the Ray, Triton and FastAPI counters every second. CPU use per process comes from the system.</li>
<li><b>Model:</b> random weights, ONNX Runtime on CPU, same settings in every stack.</li>
<li><b>One computer:</b> the servers and the load tool share one CPU chip. Real cloud numbers will differ.</li></ul>
<h3>Versions</h3><div class="tw"><table><tbody>{flat}</tbody></table></div>""")


def appendix_section(df, rows) -> str:
    cov = df.groupby(["scenario", "stk"]).agg(steps=("valid", "size"), invalid=("valid", lambda s: int((~s).sum()))).reset_index()
    cov_rows = [{"s": tname(r.scenario), "st": label(r.stk), "n": r.steps, "i": r.invalid} for r in cov.itertuples()]
    bad = [{"s": tname(r.scenario), "st": label(r.stk), "v": r.variant, "rate": r.rate, "why": esc(r.invalid)} for r in df[~df.valid].itertuples()]
    allrows = [{"s": tname(r.scenario), "v": r.variant, "st": r.stk, "k": r.kind, "rate": r.rate or r.vus, "rep": r.rep,
                "p50": fmt(r.p50, 2), "p99": fmt(r.p99, 2), "ok": r.ok, "err": r.err, "dropped": r.dropped, "cpu": fmt(r.cpu, 2),
                "rss": fmt(r.rss, 0), "valid": "yes" if r.valid else "NO"} for r in df.itertuples()]
    return section("appendix", "Appendix", "What ran, bad steps, and every step. Click a column name to sort.",
                   "<h3>What ran</h3>" + table(cov_rows, [("s", "Test"), ("st", "Stack"), ("n", "Steps"), ("i", "Bad")])
                   + "<h3>Bad steps</h3>" + table(bad, [("s", "Test"), ("st", "Stack"), ("v", "Variant"), ("rate", "Rate"), ("why", "Reason")])
                   + "<h3>All steps</h3>" + table(allrows, [("s", "Test"), ("v", "Variant"), ("st", "Stack"), ("k", "Kind"), ("rate", "Load or users"),
                                                          ("rep", "Rep"), ("p50", "p50 ms"), ("p99", "p99 ms"), ("ok", "OK"), ("err", "Err"),
                                                          ("dropped", "Not sent"), ("cpu", "CPU cores"), ("rss", "Memory MB"), ("valid", "OK step")], cls="sortable")
                   )


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
.sortable th{cursor:pointer}.test{border-left:3px solid var(--acc);padding:2px 14px;margin:14px 0}.test h3{margin:4px 0}.test p{margin:4px 0}.fig{margin:8px 0}pre{background:var(--card);padding:10px;border-radius:6px;overflow-x:auto}
@media (max-width:640px){main,header,nav{padding-left:16px;padding-right:16px}}
"""
SORT_JS = """document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{
const b=t.tBodies[0],r=[...b.rows],d=th.dataset.d=th.dataset.d==='1'?'-1':'1';
r.sort((x,y)=>{const a=x.cells[i].textContent.replace(/[, ]/g,''),c=y.cells[i].textContent.replace(/[, ]/g,'');
const n=parseFloat(a),m=parseFloat(c);return (isNaN(n)||isNaN(m)?a.localeCompare(c):n-m)*d});r.forEach(x=>b.appendChild(x))}));"""


def to_markdown(sc: pd.DataFrame, df, env: str) -> str:
    lines = ["# Serving benchmark: results", "",
             f"Generated {datetime.now():%Y-%m-%d}.", ""]
    if not sc.empty:
        pr = an.prices()
        lines += ["## Scorecard (p99 under 50 ms, DeepFM small, 200 rows per request)", "",
                  "| Stack | Capacity (req/s) | Likely range | p99 at 100 req/s | CPU ms per request | req/s per CPU | Peak memory | Start-up | $ per 1M (reserved CPUs) | $ per 1M (CPUs used) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sc.itertuples():
            lines.append(f"| {label(r.stk)} | {r.capacity} | {r.ci} | {fmt(r.p99_at_ref, 1, ' ms')} | {fmt(r.core_ms_per_req, 2)} | "
                         f"{fmt(r.rps_per_vcpu, 1)} | {fmt(r.rss_mb, 0, ' MB')} | {fmt(r.cold_s, 0, ' s')} | {fmt(r.usd_per_m_alloc, 3)} | {fmt(r.usd_per_m_used, 3)} |")
        lines += ["", f"Prices: {pr['name']} ${pr['usd_per_vcpu_hour']}/vCPU-h, ${pr['usd_per_gb_hour']}/GB-h "
                      f"({'verified' if pr['verified'] else 'NOT verified'}, as of {pr['as_of']}).", ""]
    for sid, title in (("I1", "Ray ingress (capacity, p99 under 50 ms)"), ("F1", "Many models in one service (capacity, p99 under 100 ms)"),
                       ("R1", "Too much load (capacity, p99 under 100 ms)")):
        c = an.search_capacity(df, sid, 50 if sid == "I1" else 100)
        if len(c):
            lines += [f"## {title}", "", "| Setting | Capacity (req/s) |", "|---|---|"] + [f"| {r.variant} | {r.capacity} |" for r in c.itertuples()] + [""]
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
                           "More than a plain web server: ingress options, many models in one service, steps in a row, auto-scaling, too much load, and a process that dies.",
                           ingress_section(df) + family_section(df, cold_v) + composition_section(df) + autoscale_section(df)
                           + backpressure_section(df) + failure_section(df))
    parts = [summary_section(df, sc, cap, cold_v), tests_section(df), methodology_section(df, meta, rows), latency_section(df, order),
             spectrum_section(df, order), capacity_section(df, order, cap), fleet_section(df, order, sc),
             breakdown_section(df, order), resources_section(df, order, cold), cost_section(sc), ray_features,
             overload_section(df, order, cold), appendix_section(df, rows)]
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (("summary", "Summary"), ("tests", "The tests"), ("method", "How we measured"), ("latency", "Latency"), ("spectrum", "Slow requests"),
                  ("capacity", "Capacity"), ("fleet", "Replicas"), ("breakdown", "Time"), ("resources", "Resources"), ("cost", "Cost"),
                  ("ray", "Ray Serve features"), ("overload", "Too much load"), ("appendix", "Appendix")))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>Serving benchmark report</title><style>{CSS}</style><script>{get_plotlyjs()}</script></head><body>"
           f"<header><h1>Serving benchmark: FastAPI vs Ray Serve vs Triton</h1><div class='muted'>{datetime.now():%Y-%m-%d} · {len(df)} measured steps</div></header><nav>{nav}</nav><main>{''.join(parts)}</main>"
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
