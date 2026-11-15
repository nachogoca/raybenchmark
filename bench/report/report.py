"""Single-file interactive HTML report (Plotly inlined, works offline) + REPORT.md headline tables.

Usage: uv run python -m bench.report.report [--env laptop]
Writes bench/report/report.html and REPORT.md from bench/results/<env>.jsonl.
"""
from __future__ import annotations

import argparse
import html
import math
import re
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
    "S1": ("Latency and capacity",
           "How do latency and capacity compare across stacks as offered load grows?",
           "Open-loop k6 at a common set of offered rates (25 to 650 req/s), 200 rows per request, 2 repetitions in shuffled stack order. "
           "Then overload at 1.5× and 3× capacity, and a closed-loop max-throughput run.",
           "p50 to p99.99, capacity under each SLO, CPU per request, memory, cold start, cost per 1M requests.",
           "Capacity is the highest offered load at which p99 stays within the SLO. Under open-loop load, latency rises sharply near capacity because requests queue. The closed-loop run shows peak throughput when each client waits for its reply."),
    "S1_scale": ("Replica scaling",
                 "How does capacity scale with replica count?",
                 "Ray, FastAPI and Triton at 1, 2 and 4 replicas on a fixed 6-CPU budget.",
                 "Capacity and scaling efficiency per replica count. This isolates per-replica overhead; it is not a multi-node test.",
           "A replica is one copy of the model server. On a fixed CPU budget, more replicas split the cores, so any gain comes from removing per-process limits (event loop, GIL), not from extra hardware."),
    "S1_batch": ("Batching",
                 "When does request batching pay off?",
                 "Ray @serve.batch (2 and 5 ms wait) and Triton dynamic batching against no batching, at 200 rows per request.",
                 "Capacity under each SLO.",
           "Request batching merges concurrent requests into one model call. It raises throughput per core because the model runs once on a larger input. It adds a wait of up to the batch window, so it helps at high load and can hurt p99 at low load."),
    "I1": ("Ingress",
           "What does the ingress layer cost?",
           "Ray behind the Python proxy, HAProxy and gRPC ingress; Triton over HTTP and gRPC.",
           "p99 at fixed offered loads, and capacity. The k6 gRPC client is CPU-heavy: gRPC steps above about 200 req/s can be limited by the load generator and are then flagged invalid.",
           "The ingress is the first hop: it accepts the request and routes it to a replica. Ray's default is a Python proxy actor. HAProxy is a native proxy that routes straight to replicas. gRPC replaces JSON over HTTP with a binary protocol."),
    "F1": ("Model family (multiplexing)",
           "How do multi-model layouts compare for a 16-model family under Zipf-skewed traffic?",
           "Ray multiplexing (LRU of 4 and of 16 models), one Ray deployment per model, Triton with 16 models, FastAPI with 16 sessions. "
           "Ray multiplexing is also scaled from 4 to 64 models.",
           "Capacity, memory, model load rate.",
           "A model family is many models with one architecture and different weights, for example one per market. Ray multiplexing (@serve.multiplexed) serves all of them from one deployment: each replica keeps an LRU cache of loaded models and loads on demand. The alternatives are one deployment per model, or a multi-model server such as Triton."),
    "F2": ("Cold model load",
           "What is the first-request latency of a model that is not loaded?",
           "First request to 5 unused models per stack (multiplexed load in Ray, lazy load in FastAPI, explicit load in Triton), "
           "plus a model evicted from Ray's cache.",
           "First-request latency against steady state.",
           "A model that is not in memory must be loaded before it can answer. First-request latency is what a caller pays for a rarely used model, or one the cache evicted."),
    "F3": ("Ensemble fan-out",
           "What does scoring one request with K models and averaging cost as K grows?",
           "K = 1, 4 and 8 at a fixed 40 req/s: Ray handle gather, FastAPI asyncio.gather, Triton ensemble.",
           "p99 and CPU per request against K.",
           "Ensemble fan-out scores one request with K models and averages the scores. The K calls run in parallel, so latency should grow slowly with K while CPU grows linearly. It shows how cheaply each stack calls several models."),
    "F5": ("Multi-stage pipeline",
           "What does a multi-stage pipeline cost?",
           "Feature step, routed model, calibration step: Ray composed deployments, FastAPI in-process, Triton ensemble with a Python-backend step.",
           "p99 against offered load, and capacity.",
           "A pipeline chains stages: feature preparation, the model, calibration. In Ray each stage can be its own deployment called through a handle. In Triton an ensemble wires the stages inside the server. In FastAPI it is one function call."),
    "A1": ("Autoscaling under a 10× burst",
           "How quickly does each setup absorb a 10× step in load?",
           "30 → 300 req/s for 90 s, then back. Static fleets of 1 and 4 replicas against Ray autoscaling "
           "(target_ongoing_requests 2 and 8, upscale delay 0 and 10 s). Fresh service per run, 2 runs.",
           "Time to recover p99, errors, replica count over time, core-seconds.",
           "Autoscaling changes the replica count with load. Ray adds replicas when ongoing requests per replica exceed target_ongoing_requests, after an optional upscale delay. Static fleets are the baseline: too small saturates, too large wastes CPU."),
    "R1": ("Overload and backpressure",
           "How does each stack behave past capacity?",
           "1.5× and 3× capacity with unbounded and bounded queues (Ray max_queued_requests, uvicorn limit-concurrency, Triton max_queue_size).",
           "Goodput, rejected requests, p99 of successful requests.",
           "Backpressure rejects work the server cannot take instead of queueing it without limit. A bounded queue keeps the latency of accepted requests low, at the price of errors for the rest. Goodput is the rate of successful requests."),
    "R2": ("Failure recovery",
           "How does each stack recover when a serving process is killed?",
           "Steady 100 req/s; kill one Ray replica, one uvicorn worker or the Triton container; 3 runs each.",
           "Errors, and time until p99 returns to baseline.",
           "When a serving process dies, requests routed to it fail until a supervisor or orchestrator replaces it. The gap, and how fast p99 returns to baseline, is the recovery cost."),
}
PRICE_NAMES = {"gke_n2": "Google Cloud n2 machines", "eks_m6i": "AWS m6i machines"}
VERSION_KEYS = ("python", "ray", "onnxruntime", "tritonclient", "fastapi", "uvicorn", "numpy", "k6",
                "image nvcr.io/nvidia/tritonserver:26.10-py3")


VARIANT_LABELS = {
    "mux4": "Ray multiplexing, cache of 4 models", "mux16": "Ray multiplexing, cache of 16 models",
    "deployments": "Ray, one deployment per model", "triton16": "Triton, 16 models", "fastapi16": "FastAPI, 16 models",
    "mux16_v4": "Ray multiplexing, 4-model family", "mux16_v64": "Ray multiplexing, 64-model family",
    "ray_python_proxy": "Ray, Python proxy", "ray_haproxy": "Ray, HAProxy", "ray_grpc": "Ray, gRPC",
    "triton_http": "Triton, HTTP", "triton_grpc": "Triton, gRPC",
    "ray_mux": "Ray multiplexing", "fastapi_lazy": "FastAPI, lazy load", "triton_explicit": "Triton, explicit load",
    "ray_fanout": "Ray, handle gather", "fastapi_fanout": "FastAPI, asyncio.gather", "triton_ensemble": "Triton ensemble",
    "ray_pipeline": "Ray, composed deployments", "fastapi_pipeline": "FastAPI, in-process", "triton_pipeline": "Triton ensemble",
    "static1": "Static, 1 replica", "static4": "Static, 4 replicas",
    "auto_t2_up0": "Autoscaling, target 2, no upscale delay", "auto_t8_up0": "Autoscaling, target 8, no upscale delay",
    "auto_t2_up10": "Autoscaling, target 2, 10 s upscale delay",
    "ray_unbounded": "Ray, unbounded queue", "ray_queue50": "Ray, max 50 queued",
    "fastapi_unbounded": "FastAPI, unbounded", "fastapi_limit50": "FastAPI, limit 50 concurrent",
    "triton_unbounded": "Triton, unbounded queue", "triton_queue50": "Triton, max 50 queued",
    "ray": "Ray replica killed", "fastapi": "FastAPI worker killed", "triton": "Triton container killed",
    "nobatch": "no batching", "wait2": "batching, 2 ms wait", "wait5": "batching, 5 ms wait", "dynamic": "dynamic batching",
}


def vl(v) -> str:
    """Readable name of a variant key; unknown keys pass through."""
    v = str(v)
    m = re.match(r"^(.*)_r(\d+)$", v)
    if m and m.group(1) in VARIANT_LABELS:
        return f"{VARIANT_LABELS[m.group(1)]} ({m.group(2)} rows)"
    return VARIANT_LABELS.get(v, v)


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


def layout(fig: go.Figure, title: str = "", xt: str = "", yt: str = "", h: int = 420, logy=False, logx=False,
           ymax: float | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.01, font=dict(size=14)), height=h,
                      margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.2),
                      hovermode="x unified", font=dict(family="system-ui, sans-serif", size=12))
    fig.update_xaxes(title=xt, type="log" if logx else None)
    fig.update_yaxes(title=yt, type="log" if logy else None)
    if ymax:                      # cap the axis so overload spikes do not flatten the useful range
        fig.update_yaxes(range=[math.log10(0.5), math.log10(ymax)] if logy else [0, ymax])
        fig.add_annotation(text=f"values above {ymax:g} ms are cut off", xref="paper", yref="paper", x=1, y=1.02,
                           showarrow=False, xanchor="right", font=dict(size=11, color="#888"))
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
    return f'<p class="muted">No data: the test "{esc(what)}" did not run or produced no valid steps.</p>'


def test_block(key: str) -> str:
    name, question, method, metrics, concept = TESTS[key]
    return (f'<div class="test"><p><b>Concept.</b> {esc(concept)}</p><p><b>Question.</b> {esc(question)}</p>'
            f'<p><b>Method.</b> {esc(method)}</p><p><b>Metrics.</b> {esc(metrics)}</p></div>')


def group(sid: str, title: str, keys: list[str], inner: str) -> str:
    blocks = "".join((f"<h3>{esc(tname(k))}</h3>" if len(keys) > 1 else "") + test_block(k) for k in keys)
    return f'<section id="{sid}"><h2>{esc(title)}</h2>{blocks}{inner}</section>'


def sub(sid: str, title: str, intro: str, body: str) -> str:
    return f'<div id="{sid}"><h3>{esc(title)}</h3>' + (f'<p class="intro">{intro}</p>' if intro else "") + f"{body}</div>"


def section(sid: str, title: str, intro: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{esc(title)}</h2><p class="intro">{intro}</p>{body}</section>'


# ------------------------------------------------------------------ S1: latency / capacity
def s1(df, stack_order):
    return df[(df.scenario == "S1") & (df.rows == 200) & df.kind.isin(["grid", "refine"]) & df.valid]


def latency_section(df, order) -> str:
    d = s1(df, order)
    if d.empty:
        return sub("latency", "Latency vs offered load", "", nodata("S1"))
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
        figs.append(fig_html(layout(fig, f"{name} latency vs offered load (median of repetitions; band = min–max)",
                                    "offered load (req/s)", f"{name} (ms)", ymax={"p50": 100, "p99": 250, "p999": 500}[col])))
    fig = go.Figure()
    for st in order:
        g = d[d.stk == st].groupby("rate").rps.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.rps, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    m = float(d.rate.max())
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", name="perfect", line=dict(dash="dot", color="#999")))
    figs.append(fig_html(layout(fig, "Achieved vs offered throughput", "offered (req/s)", "achieved (req/s)")))
    return sub("latency", "Latency vs offered load",
                   "Each stack stops two grid points after it misses the loosest SLO. Charts cap the y-axis so overload spikes do not hide the useful range.", "".join(figs))


def spectrum_section(df, order) -> str:
    d = s1(df, order)
    rates = [r for r in (50, 100, 200) if d[d.rate == r].stk.nunique() >= 2]
    if not rates:
        return sub("spectrum", "Tail latency", "", nodata("S1 grid"))
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
        body.append(fig_html(layout(fig, f"Latency CDF at {r} req/s (repetitions pooled)", "latency (ms)", "fraction of requests", logx=True)))
    return sub("spectrum", "Tail latency",
                   "Tail behaviour at fixed offered loads: percentile curve (p50 to p99.99, log-of-nines axis) and the pooled latency CDF.",
                   "".join(body))


def capacity_section(df, order, cap) -> str:
    if cap.empty:
        return sub("capacity", "Capacity", "", nodata("S1"))
    fig = go.Figure()
    for slo in SLOS:
        c = cap[cap.slo == slo].set_index("stk").reindex([s for s in order if s in set(cap.stk)])
        fig.add_trace(go.Bar(x=[label(s) for s in c.index], y=c.capacity, name=f"p99 ≤ {slo} ms",
                             error_y=dict(type="data", symmetric=False, array=(c.hi - c.capacity).clip(lower=0),
                                          arrayminus=(c.capacity - c.lo).clip(lower=0))))
    body = [fig_html(layout(fig, "Capacity: max throughput with p99 ≤ each SLO (error bars: 95 % bootstrap CI)", "", "req/s", h=400))]
    d = df[(df.scenario == "S1") & (df.kind == "closed")]
    if not d.empty:
        g = d.groupby("stk").rps.median().reindex([s for s in order if s in set(d.stk)])
        fig = go.Figure(go.Bar(x=[label(s) for s in g.index], y=g.values, marker_color=[color(s) for s in g.index]))
        body.append(fig_html(layout(fig, "Closed-loop max throughput (4 virtual users per replica)", "", "req/s", h=340)))
    rows = []
    for st in order:
        pts = an.s1_points(df, st)
        if pts:
            rows.append({"stack": label(st), "knee": fmt(an.knee(pts), 0, " req/s") if an.knee(pts) else "–"})
    body.append("<h3>Latency knee</h3><p class='intro'>First offered load where the median p99 exceeds 2× its lowest-load value.</p>"
                + table(rows, [("stack", "Stack"), ("knee", "Knee")]))
    return sub("capacity", "Capacity",
                   "Capacity = highest offered load with p99 ≤ SLO, zero errors and ≤ 0.1 % dropped arrivals (requests k6 could not issue on schedule). "
                   "Error bars: 95 % bootstrap CI over repetitions.", "".join(body))


def fleet_section(df, order, sc) -> str:
    d = an.search_capacity(df, "S1_scale", 100)
    if d.empty:
        return sub("fleet", "Replica scaling", "", nodata("S1_scale"))
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
    body = [fig_html(layout(fig, "Capacity vs replicas (fixed 6-CPU budget; dotted = linear scaling)", "replicas", "req/s"))]
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
        body.append("<h3>Fleet size for a target load</h3><p class='intro'>Linear extrapolation: target ÷ capacity per vCPU at p99 ≤ 50 ms.</p>"
                    + table(rows, [("stack", "Stack"), ("1000", "1k req/s"), ("10000", "10k req/s"), ("100000", "100k req/s")]))
    return sub("fleet", "Replica scaling",
                   "Replicas share one 6-CPU budget, so this isolates per-replica overhead. It is not a multi-node scale-out test.", "".join(body))


def breakdown_section(df, order, ref=100) -> str:
    d = s1(df, order)
    d = d[d.rate == ref]
    if d.empty:
        return sub("breakdown", "Latency breakdown", "", nodata(f"S1 at {ref} req/s"))
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
    for nm, vals, col in (("queueing", queue, "#d9822b"), ("model compute", comp, "#2f6fb3"),
                          ("ingress + network", over, "#9aa5b1")):
        fig.add_trace(go.Bar(x=names, y=vals, name=nm, marker_color=col))
    fig.update_layout(barmode="stack")
    return sub("breakdown", "Latency breakdown",
                   f"Mean latency at {ref} req/s, split with server-side counters (Triton queue and compute, FastAPI handler time, Ray replica "
                   "processing time). The remainder is ingress, network and client. For Ray → Triton, model compute is Triton's time and Ray's extra hop is in the remainder.",
                   fig_html(layout(fig, "Mean latency by component (ms)", "", "ms", h=380)))


def resources_section(df, order, cold, ref=100) -> str:
    d = s1(df, order)
    if d.empty:
        return sub("resources", "Resource usage", "", nodata("S1"))
    body = []
    g = d[d.rate == ref]
    comps = sorted({c for x in g.comp for c in x})
    if comps:
        fig = go.Figure()
        for c in comps:
            ys = [float(np.median([x.get(c, 0) for x in g[g.stk == st].comp] or [0])) for st in order if st in set(g.stk)]
            fig.add_trace(go.Bar(x=[label(s) for s in order if s in set(g.stk)], y=ys, name=c))
        fig.update_layout(barmode="stack")
        body.append(fig_html(layout(fig, f"CPU by component at {ref} req/s", "", "cores", h=400)))
    fig = go.Figure()
    for st in order:
        g = d[(d.stk == st) & d.cpu.notna() & (d.rps > 0)]
        x = g.assign(v=g.cpu / g.rps * 1000).groupby("rate").v.median().reset_index()
        fig.add_trace(go.Scatter(x=x.rate, y=x.v, mode="lines+markers", name=label(st), line=dict(color=color(st))))
    body.append(fig_html(layout(fig, "CPU cost per request", "offered load (req/s)", "CPU ms / request", logy=True)))
    cs = cold[cold.scenario == "S1"].groupby("stk").agg(ready=("ready_s", "median"), idle=("idle_rss_mb", "median"),
                                                          image=("image_mb", "median")).reindex([s for s in order if s in set(cold[cold.scenario == "S1"].stk)])
    peak = d.groupby("stk").rss.max()
    rows = [{"stack": label(s), "ready": fmt(r.ready, 1, " s"), "idle": fmt(r.idle, 0, " MB"),
             "peak": fmt(peak.get(s), 0, " MB"), "image": fmt(r.image, 0, " MB")} for s, r in cs.iterrows()]
    body.append(table(rows, [("stack", "Stack"), ("ready", "Cold start"), ("idle", "Idle RSS"), ("peak", "Peak RSS under load"), ("image", "Image")]))
    return sub("resources", "Resource usage",
                   "Per-process CPU by role: ingress (proxy / HAProxy), Ray replicas, Ray system processes, Triton, uvicorn.",
                   "".join(body))


def cost_section(sc) -> str:
    if sc.empty:
        return sub("cost", "Cost", "", nodata("S1"))
    pr = an.prices()
    sc = sc.dropna(subset=["usd_per_m_alloc"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_alloc, name="allocated (6 vCPU)"))
    fig.add_trace(go.Bar(x=[label(s) for s in sc.stk], y=sc.usd_per_m_used, name="consumed (measured CPU)"))
    body = [fig_html(layout(fig, "Cost per 1M requests at the p99 ≤ 50 ms capacity", "", "USD / 1M requests", h=380)),
            ]
    fig = go.Figure()
    for _, r in sc.iterrows():
        if r.p99_at_ref:
            fig.add_trace(go.Scatter(x=[r.usd_per_m_alloc], y=[r.p99_at_ref], mode="markers+text", text=[label(r.stk)],
                                     textposition="top center", marker=dict(size=14, color=color(r.stk)), showlegend=False))
    body.append(fig_html(layout(fig, "Cost vs latency (lower-left is better)", "USD / 1M requests (allocated)", "p99 at 100 req/s (ms)")))
    note = (f"Prices used: {esc(PRICE_NAMES.get(pr['name'], pr['name']))}, ${pr['usd_per_vcpu_hour']}/vCPU-h + ${pr['usd_per_gb_hour']}/GB-h, as of {esc(pr['as_of'])}, "
            f"<b>{'verified' if pr['verified'] else 'NOT checked against the price page'}</b>. "
            "CPU time is priced at these rates. Treat absolute dollars as indicative; the ranking is the result.")
    return sub("cost", "Cost", note, "".join(body))


# ------------------------------------------------------------------ Ray Serve feature scenarios
def ingress_section(df) -> str:
    d = df[(df.scenario == "I1") & (df.rows == 200)]
    if d.empty:
        return "" + nodata("I1")
    cap = an.search_capacity(df, "I1", 100).set_index("variant").capacity
    cap50 = an.search_capacity(df, "I1", 50).set_index("variant").capacity
    fig = go.Figure()
    for v in d.variant.unique():
        g = d[(d.variant == v) & (d.kind == "grid")].groupby("rate").p99.median().reset_index()
        fig.add_trace(go.Scatter(x=g.rate, y=g.p99, mode="lines+markers", name=vl(v), line=dict(color=color(v))))
    out = [fig_html(layout(fig, "p99 by ingress", "offered load (req/s)", "p99 (ms)", ymax=250, h=360))]
    fig = go.Figure()
    for slo, c in ((50, cap50), (100, cap)):
        fig.add_trace(go.Bar(x=[vl(x) for x in c.index], y=list(c.values), name=f"p99 ≤ {slo} ms"))
    out.append(fig_html(layout(fig, "Capacity by ingress (short search)", "", "req/s", h=340)))
    return "" + "".join(out)


def family_section(df, cold_v) -> str:
    out = []
    d = df[df.scenario == "F1"]
    if d.empty:
        out.append("" + nodata("F1"))
    else:
        c50, c100 = an.search_capacity(df, "F1", 50), an.search_capacity(df, "F1", 100)
        fx = d[d.kind == "family_fixed"]
        rows = []
        for v in c100.variant:
            f = fx[fx.variant == v]
            dv = (f.prom.iloc[0].get("derived") if len(f) and f.prom.iloc[0] else None) or {}
            miss = (dv["mux_loads"] / dv["mux_requests"]) if dv.get("mux_requests") else None
            rows.append({"variant": vl(v), "stack": label(c100[c100.variant == v].stk.iloc[0]),
                         "c50": int(c50[c50.variant == v].capacity.iloc[0]) if (c50.variant == v).any() else "–",
                         "c100": int(c100[c100.variant == v].capacity.iloc[0]),
                         "p99": fmt(f.p99.median() if len(f) else None, 1, " ms"),
                         "rss": fmt(f.rss.median() if len(f) else None, 0, " MB"),
                         "cpu": fmt(f.cpu.median() if len(f) else None, 2),
                         "miss": fmt(100 * miss, 1, " %") if miss is not None else "–"})
        fig = go.Figure(go.Bar(x=[vl(x) for x in c100.variant], y=c100.capacity, name="p99 ≤ 100 ms"))
        fig.add_trace(go.Bar(x=[vl(x) for x in c50.variant], y=c50.capacity, name="p99 ≤ 50 ms"))
        out.append(""
                   + fig_html(layout(fig, "Capacity, 16-model family, Zipf(1.1) traffic", "", "req/s", h=360))
                   + table(rows, [("variant", "Config"), ("stack", "Stack"), ("c50", "Capacity (p99 ≤ 50 ms)"), ("c100", "Capacity (p99 ≤ 100 ms)"),
                                  ("p99", "p99 at 60 req/s"), ("rss", "Peak RSS"), ("cpu", "Cores busy"), ("miss", "Model load rate")]))
    if not cold_v.empty:
        fig = go.Figure()
        for (v, kind), g in cold_v.groupby(["variant", "kind"]):
            fig.add_trace(go.Box(y=g.first_ms, name=f"{vl(v)} ({kind})", boxpoints="all"))
        fig.add_trace(go.Scatter(x=[f"{vl(v)} ({k})" for (v, k), g in cold_v.groupby(["variant", "kind"])],
                                 y=[g.steady_ms.median() for _, g in cold_v.groupby(["variant", "kind"])],
                                 mode="markers", name="steady-state", marker=dict(symbol="line-ew", size=18, color="#333", line=dict(width=2))))
        out.append(fig_html(layout(fig, "First-request latency, cold model (ms)", "", "ms", logy=True, h=380)))
    return "".join(out)


def fanout_section(df) -> str:
    out = []
    d = df[(df.scenario == "F3") & (df.kind == "fanout")]
    if d.empty:
        out.append("" + nodata("F3"))
    else:
        d = d.assign(system=d.variant.str.replace(r"_k\d+$", "", regex=True))
        f1, f2 = go.Figure(), go.Figure()
        for s, g in d.groupby("system"):
            a = g.groupby("k").p99.median().reset_index()
            f1.add_trace(go.Scatter(x=a.k, y=a.p99, mode="lines+markers", name=vl(s), line=dict(color=color(s))))
            b = g.assign(v=g.cpu / g.rps * 1000).groupby("k").v.median().reset_index()
            f2.add_trace(go.Scatter(x=b.k, y=b.v, mode="lines+markers", name=vl(s), line=dict(color=color(s))))
        out.append(""
                   + fig_html(layout(f1, "p99 vs K (40 req/s)", "K models per request", "p99 (ms)", ymax=300, h=340))
                   + fig_html(layout(f2, "CPU cost per request vs K", "K models per request", "CPU ms / request", h=340)))
    return "".join(out)


def pipeline_section(df) -> str:
    out = []
    p = df[(df.scenario == "F5") & (df.kind == "grid")]
    if p.empty:
        out.append("" + nodata("F5"))
    else:
        fig = go.Figure()
        for v, g in p.groupby("variant"):
            a = g.groupby("rate").p99.median().reset_index()
            fig.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=vl(v), line=dict(color=color(v))))
        cap = an.search_capacity(df, "F5", 100)
        out.append(""
                   + fig_html(layout(fig, "p99 vs offered load", "req/s", "p99 (ms)", ymax=250, h=340))
                   + table([{"v": vl(r.variant), "c": int(r.capacity)} for r in cap.itertuples()], [("v", "Stack"), ("c", "Capacity (p99 ≤ 100 ms)")]))
    return "".join(out)


def batching_section(df) -> str:
    c50, c100 = an.search_capacity(df, "S1_batch", 50), an.search_capacity(df, "S1_batch", 100)
    if c100.empty:
        return nodata("S1_batch")
    c100 = c100.sort_values(["variant", "stk"])
    names = [f"{label(r.stk)} · {vl(r.variant)}" for r in c100.itertuples()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=c100.capacity, name="p99 ≤ 100 ms"))
    m50 = c50.set_index(["variant", "stk"]).capacity
    fig.add_trace(go.Bar(x=names, y=[m50.get((r.variant, r.stk), 0) for r in c100.itertuples()], name="p99 ≤ 50 ms"))
    return fig_html(layout(fig, "Capacity by batching setting (r1 = 1 row per request, r200 = 200 rows)", "", "req/s", h=400))


def _rel(series_t0, t_measure):  # seconds offset of k6 series start vs t_measure
    return (series_t0 - t_measure) if series_t0 and t_measure else 0.0


def autoscale_section(df, slo=50) -> str:
    d = df[(df.scenario == "A1") & (df.kind == "burst")]
    if d.empty:
        return "" + nodata("A1")
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
                f_lat.add_trace(go.Scatter(x=t, y=[s["p99"] for s in ser], name=vl(pol), mode="lines", line=dict(color=c)))
                f_err.add_trace(go.Scatter(x=t, y=[s["err"] for s in ser], name=vl(pol), mode="lines", line=dict(color=c)))
                pt = r.prom.get("t") or []
                if gauge and pt:
                    off = (r.t_measure or pt[0])
                    f_rep.add_trace(go.Scatter(x=[x - off for x in pt], y=gauge, name=vl(pol), mode="lines", line=dict(color=c, shape="hv")))
        rows.append({"p": vl(pol), "rec": fmt(np.nanmedian([x for x in rec if x is not None] or [np.nan]), 0, " s") if any(x is not None for x in rec) else "never",
                     "err": fmt(np.median(errs), 0), "p99": fmt(np.nanmedian(p99b), 1, " ms"),
                     "core": fmt(np.nanmedian([x for x in cores if x] or [np.nan]), 0), "max": fmt(np.nanmax([x for x in maxr if x] or [np.nan]), 0)})
    for f in (f_lat, f_rep, f_err):
        f.add_vrect(x0=pre, x1=pre + int(df[df.scenario == "A1"].steps.iloc[0].split(",")[1].split(":")[1]),
                    fillcolor="#b5473a", opacity=0.07, line_width=0, annotation_text="10× burst", annotation_position="top left")
    return (""
            + fig_html(layout(f_lat, "p99 per second during the burst (first repetition)", "seconds into measurement", "p99 (ms)", logy=True, h=360))
            + fig_html(layout(f_rep, "Healthy replicas", "seconds into measurement", "replicas", h=300))
            + fig_html(layout(f_err, "Errors per second", "seconds into measurement", "errors", h=300))
            + table(rows, [("p", "Policy"), ("rec", f"Time to recover (p99 < {slo} ms)"), ("err", "Errors"),
                           ("p99", "Median p99 after burst start"), ("core", "Core-seconds"), ("max", "Max replicas")]))


def backpressure_section(df) -> str:
    d = df[(df.scenario == "R1") & df.kind.isin(["search", "overload"])]
    if d.empty:
        return "" + nodata("R1")
    f1, f2 = go.Figure(), go.Figure()
    rows = []
    for v, g in d.groupby("variant"):
        a = g.groupby("rate").agg(ok=("ok", "median"), dur=("duration_s", "median"), p99=("p99", "median")).reset_index()
        f1.add_trace(go.Scatter(x=a.rate, y=a.ok / a.dur, mode="lines+markers", name=vl(v), line=dict(color=color(v))))
        f2.add_trace(go.Scatter(x=a.rate, y=a.p99, mode="lines+markers", name=vl(v), line=dict(color=color(v))))
    mx = float(d.rate.max())
    f1.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="perfect", line=dict(dash="dot", color="#999")))
    for r in d[d.kind == "overload"].itertuples():
        codes = ", ".join(f"{k}: {v}" for k, v in r.error_codes.items()) or "none"
        rows.append({"v": vl(r.variant), "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0),
                     "p99": fmt(r.p99, 1, " ms"), "codes": codes})
    return (""
            + fig_html(layout(f1, "Goodput vs offered load", "offered (req/s)", "goodput (req/s)", h=360))
            + fig_html(layout(f2, "p99 of successful requests vs offered load", "offered (req/s)", "p99 (ms)", ymax=500, h=340))
            + table(rows, [("v", "Config"), ("f", "× capacity"), ("rate", "Offered"), ("good", "Goodput"), ("p99", "p99 (ok)"), ("codes", "Non-200 responses")]))


def failure_section(df) -> str:
    d = df[(df.scenario == "R2") & (df.kind == "failure")]
    if d.empty:
        return "" + nodata("R2")
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
                fig.add_trace(go.Scatter(x=t, y=[s["err"] for s in ser], name=vl(pol), line=dict(color=color(pol))))
                fig2.add_trace(go.Scatter(x=t, y=[s["p99"] for s in ser], name=vl(pol), line=dict(color=color(pol))))
        rows.append({"p": vl(pol), "err": fmt(np.median(errs) if errs else None, 0), "win": fmt(np.median(win) if win else None, 0, " s"),
                     "rec": fmt(np.nanmedian([x for x in recs if x is not None] or [np.nan]), 0, " s")})
    for f in (fig, fig2):
        f.add_vline(x=0, line=dict(color="#b5473a", dash="dash"), annotation_text="kill")
    return (""
            + fig_html(layout(fig, "Errors per second around the kill (first repetition)", "seconds relative to kill", "errors", h=320))
            + fig_html(layout(fig2, "p99 per second around the kill", "seconds relative to kill", "p99 (ms)", logy=True, h=320))
            + table(rows, [("p", "Stack"), ("err", "Errors (median)"), ("win", "Seconds with errors"), ("rec", "Time to recover (p99 < 2× baseline)")]))


def overload_section(df, order, cold) -> str:
    d = df[(df.scenario == "S1") & (df.kind == "overload")]
    rows = [{"s": label(r.stk), "f": f"{r.factor}×", "rate": r.rate, "good": fmt(r.ok / r.duration_s, 0), "p99": fmt(r.p99, 0, " ms"),
             "err": r.err, "dropped": r.dropped} for r in d.sort_values(["stk", "factor"]).itertuples()]
    return sub("overload", "Overload behaviour",
                   "Offered load at 1.5× and 3× each stack's capacity at p99 ≤ 100 ms (first repetition). Goodput = successful requests per second.",
                   table(rows, [("s", "Stack"), ("f", "× capacity"), ("rate", "Offered"), ("good", "Goodput"), ("p99", "p99"),
                                ("err", "Errors"), ("dropped", "Dropped arrivals")]))


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
            card("Most CPU-efficient", esc(label(e.iloc[0].stk)), f"{e.iloc[0].core_ms_per_req:.2f} core-ms / request")
        f = sc.dropna(subset=["cold_s"]).sort_values("cold_s")
        if len(f):
            card("Fastest cold start", esc(label(f.iloc[0].stk)), f"{f.iloc[0].cold_s:.1f} s")
    i1 = an.search_capacity(df, "I1", 50)
    if len(i1):
        top = i1.sort_values("capacity", ascending=False).iloc[0]
        card("Best Ray ingress", esc(vl(top.variant)), f"{top.capacity} req/s at 50 ms")
    f1 = an.search_capacity(df, "F1", 100)
    if len(f1):
        top = f1.sort_values("capacity", ascending=False).iloc[0]
        card("Best model-family layout", esc(vl(top.variant)), f"{top.capacity} req/s at 100 ms")
    table_html = ""
    if not sc.empty:
        rows = [{"stack": label(r.stk), "cap": r.capacity, "ci": r.ci, "p99": fmt(r.p99_at_ref, 1, " ms"), "cores": fmt(r.cores_at_cap, 2),
                 "cms": fmt(r.core_ms_per_req, 2), "rpv": fmt(r.rps_per_vcpu, 1), "rss": fmt(r.rss_mb, 0, " MB"),
                 "cold": fmt(r.cold_s, 1, " s"), "ca": fmt(r.usd_per_m_alloc, 3), "cu": fmt(r.usd_per_m_used, 3)} for r in sc.itertuples()]
        table_html = table(rows, [("stack", "Stack"), ("cap", "Capacity (p99 ≤ 50 ms)"), ("ci", "95 % CI"), ("p99", "p99 @ 100 req/s"),
                                  ("cores", "Cores busy @ capacity"), ("cms", "CPU ms / request"), ("rpv", "req/s / vCPU"), ("rss", "Peak RSS"),
                                  ("cold", "Cold start"), ("ca", "$ / 1M (allocated)"), ("cu", "$ / 1M (consumed)")], cls="score")
    tests_line = " · ".join(tname(k) for k in TESTS)
    return section("summary", "Summary", "DeepFM small (ONNX), 200 rows per request, KServe v2 JSON, k6 open-loop. "
                   "Single host: 6 CPUs for serving, 2 for the load generator. Use the numbers to rank stacks, not to size a fleet."
                   f'<p class="muted"><b>Tests:</b> {esc(tests_line)}.</p>',
                   f'<div class="cards">{"".join(cards)}</div>{table_html}')

def versions_section(meta) -> str:
    v = meta.get("versions", {})
    flat = "".join(f"<tr><td>{esc(k)}</td><td>{esc(val)}</td></tr>" for k, val in sorted(
        (v if isinstance(v, dict) else {}).items()) if k in VERSION_KEYS and not isinstance(val, (dict, list)))
    return section("versions", "Software versions", "", f'<div class="tw"><table><tbody>{flat}</tbody></table></div>')


def methodology_section(df, meta, rows) -> str:
    m = an.matrix()
    inval = int((~df.valid).sum()) if len(df) else 0
    slos = ", ".join(map(str, m["slos_ms"]))
    nrep = m["scenarios"]["S1"]["reps"]
    return section("method", "Methodology", "How the numbers were produced. Read this before comparing.", f"""<ul>
<li><b>Load:</b> k6 open-loop (constant arrival rate) on CPUs 6-7; servers on CPUs 0-5 with identical memory limits.</li>
<li><b>Capacity:</b> highest grid rate with median-of-{nrep} p99 ≤ SLO ({slos} ms), zero errors and ≤ 0.1 % dropped arrivals; 95 % CI by bootstrapping the repetitions.</li>
<li><b>Order:</b> stack order shuffled per repetition so thermal drift does not favour one stack; warm-up excluded from every statistic.</li>
<li><b>Validity:</b> k6 above 80 % of its cores, a collapsed CPU clock (below 60 % of maximum with thermal-throttle events) or a container restart marks a step invalid. Mild throttling at the thermal limit (clock near 85 % of maximum) is the same for every stack and is not flagged. Invalid steps are excluded from capacity and listed in the appendix. {inval} of {len(df)} steps are invalid.</li>
<li><b>Server metrics:</b> Ray, Triton and FastAPI Prometheus counters at 1 Hz; per-process CPU from cgroups and /proc.</li>
<li><b>Model:</b> DeepFM with seeded random weights, ONNX Runtime on CPU, identical session options in every stack.</li>
<li><b>Single host:</b> serving and load generator share one CPU package; absolute numbers will differ on a cluster.</li></ul>
""")


def appendix_section(df, rows) -> str:
    cov = df.groupby(["scenario", "stk"]).agg(steps=("valid", "size"), invalid=("valid", lambda s: int((~s).sum()))).reset_index()
    cov_rows = [{"s": tname(r.scenario), "st": label(r.stk), "n": r.steps, "i": r.invalid} for r in cov.itertuples()]
    bad = [{"s": tname(r.scenario), "st": label(r.stk), "v": vl(r.variant), "rate": r.rate, "why": esc(r.invalid)} for r in df[~df.valid].itertuples()]
    allrows = [{"s": tname(r.scenario), "v": vl(r.variant), "st": r.stk, "k": r.kind, "rate": r.rate or r.vus, "rep": r.rep,
                "p50": fmt(r.p50, 2), "p99": fmt(r.p99, 2), "ok": r.ok, "err": r.err, "dropped": r.dropped, "cpu": fmt(r.cpu, 2),
                "rss": fmt(r.rss, 0), "valid": "yes" if r.valid else "NO"} for r in df.itertuples()]
    return section("appendix", "Appendix", "Coverage, invalid steps and every measured step (click a header to sort).",
                   "<h3>Coverage</h3>" + table(cov_rows, [("s", "Test"), ("st", "Stack"), ("n", "Steps"), ("i", "Invalid")])
                   + "<h3>Invalid steps</h3>" + table(bad, [("s", "Test"), ("st", "Stack"), ("v", "Variant"), ("rate", "Rate"), ("why", "Reason")])
                   + "<h3>All steps</h3>" + table(allrows, [("s", "Test"), ("v", "Variant"), ("st", "Stack"), ("k", "Kind"), ("rate", "Rate / VUs"),
                                                          ("rep", "Rep"), ("p50", "p50 ms"), ("p99", "p99 ms"), ("ok", "OK"), ("err", "Err"),
                                                          ("dropped", "Dropped"), ("cpu", "Cores"), ("rss", "RSS MB"), ("valid", "Valid")], cls="sortable")
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
.note{background:#fff4d6;color:#5a4300;border:1px solid #e3c36a;border-radius:6px;padding:8px 12px;margin:10px 0;font-size:14px}.sortable th{cursor:pointer}.test{border-left:3px solid var(--acc);padding:2px 14px;margin:14px 0}.test h3{margin:4px 0}.test p{margin:4px 0}.fig{margin:8px 0}pre{background:var(--card);padding:10px;border-radius:6px;overflow-x:auto}
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
        lines += ["## Scorecard (p99 ≤ 50 ms, DeepFM small, 200 rows per request)", "",
                  "| Stack | Capacity (req/s) | 95 % CI | p99 @ 100 req/s | CPU ms / request | req/s / vCPU | Peak RSS | Cold start | $ / 1M (allocated) | $ / 1M (consumed) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sc.itertuples():
            lines.append(f"| {label(r.stk)} | {r.capacity} | {r.ci} | {fmt(r.p99_at_ref, 1, ' ms')} | {fmt(r.core_ms_per_req, 2)} | "
                         f"{fmt(r.rps_per_vcpu, 1)} | {fmt(r.rss_mb, 0, ' MB')} | {fmt(r.cold_s, 1, ' s')} | {fmt(r.usd_per_m_alloc, 3)} | {fmt(r.usd_per_m_used, 3)} |")
        lines += ["", f"Prices: {pr['name']} ${pr['usd_per_vcpu_hour']}/vCPU-h, ${pr['usd_per_gb_hour']}/GB-h "
                      f"({'verified' if pr['verified'] else 'NOT verified'}, as of {pr['as_of']}).", ""]
    for sid, title in (("I1", "Ingress (capacity, p99 ≤ 50 ms)"), ("F1", "Model family (capacity, p99 ≤ 100 ms)"),
                       ("R1", "Overload and backpressure (capacity, p99 ≤ 100 ms)")):
        c = an.search_capacity(df, sid, 50 if sid == "I1" else 100)
        if len(c):
            lines += [f"## {title}", "", "| Config | Capacity (req/s) |", "|---|---|"] + [f"| {vl(r.variant)} | {r.capacity} |" for r in c.itertuples()] + [""]
    return "\n".join(lines)


def build(env: str = "laptop", out_html: Path | None = None, out_md: Path | None = None, note: str = "") -> Path:
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
    parts = [
        summary_section(df, sc, cap, cold_v), methodology_section(df, meta, rows),
        group("s1", "Latency and capacity", ["S1"], latency_section(df, order) + spectrum_section(df, order)
              + capacity_section(df, order, cap) + breakdown_section(df, order) + resources_section(df, order, cold)
              + cost_section(sc) + overload_section(df, order, cold)),
        group("scale", "Replica scaling", ["S1_scale"], fleet_section(df, order, sc)),
        group("batch", "Batching", ["S1_batch"], batching_section(df)),
        group("ingress", "Ingress", ["I1"], ingress_section(df)),
        group("family", "Model family (multiplexing and cold load)", ["F1", "F2"], family_section(df, cold_v)),
        group("fanout", "Ensemble fan-out", ["F3"], fanout_section(df)),
        group("pipeline", "Multi-stage pipeline", ["F5"], pipeline_section(df)),
        group("autoscale", "Autoscaling under a 10× burst", ["A1"], autoscale_section(df)),
        group("backpressure", "Overload and backpressure", ["R1"], backpressure_section(df)),
        group("failure", "Failure recovery", ["R2"], failure_section(df)),
        versions_section(meta), appendix_section(df, rows)]
    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in (
        ("summary", "Summary"), ("method", "Methodology"), ("s1", "Latency and capacity"), ("scale", "Replica scaling"),
        ("batch", "Batching"), ("ingress", "Ingress"), ("family", "Model family"), ("fanout", "Fan-out"),
        ("pipeline", "Pipeline"), ("autoscale", "Autoscaling"), ("backpressure", "Overload"), ("failure", "Failure"),
        ("versions", "Versions"), ("appendix", "Appendix")))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>Serving benchmark report{" (incomplete)" if note else ""}</title><style>{CSS}</style><script>{get_plotlyjs()}</script></head><body>"
           f"<header><h1>Serving benchmark: FastAPI vs Ray Serve vs Triton</h1><div class='muted'>{datetime.now():%Y-%m-%d} · {len(df)} measured steps</div>{f'<div class="note">{esc(note)}</div>' if note else ""}</header><nav>{nav}</nav><main>{''.join(parts)}</main>"
           f"<script>{SORT_JS}</script></body></html>")
    out = out_html or ROOT / "bench" / "report" / "report.html"
    out.write_text(doc)
    (out_md or ROOT / "REPORT.md").write_text(to_markdown(sc, df, env))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="laptop")
    ap.add_argument("--html", type=Path, help="output file (default bench/report/report.html)")
    ap.add_argument("--note", default="", help="banner text, e.g. to mark an incomplete run")
    a = ap.parse_args()
    out = build(a.env, a.html, None, a.note)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and REPORT.md")


if __name__ == "__main__":
    main()
