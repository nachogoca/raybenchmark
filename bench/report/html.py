"""Self-contained HTML benchmark report: inline SVG charts, no CDN, works offline, light/dark.

    uv run python -m bench.report.html            # -> bench/report/report.html
    just report-html

Reads bench/results/*.jsonl plus bench/results/archive/*.jsonl (older configs kept for the
"tuning journey" chart) and reuses bench.report.analyze, so numbers match the CSVs exactly.
"""
from __future__ import annotations

import argparse
import html
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from bench.report import analyze

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

NAMES = {
    "fastapi_ort": "FastAPI", "ray_ort": "Ray Serve", "ray_fastapi": "Ray Serve + FastAPI",
    "ray_triton_grpc": "Ray Serve → Triton (gRPC)", "ray_triton_inproc": "Ray Serve + in-process Triton",
    "triton": "Triton (standalone)",
}
# Okabe-Ito: colour-blind safe, one per stack; runtime variants reuse the stack colour (dashed)
COLORS = {"fastapi_ort": "#0072B2", "ray_ort": "#E69F00", "ray_fastapi": "#56B4E9",
          "ray_triton_grpc": "#009E73", "ray_triton_inproc": "#CC79A7", "triton": "#D55E00"}
ENV_DESC = {"laptop-cpu": "default", "laptop-cpu-tuned": "tuned threads",
            "laptop-cpu-haproxy": "tuned + HAProxy"}
SLOS = (20, 50, 100)

COVERAGE = [  # (axis, status, detail)
    ("Serving stacks", "covered", "FastAPI · Ray Serve · Ray Serve + FastAPI · Ray → Triton gRPC · Ray + in-process Triton · Triton"),
    ("Runtime: ONNX Runtime", "covered", "all stacks"),
    ("Runtime: PyTorch (eager)", "covered", "FastAPI, Ray Serve, Ray Serve + FastAPI (Triton libtorch: planned)"),
    ("Models", "partial", "DeepFM and DCNv2 small supported; large sizes via the `large` profile"),
    ("Batching on / off", "covered", "Ray @serve.batch and Triton dynamic batching"),
    ("Replica scaling 1·2·4", "planned", "`core` / `full` profiles; smoke runs 1 replica"),
    ("Rows per request 1·50·200·500", "partial", "smoke runs 200; `full` sweeps all four"),
    ("Ray front end: Python proxy vs HAProxy", "covered", "default vs `laptop-cpu-haproxy` env"),
    ("ORT thread budget (cores left for Ray)", "covered", "default vs `tuned` env"),
    ("Binary protocol (gRPC / raw tensors)", "planned", "JSON decode is ~0.8 ms per 200-row request"),
    ("Realistic traffic: mixed rows, bursts, overload", "planned", "overload steps exist in `core` / `full`"),
    ("Multi-stage pipeline (preprocess → model)", "planned", "where Ray composition should pay off"),
    ("GPU", "cloud", "`k8s-gpu` env; laptop GPU unused"),
    ("Kubernetes (KubeRay) routing and autoscaling", "cloud", "`deploy/k8s`"),
]


def pretty(stack: str) -> str:
    base, _, rt = stack.partition("[")
    return NAMES.get(base, base) + (" · PyTorch" if rt.startswith("torch") else "")


def base(stack: str) -> str:
    return stack.split("[")[0]


def esc(x) -> str:
    return html.escape(str(x))


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(analyze.RESULTS.glob("*.jsonl")) + sorted((analyze.RESULTS / "archive").glob("*.jsonl"))
    df = analyze.flatten(analyze.load_rows(files))
    slos = yaml.safe_load((ROOT / "config" / "matrix.yaml").read_text())["slos_ms"]
    return df, analyze.capacity_table(df, slos), analyze.coldstart_table(df)


def variant(r) -> str:
    r = r._asdict() if hasattr(r, "_asdict") else r  # accepts itertuples rows or dicts
    bits = [ENV_DESC.get(r["env"], r["env"])]
    if base(r["stack"]) in ("ray_ort", "ray_fastapi", "ray_triton_grpc", "ray_triton_inproc", "triton"):
        bits.append("batching on" if r["batching"] else "batching off")
    if r.get("replicas", 1) != 1:
        bits.append(f"{r['replicas']} replicas")
    return " · ".join(bits)


# ---------------------------------------------------------------- SVG helpers
def hbar(rows: list[dict], *, unit: str, width=760, label_w=230, bar_h=26, gap=10) -> str:
    """rows: label, sub, value, color, hatch(bool), tip"""
    top = max((r["value"] for r in rows), default=1) or 1
    h = len(rows) * (bar_h + gap) + 28
    plot_w = width - label_w - 70
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for i in range(5):  # gridlines
        x = label_w + plot_w * i / 4
        out.append(f'<line x1="{x:.0f}" x2="{x:.0f}" y1="0" y2="{h-22}" class="grid"/>'
                   f'<text x="{x:.0f}" y="{h-6}" text-anchor="middle" class="tick">{top*i/4:.0f}</text>')
    for i, r in enumerate(rows):
        y = i * (bar_h + gap) + 4
        w = max(plot_w * r["value"] / top, 2 if r["value"] else 0)
        fill = f'url(#h-{i})' if r.get("hatch") else r["color"]
        pat = (f'<defs><pattern id="h-{i}" width="6" height="6" patternUnits="userSpaceOnUse" '
               f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{r["color"]}" opacity=".35"/>'
               f'<rect width="3" height="6" fill="{r["color"]}"/></pattern></defs>') if r.get("hatch") else ""
        out.append(f'{pat}<g><title>{esc(r["tip"])}</title>'
                   f'<text x="{label_w-10}" y="{y+bar_h/2-3:.0f}" text-anchor="end" class="lbl">{esc(r["label"])}</text>'
                   f'<text x="{label_w-10}" y="{y+bar_h/2+10:.0f}" text-anchor="end" class="sub">{esc(r["sub"])}</text>'
                   f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{fill}"/>'
                   f'<text x="{label_w+w+6:.1f}" y="{y+bar_h/2+4:.0f}" class="val">{r["value"]:.0f} {unit}</text></g>')
    out.append("</svg>")
    return "".join(out)


def vbar(rows: list[dict], *, unit: str, refs: list[tuple[str, float, str]] = (), width=760, h=300) -> str:
    top = max([r["value"] for r in rows] + [v for _, v, _ in refs] + [1]) * 1.12
    pad_l, pad_b, n = 46, 62, max(len(rows), 1)
    plot_w, plot_h = width - pad_l - 12, h - pad_b - 14
    slot = plot_w / n
    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for i in range(5):
        y = 14 + plot_h * (1 - i / 4)
        out.append(f'<line x1="{pad_l}" x2="{width-12}" y1="{y:.0f}" y2="{y:.0f}" class="grid"/>'
                   f'<text x="{pad_l-6}" y="{y+4:.0f}" text-anchor="end" class="tick">{top*i/4:.0f}</text>')
    for name, v, col in refs:
        y = 14 + plot_h * (1 - v / top)
        out.append(f'<line x1="{pad_l}" x2="{width-12}" y1="{y:.0f}" y2="{y:.0f}" stroke="{col}" '
                   f'stroke-dasharray="5 4" stroke-width="1.5"/><text x="{width-14}" y="{y-5:.0f}" '
                   f'text-anchor="end" class="tick" fill="{col}">{esc(name)} {v:.0f}</text>')
    for i, r in enumerate(rows):
        bw = slot * 0.56
        x = pad_l + slot * i + (slot - bw) / 2
        bh = plot_h * r["value"] / top
        y = 14 + plot_h - bh
        out.append(f'<g><title>{esc(r["tip"])}</title><rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                   f'height="{max(bh, 1.5):.1f}" rx="3" fill="{r["color"]}"/>'
                   f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="val">{r["value"]:.0f}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{h-pad_b+16}" text-anchor="middle" class="lbl2">{esc(r["label"])}</text>'
                   f'<text x="{x+bw/2:.1f}" y="{h-pad_b+30}" text-anchor="middle" class="sub">{esc(r.get("sub",""))}</text></g>')
    out.append(f'<text x="12" y="10" class="tick">{esc(unit)}</text></svg>')
    return "".join(out)


def lines(series: list[dict], *, width=760, h=360) -> str:
    """series: label, color, dashed, pts=[(rate, p99, clean)]; log-x on rate, log-y on p99."""
    import math

    pts = [p for s in series for p in s["pts"]]
    if not pts:
        return "<p>no data</p>"
    xmin, xmax = min(p[0] for p in pts) * 0.9, max(p[0] for p in pts) * 1.1
    ymin, ymax = 1.0, max(max(p[1] for p in pts), 120) * 1.2
    pad_l, pad_b = 52, 40
    pw, ph = width - pad_l - 14, h - pad_b - 14

    def X(v): return pad_l + pw * (math.log(v) - math.log(xmin)) / (math.log(xmax) - math.log(xmin))
    def Y(v): return 14 + ph * (1 - (math.log(max(v, ymin)) - math.log(ymin)) / (math.log(ymax) - math.log(ymin)))

    out = [f'<svg viewBox="0 0 {width} {h}" role="img" class="chart">']
    for y in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if ymin <= y <= ymax:
            out.append(f'<line x1="{pad_l}" x2="{width-14}" y1="{Y(y):.0f}" y2="{Y(y):.0f}" class="grid"/>'
                       f'<text x="{pad_l-6}" y="{Y(y)+4:.0f}" text-anchor="end" class="tick">{y}</text>')
    for x in (10, 20, 50, 100, 200, 500, 1000):
        if xmin <= x <= xmax:
            out.append(f'<text x="{X(x):.0f}" y="{h-pad_b+16}" text-anchor="middle" class="tick">{x}</text>')
    for slo in SLOS:
        out.append(f'<line x1="{pad_l}" x2="{width-14}" y1="{Y(slo):.0f}" y2="{Y(slo):.0f}" class="slo"/>'
                   f'<text x="{width-16}" y="{Y(slo)-4:.0f}" text-anchor="end" class="tick">SLO {slo} ms</text>')
    for s in sorted(series, key=lambda s: s["label"]):
        d = " ".join(f'{"M" if i == 0 else "L"}{X(r):.1f},{Y(p):.1f}' for i, (r, p, _) in enumerate(s["pts"]))
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        out.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"{dash}/>')
        for r, p, clean in s["pts"]:
            fill = s["color"] if clean else "var(--bg)"
            out.append(f'<circle cx="{X(r):.1f}" cy="{Y(p):.1f}" r="3.6" fill="{fill}" stroke="{s["color"]}" stroke-width="1.6">'
                       f'<title>{esc(s["label"])}: {r:.0f} req/s → p99 {p:.1f} ms{"" if clean else " (errors or drops)"}</title></circle>')
    out.append(f'<text x="{pad_l+pw/2:.0f}" y="{h-4}" text-anchor="middle" class="tick">offered load, req/s (log)</text>'
               f'<text x="12" y="10" class="tick">p99 latency, ms (log)</text></svg>')
    return "".join(out)


def legend(items: list[tuple[str, str, bool]]) -> str:
    sw = "".join(f'<span class="key"><i style="background:{c}"{" class=dash" if d else ""}></i>{esc(n)}</span>'
                 for n, c, d in items)
    return f'<div class="legend">{sw}</div>'


# ---------------------------------------------------------------- sections
def best_per_stack(cap: pd.DataFrame, model: str, rows: int, slo: int) -> pd.DataFrame:
    c = cap[(cap.model == model) & (cap.rows == rows) & (cap.slo_ms == slo)].copy()
    c = c[c.capacity_rps > 0]
    return c.loc[c.groupby("stack").capacity_rps.idxmax()].sort_values("capacity_rps", ascending=False)


def headline_charts(cap, model, rows) -> str:
    panes, tabs = [], []
    for i, slo in enumerate(SLOS):
        b = best_per_stack(cap, model, rows, slo)
        items = [{"label": pretty(r.stack), "sub": variant(r), "value": r.capacity_rps,
                  "color": COLORS.get(base(r.stack), "#777"), "hatch": "[" in r.stack,
                  "tip": f"{pretty(r.stack)} — {variant(r)}\n{r.capacity_rps:.0f} req/s at p99 ≤ {slo} ms "
                         f"(p99 {r.p99_ms} ms, p50 {r.p50_ms} ms, {r.cpu_cores} cores, {r.rss_peak_mb:.0f} MB RSS)"}
                 for r in b.itertuples()]
        tabs.append(f'<button class="tab{" on" if i == 1 else ""}" data-pane="cap{slo}">p99 ≤ {slo} ms</button>')
        panes.append(f'<div class="pane{" on" if i == 1 else ""}" id="cap{slo}">{hbar(items, unit="req/s")}</div>')
    return (f'<div class="tabs">{"".join(tabs)}</div>{"".join(panes)}'
            f'<p class="note">Best configuration found for each stack. Each bar is the highest request rate at which '
            f'p99 latency stayed under the SLO with zero errors; {rows} rows per request, hover for details. '
            f'Hatched = PyTorch runtime.</p>')


def journey(cap, model, rows) -> str:
    """Ray tuning steps at 50 ms: default→batching off→tuned threads→HAProxy, plus Triton/FastAPI refs."""
    c = cap[(cap.model == model) & (cap.rows == rows) & (cap.slo_ms == 50)]

    def get(stack, env, batching):
        s = c[(c.stack == stack) & (c.env == env) & (c.batching == batching)]
        return float(s.capacity_rps.max()) if len(s) else None

    steps = [("Default", "batching on · 6 threads", "laptop-cpu", True),
             ("Batching off", "6 threads", "laptop-cpu", False),
             ("Fewer ORT threads", "4 threads", "laptop-cpu-tuned", False),
             ("HAProxy front end", "4 threads", "laptop-cpu-haproxy", False)]
    rows_ = []
    for name, sub, env, bt in steps:
        v = get("ray_ort", env, bt)
        if v is not None:
            rows_.append({"label": name, "sub": sub, "value": v, "color": COLORS["ray_ort"],
                          "tip": f"Ray Serve · {name} ({sub}): {v:.0f} req/s at p99 ≤ 50 ms"})
    if len(rows_) < 2:
        return ""
    refs = []
    for stack in ("fastapi_ort", "triton"):
        s = c[c.stack == stack]
        if len(s):
            refs.append((NAMES[stack], float(s.capacity_rps.max()), COLORS[stack]))
    return (f'<h2>How Ray Serve got faster</h2>'
            f'<p class="lede">Each step changes one thing for <b>Ray Serve + ORT</b>; capacity at p99 ≤ 50 ms. '
            f'Dashed lines are the FastAPI and Triton references.</p>{vbar(rows_, unit="req/s at p99 ≤ 50 ms", refs=refs)}')


def latency_section(df, cap, model, rows) -> str:
    b = best_per_stack(cap, model, rows, 50)
    series = []
    for r in b.itertuples():
        g = df[(df.stack == r.stack) & (df.env == r.env) & (df.model == model) & (df.rows == rows)
               & (df.batching == r.batching) & (df.replicas == r.replicas)
               & df.step_kind.isin(["search", "confirm"]) & df.p99_ms.notna()]
        pts = sorted({(float(x.rate), float(x.p99_ms), bool(x.err == 0 and x.dropped == 0)) for x in g.itertuples()})
        series.append({"label": f"{pretty(r.stack)} ({variant(r)})", "color": COLORS.get(base(r.stack), "#777"),
                       "dashed": "[" in r.stack, "pts": pts})
    leg = legend([(s["label"], s["color"], s["dashed"]) for s in series])
    return (f'<h2>Latency under load</h2><p class="lede">p99 latency as offered load rises (best configuration per '
            f'stack). A curve that stays flat then turns up has found its limit; hollow points had errors or '
            f'dropped requests.</p>{leg}{lines(series)}')


def resources(cap, cold, model, rows) -> str:
    b = best_per_stack(cap, model, rows, 50)
    cpu = [{"label": pretty(r.stack).replace("Ray Serve", "Ray"), "sub": "", "value": float(r.cpu_cores or 0),
            "color": COLORS.get(base(r.stack), "#777"),
            "tip": f"{pretty(r.stack)}: {r.cpu_cores} cores busy at capacity (of 6)"} for r in b.itertuples()]
    mem = [{"label": pretty(r.stack).replace("Ray Serve", "Ray"), "sub": "", "value": float(r.rss_peak_mb or 0),
            "color": COLORS.get(base(r.stack), "#777"), "tip": f"{pretty(r.stack)}: {r.rss_peak_mb:.0f} MB peak RSS"}
           for r in b.itertuples()]
    cs = []
    if len(cold):
        c = cold[cold.model == model].sort_values("start_to_ready_s")
        c = c.loc[c.groupby("stack").start_to_ready_s.idxmin()]
        cs = [{"label": pretty(r.stack).replace("Ray Serve", "Ray"), "sub": "", "value": float(r.start_to_ready_s),
               "color": COLORS.get(base(r.stack), "#777"),
               "tip": f"{pretty(r.stack)}: ready in {r.start_to_ready_s:.1f} s, idle RSS {r.rss_idle_mb:.0f} MB"}
              for r in c.itertuples()]
    return ('<h2>Resources</h2><div class="grid3">'
            f'<div><h3>CPU cores busy at capacity</h3>{vbar(cpu, unit="cores (6 available)", h=260, width=420)}</div>'
            f'<div><h3>Peak memory</h3>{vbar(mem, unit="MB", h=260, width=420)}</div>'
            f'<div><h3>Cold start</h3>{vbar(cs, unit="seconds to ready", h=260, width=420)}</div></div>')


def coverage_html() -> str:
    lab = {"covered": "tested", "partial": "partly", "planned": "planned", "cloud": "cloud phase"}
    rows = "".join(f'<tr><td>{esc(a)}</td><td><span class="pill {s}">{lab[s]}</span></td><td>{esc(d)}</td></tr>'
                   for a, s, d in COVERAGE)
    return (f'<h2>What this benchmark covers</h2><table class="cov"><thead><tr><th>Axis</th><th>Status</th>'
            f'<th>Detail</th></tr></thead><tbody>{rows}</tbody></table>')


def table_html(cap, model, rows) -> str:
    c = cap[(cap.model == model) & (cap.rows == rows)].copy()
    piv = c.pivot_table(index=["stack", "env", "batching", "replicas"], columns="slo_ms", values="capacity_rps", aggfunc="max")
    out = []
    for (stack, env, batching, replicas), r in piv.iterrows():
        tag = variant({"env": env, "stack": stack, "batching": batching, "replicas": replicas})
        out.append((r.get(50, 0), f'<tr><td>{esc(pretty(stack))}</td><td>{esc(tag)}</td>' +
                    "".join(f'<td class="n">{r.get(s, 0):.0f}</td>' for s in SLOS) + "</tr>"))
    body = "".join(x for _, x in sorted(out, key=lambda t: -t[0]))
    return ('<h2>All configurations</h2><table class="all"><thead><tr><th>Stack</th><th>Config</th>'
            + "".join(f'<th class="n">p99 ≤ {s} ms</th>' for s in SLOS) + f'</tr></thead><tbody>{body}</tbody></table>')


CSS = """
:root{--bg:#fff;--fg:#1d2125;--mute:#5c6670;--grid:#e3e7eb;--card:#f6f8fa;--line:#d5dbe1;--accent:#0072B2}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15181c;--fg:#e6e9ec;--mute:#9aa5b0;--grid:#2a3036;--card:#1c2025;--line:#333a42;--accent:#56B4E9}}
:root[data-theme=dark]{--bg:#15181c;--fg:#e6e9ec;--mute:#9aa5b0;--grid:#2a3036;--card:#1c2025;--line:#333a42;--accent:#56B4E9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:21px;margin:44px 0 6px}h3{font-size:14px;margin:0 0 4px;color:var(--mute);font-weight:600}
.lede,.note,.meta{color:var(--mute);margin:4px 0 14px}.note{font-size:13px}
.chart{width:100%;height:auto;display:block}.chart .grid{stroke:var(--grid)}.chart .slo{stroke:var(--mute);stroke-dasharray:3 4;opacity:.6}
.chart text{fill:var(--fg);font:12px system-ui,sans-serif}.chart .tick,.chart .sub{fill:var(--mute);font-size:11px}
.chart .lbl{font-weight:600;font-size:13px}.chart .lbl2{font-weight:600;font-size:12px}.chart .val{font-weight:700;font-size:12px}
.tabs{display:flex;gap:6px;margin:8px 0}.tab{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 14px;border-radius:999px;cursor:pointer;font:inherit}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}.pane{display:none}.pane.on{display:block}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 0 8px;font-size:13px}.key{display:inline-flex;align-items:center;gap:6px}
.key i{width:22px;height:3px;border-radius:2px;display:inline-block}.key i.dash{background:repeating-linear-gradient(90deg,currentColor 0 6px,transparent 6px 10px)!important}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:600}.n{text-align:right;font-variant-numeric:tabular-nums}
.pill{padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;background:var(--card);border:1px solid var(--line)}
.pill.covered{background:#0f9d5822;border-color:#0f9d58;color:#0f9d58}.pill.partial{background:#f4b40022;border-color:#c98a00;color:#c98a00}
.pill.planned{color:var(--mute)}.pill.cloud{color:#0072B2;border-color:#0072B2}
.callout{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);padding:12px 16px;border-radius:6px;margin:16px 0}
@media (max-width:560px){main{padding:18px 12px}h1{font-size:23px}}
"""
JS = "document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{const p=b.closest('section');" \
     "p.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));" \
     "p.querySelectorAll('.pane').forEach(x=>x.classList.toggle('on',x.id===b.dataset.pane))})"


def build(out: Path) -> Path:
    df, cap, cold = load()
    sections = []
    combos = cap[["model", "rows"]].drop_duplicates().sort_values(["model", "rows"]).itertuples(index=False)
    for model, rows in combos:
        rows = int(rows)
        if not len(best_per_stack(cap, model, rows, 50)):
            continue
        sections.append(f'<section><h2>Capacity — {esc(model)}, {rows} rows/request</h2>'
                        f'{headline_charts(cap, model, rows)}{journey(cap, model, rows)}'
                        f'{latency_section(df, cap, model, rows)}{resources(cap, cold, model, rows)}'
                        f'{table_html(cap, model, rows)}</section>')
    top = best_per_stack(cap, cap.model.iloc[0], int(cap.rows.iloc[0]), 50) if len(cap) else None
    callout = ""
    if top is not None and len(top):
        leader, last = top.iloc[0], top.iloc[-1]
        callout = (f'<div class="callout"><b>At a glance:</b> {esc(pretty(leader.stack))} leads at '
                   f'{leader.capacity_rps:.0f} req/s (p99 ≤ 50 ms); the slowest tested configuration, '
                   f'{esc(pretty(last.stack))}, reaches {last.capacity_rps:.0f}. Single-repeat search numbers on a laptop; '
                   f'treat differences of one ladder step (×1.5) as noise.</div>')
    envs = ", ".join(sorted(cap.env.unique()))
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Serving benchmark report</title><style>{CSS}</style></head><body><main>'
           f'<h1>Serving benchmark: FastAPI vs Ray Serve vs Triton</h1>'
           f'<p class="meta">{date.today().isoformat()} · envs: {esc(envs)} · open-loop k6, same model weights, '
           f'capacity = highest rate with p99 under the SLO and no errors</p>{callout}'
           f'{"".join(sections)}{coverage_html()}'
           f'<h2>Method and caveats</h2><p class="note">Serving containers are pinned to cores 0–5 and the load generator '
           f'to 6–7 (this CPU has hyperthread siblings). Capacities come from short search steps, one repeat, and land on a '
           f'×1.5 rate ladder, so neighbouring values are one step apart. Run-to-run noise on this laptop is about ±3 ms. '
           f'Results are a laptop-CPU snapshot, not cloud numbers.</p></main><script>{JS}</script></body></html>')
    out.write_text(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "report.html"))
    a = ap.parse_args()
    p = build(Path(a.out))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
