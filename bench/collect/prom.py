"""1 Hz Prometheus scraper for Ray Serve (:8080), Triton (:8002) and the FastAPI /metrics.

Counters and histogram sum/count pairs are reported as deltas over the measure window; gauges as
a per-second series. Only the metric families listed in WATCH are kept, summed over labels
(optionally split by one label), so a step row stays small.
"""
from __future__ import annotations

import contextlib
import re
import threading
import time
from collections import defaultdict

import httpx

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# family -> label to keep separate ("" = sum everything). Names match Ray 2.60 / Triton 26.x.
WATCH: dict[str, str] = {
    # Ray Serve: counters / histograms (sum, count) -> deltas
    "ray_serve_deployment_processing_latency_ms_sum": "deployment",
    "ray_serve_deployment_processing_latency_ms_count": "deployment",
    "ray_serve_http_request_latency_ms_sum": "",
    "ray_serve_http_request_latency_ms_count": "",
    "ray_serve_deployment_request_counter_total": "deployment",
    "ray_serve_deployment_error_counter_total": "deployment",
    "ray_serve_multiplexed_models_load_counter_total": "",
    "ray_serve_multiplexed_models_unload_counter_total": "",
    "ray_serve_multiplexed_get_model_requests_counter_total": "",
    "ray_serve_multiplexed_model_load_latency_ms_sum": "",
    "ray_serve_multiplexed_model_load_latency_ms_count": "",
    "ray_serve_batch_wait_time_ms_sum": "",
    "ray_serve_batch_wait_time_ms_count": "",
    "ray_serve_actual_batch_size_sum": "",
    "ray_serve_actual_batch_size_count": "",
    # Ray Serve: gauges -> series
    "ray_serve_replica_processing_queries": "deployment",
    "ray_serve_deployment_queued_queries": "deployment",
    "ray_serve_deployment_replica_healthy": "deployment",
    "ray_serve_num_multiplexed_models": "",
    "ray_serve_autoscaling_target_replicas": "deployment",
    # Triton: cumulative µs / counts -> deltas
    "nv_inference_request_success": "",
    "nv_inference_count": "",
    "nv_inference_exec_count": "",
    "nv_inference_queue_duration_us": "",
    "nv_inference_compute_input_duration_us": "",
    "nv_inference_compute_infer_duration_us": "",
    "nv_inference_compute_output_duration_us": "",
    "nv_inference_request_duration_us": "",
    # FastAPI middleware (bench.stacks.fastapi_ort.app)
    "raybench_handler_seconds_sum": "",
    "raybench_handler_seconds_count": "",
}
GAUGES = {k for k in WATCH if any(s in k for s in (
    "processing_queries", "queued_queries", "replica_healthy", "num_multiplexed_models",
    "target_replicas"))}


def parse(text: str) -> dict[str, dict[str, float]]:
    """-> {family: {label_value|'': value}} for WATCH families."""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = LINE.match(line)
        if not m or m.group(1) not in WATCH:
            continue
        name, labels, val = m.groups()
        key = WATCH[name]
        lab = dict(LABEL.findall(labels or ""))
        try:
            v = float(val)
        except ValueError:
            continue
        out[name][lab.get(key, "") if key else ""] += v
    return {k: dict(v) for k, v in out.items()}


class PromSampler:
    def __init__(self, urls: list[str], interval_s: float = 1.0):
        self.urls, self.interval = urls, interval_s
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> PromSampler:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=5)

    def _scrape(self) -> dict:
        merged: dict[str, dict[str, float]] = {}
        with httpx.Client(timeout=1.5) as c:
            for u in self.urls:
                with contextlib.suppress(httpx.HTTPError):
                    for k, v in parse(c.get(u).text).items():
                        d = merged.setdefault(k, {})
                        for lab, x in v.items():
                            d[lab] = d.get(lab, 0.0) + x
        return merged

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            with contextlib.suppress(Exception):
                self.samples.append((t, self._scrape()))
            self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    def summary(self, since: float | None = None) -> dict:
        """{family: delta (counters) | [per-second values] (gauges)} and derived means."""
        s = [(t, d) for t, d in self.samples if since is None or t >= since]
        if len(s) < 2:
            return {}
        first, last = s[0][1], s[-1][1]
        out: dict = {"deltas": {}, "gauges": {}, "t": [round(t, 2) for t, _ in s]}
        for fam in WATCH:
            if fam in GAUGES:
                out["gauges"][fam] = [sum(d.get(fam, {}).values()) for _, d in s]
            elif fam in last:
                out["deltas"][fam] = sum(last[fam].values()) - sum(first.get(fam, {}).values())
        d = out["deltas"]

        def mean(sum_k: str, cnt_k: str, scale: float = 1.0) -> float | None:
            n = d.get(cnt_k)
            return round(d[sum_k] / n * scale, 4) if n and sum_k in d else None

        out["derived"] = {
            "ray_proxy_ms": mean("ray_serve_http_request_latency_ms_sum", "ray_serve_http_request_latency_ms_count"),
            "ray_replica_ms": mean("ray_serve_deployment_processing_latency_ms_sum", "ray_serve_deployment_processing_latency_ms_count"),
            "triton_queue_ms": mean("nv_inference_queue_duration_us", "nv_inference_exec_count", 1e-3),
            "triton_compute_ms": mean("nv_inference_compute_infer_duration_us", "nv_inference_exec_count", 1e-3),
            "triton_input_ms": mean("nv_inference_compute_input_duration_us", "nv_inference_exec_count", 1e-3),
            "triton_output_ms": mean("nv_inference_compute_output_duration_us", "nv_inference_exec_count", 1e-3),
            "triton_mean_batch": (d["nv_inference_count"] / d["nv_inference_exec_count"]
                                  if d.get("nv_inference_exec_count") else None),
            "fastapi_handler_ms": mean("raybench_handler_seconds_sum", "raybench_handler_seconds_count", 1e3),
            "mux_loads": d.get("ray_serve_multiplexed_models_load_counter_total"),
            "mux_requests": d.get("ray_serve_multiplexed_get_model_requests_counter_total"),
        }
        return out
