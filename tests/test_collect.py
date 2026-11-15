"""Collectors: host guards, Prometheus scraper, per-component CPU, sampler summary."""
import time

from bench.collect import host, prom, sampler


def test_parse_cpuset():
    assert host.parse_cpuset("0-2,5") == [0, 1, 2, 5]
    assert host.parse_cpuset("3") == [3]


def test_hostwatch_throttle_rule(monkeypatch):
    """Throttle events at a healthy clock do not invalidate a step; a collapsed clock or a critical temperature does."""
    monkeypatch.setattr(host, "_read", lambda p: "3500000" if "cpuinfo_max_freq" in str(p) else None)
    counts = iter([100, 160])                       # 60 events during the step
    monkeypatch.setattr(host, "throttle_total", lambda: next(counts))
    monkeypatch.setattr(host, "cpu_freq_mhz", lambda cpus: 3000.0)      # 86 % of max
    monkeypatch.setattr(host, "package_temp_c", lambda: 94.0)
    w = host.HostWatch("0-5")
    w.sample()
    s = w.summary()
    assert s["throttle_events"] == 60 and not s["throttled"] and s["freq_mean_mhz"] == 3000.0
    monkeypatch.setattr(host, "package_temp_c", lambda: 99.0)
    monkeypatch.setattr(host, "throttle_total", lambda: None)
    w2 = host.HostWatch("0-5")
    w2.sample()
    assert w2.summary()["throttled"]                                    # critical temperature
    counts2 = iter([0, 5])
    monkeypatch.setattr(host, "throttle_total", lambda: next(counts2))
    monkeypatch.setattr(host, "cpu_freq_mhz", lambda cpus: 1800.0)      # 51 % of max
    monkeypatch.setattr(host, "package_temp_c", lambda: 90.0)
    w3 = host.HostWatch("0-5")
    w3.sample()
    assert w3.summary()["throttled"]                                    # clock collapsed, with events


def test_preflight_reports_governor(monkeypatch):
    monkeypatch.setattr(host, "_read", lambda p: "powersave" if "governor" in str(p) else None)
    checks = {c.split(" ")[0] + c.split(" ")[1]: ok for c, ok, _ in host.preflight()}
    assert checks["cpugovernor"] is False


def test_prom_parse_keeps_watched_families_and_splits_labels():
    text = ('# HELP x\nray_serve_deployment_processing_latency_ms_sum{deployment="A",replica="r1"} 10\n'
            'ray_serve_deployment_processing_latency_ms_sum{deployment="A",replica="r2"} 5\n'
            'ray_serve_deployment_processing_latency_ms_sum{deployment="B"} 2\nunrelated_metric 9\nnv_inference_count 7\n')
    out = prom.parse(text)
    assert out["ray_serve_deployment_processing_latency_ms_sum"] == {"A": 15.0, "B": 2.0}
    assert out["nv_inference_count"] == {"": 7.0} and "unrelated_metric" not in out


def test_prom_summary_deltas_and_gauges():
    p = prom.PromSampler([])
    p.samples = [
        (0.0, {"ray_serve_http_request_latency_ms_sum": {"": 100.0}, "ray_serve_http_request_latency_ms_count": {"": 10.0},
               "ray_serve_deployment_replica_healthy": {"A": 1.0},
               "nv_inference_count": {"": 0.0}, "nv_inference_exec_count": {"": 0.0}}),
        (1.0, {"ray_serve_http_request_latency_ms_sum": {"": 160.0}, "ray_serve_http_request_latency_ms_count": {"": 20.0},
               "ray_serve_deployment_replica_healthy": {"A": 3.0},
               "nv_inference_count": {"": 64.0}, "nv_inference_exec_count": {"": 2.0}}),
    ]
    s = p.summary()
    assert s["derived"]["ray_proxy_ms"] == 6.0                  # (160-100)/(20-10)
    assert s["derived"]["triton_mean_batch"] == 32.0
    assert s["gauges"]["ray_serve_deployment_replica_healthy"] == [1.0, 3.0]
    assert s["t"] == [0.0, 1.0]
    assert prom.PromSampler([]).summary() == {}


def test_classify_processes():
    c = sampler.classify
    assert c("ray::ServeReplica:default:MuxModel") == "ray_replica"
    assert c("ray::ProxyActor") == "ray_proxy"
    assert c("/x/ray_haproxy/bin/haproxy -f cfg") == "haproxy"
    assert c("/opt/tritonserver/bin/tritonserver --model-repository=/m") == "triton"
    assert c("python -m uvicorn app:app") == "uvicorn"
    assert c("/bin/sleep 1") == "other"


def test_proc_cpu_reads_self():
    import os
    r = sampler._proc_cpu_usec(os.getpid())
    assert r is not None and r[1] >= 0


def test_sampler_summary_component_cores():
    s = sampler.Sampler(lambda: {})
    s.samples = [(0.0, {"cpu_usec": 0, "mem_bytes": 1 << 20, "rss_bytes": 1 << 20, "comp_usec": {"ray_replica": 0}}),
                 (2.0, {"cpu_usec": 4_000_000, "mem_bytes": 2 << 20, "rss_bytes": 2 << 20, "comp_usec": {"ray_replica": 3_000_000, "haproxy": 1_000_000}})]
    out = s.summary()
    assert out["cpu_cores_mean"] == 2.0
    assert out["cpu_by_component"] == {"ray_replica": 1.5, "haproxy": 0.5}
    assert out["rss_peak_mb"] == 2.0


def test_sampler_thread_collects():
    s = sampler.Sampler(lambda: {"cpu_usec": 1, "mem_bytes": 1, "rss_bytes": 1}, interval_s=0.05)
    with s:
        time.sleep(0.25)
    assert len(s.samples) >= 3
