"""macOS / Linux portability: docker-stats resource reader, Docker-VM CPU sizing, OS-aware preflight, arm64 skips."""
import argparse
import subprocess
from pathlib import Path

from bench.collect import host, sampler
from bench.load import scenarios as sc
from bench.load import sweep as sw
from bench.load.backends import StackCfg

ROOT = Path(__file__).resolve().parents[1]


def test_parse_size_units():
    assert sampler.parse_size("12.5MiB") == 12.5 * 1024**2
    assert sampler.parse_size("1.5GiB") == 1.5 * 1024**3
    assert sampler.parse_size("800kB") == 0.0 or sampler.parse_size("800KB") == 800e3
    assert sampler.parse_size("garbage") == 0.0


def test_docker_stats_reader_sums_containers(monkeypatch):
    out = "45.50%|120MiB / 7.7GiB\n3.00%|1GiB / 7.7GiB\n"
    monkeypatch.setattr(sampler.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    d = sampler.docker_stats_reader(["a", "b"])()
    assert abs(d["cpu_cores"] - 0.485) < 1e-9 and d["cpu_usec"] is None
    assert d["mem_bytes"] == 120 * 1024**2 + 1024**3 and d["rss_bytes"] == d["mem_bytes"]


def test_docker_reader_falls_back_without_cgroups():
    r = sampler.docker_reader(["no-such-container"])          # no cgroup dir on this host for that id
    assert "docker_stats_reader" in r.__qualname__
    s = sampler.Sampler(lambda: {})
    s.samples = [(0.0, {"cpu_usec": None, "cpu_cores": 1.0, "mem_bytes": 1 << 20, "rss_bytes": 1 << 20}),
                 (1.0, {"cpu_usec": None, "cpu_cores": 3.0, "mem_bytes": 2 << 20, "rss_bytes": 2 << 20})]
    out = s.summary()
    assert out["cpu_cores_mean"] == 3.0 and out["cpu_by_component"] == {}        # no component split on macOS


def test_auto_cpu_env_leaves_two_cpus(monkeypatch):
    base = {"auto_cpus": True, "tuned": True}
    for ncpu, want in ((8, 6), (12, 10), (4, 2), (2, 1), (1, 1)):
        monkeypatch.setattr(host, "docker_vm", lambda n=ncpu: (n, 16.0))
        e = host.auto_cpu_env(base)
        assert e["total_cpus"] == want and e["serving_cpuset"] == f"0-{want - 1}" and e["loadgen_cpuset"] == "" and e["tuned"]


def test_loadgen_cores_pinned_or_host_remainder(monkeypatch):
    assert host.loadgen_cores("6-7") == 2
    monkeypatch.setattr(host, "docker_vm", lambda: (6, 16.0))
    monkeypatch.setattr(host.os, "cpu_count", lambda: 10)
    assert host.loadgen_cores("") == 4


def test_preflight_is_os_aware(monkeypatch):
    monkeypatch.setattr(host, "is_macos", lambda: True)
    monkeypatch.setattr(host, "is_linux", lambda: False)
    monkeypatch.setattr(host, "is_arm", lambda: True)
    monkeypatch.setattr(host, "docker_vm", lambda: (4, 8.0))
    monkeypatch.setattr(host.shutil, "which", lambda t: "/usr/bin/" + t)
    monkeypatch.setattr(host.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    checks = {c: (ok, hint) for c, ok, hint in host.preflight()}
    assert not any("governor" in c or "taskset" in c or "turbo" in c for c in checks)     # Linux-only checks absent
    assert checks["Docker VM has 4 CPUs"][0] is False and "Settings" in checks["Docker VM has 4 CPUs"][1]
    assert any("arm64" in c for c in checks)


def test_x86_only_stacks_skipped_on_arm(monkeypatch):
    monkeypatch.setattr(sc, "is_arm", lambda: True)
    monkeypatch.delenv("RAYBENCH_EMULATE", raising=False)
    assert sc.unsupported_here(StackCfg("triton", "deepfm", "small", False, 1, 1, 6))
    assert sc.unsupported_here(StackCfg("ray_triton_inproc", "deepfm", "small", False, 1, 1, 6))
    assert sc.unsupported_here(StackCfg("ray_ort", "deepfm", "small", False, 1, 1, 6)) is None
    monkeypatch.setenv("RAYBENCH_EMULATE", "1")
    assert sc.unsupported_here(StackCfg("triton", "deepfm", "small", False, 1, 1, 6)) is None
    monkeypatch.setattr(sc, "is_arm", lambda: False)
    monkeypatch.delenv("RAYBENCH_EMULATE")
    assert sc.unsupported_here(StackCfg("triton", "deepfm", "small", False, 1, 1, 6)) is None


def test_mac_env_sizes_cpus_and_uses_compose(tmp_path, monkeypatch):
    monkeypatch.setattr(sw, "RESULTS", tmp_path)
    monkeypatch.setattr(host, "docker_vm", lambda: (10, 24.0))
    made = {}
    monkeypatch.setattr(sw, "ComposeBackend", lambda env_cfg, log: made.setdefault("cfg", env_cfg))
    s = sw.Sweep(argparse.Namespace(profile="smoke", env="mac", scenarios=None, dry_run=True, no_resume=False, tag="x", registry=""))
    assert s.env_cfg["total_cpus"] == 8 and made["cfg"]["serving_cpuset"] == "0-7" and made["cfg"]["loadgen_cpuset"] == ""
    assert s.out.name == "mac-smoke.jsonl"


def test_justfile_is_os_aware():
    j = (ROOT / "justfile").read_text()
    assert 'os() == "macos"' in j and "brew install k6" in j and "--env {{env}}" in j
    setup = j[j.index("setup:"):j.index("# Install profiling tools")]
    assert "apt-get" not in setup                                   # no apt on the setup path
    prof = j[j.index("setup-profiling:"):j.index("# Bring a stack up for profiling")]
    assert 'Linux only' in prof and prof.index("Linux only") < prof.index("apt-get")   # guarded before any apt call
