"""1 Hz resource sampler with pluggable readers (docker cgroup v2 locally, `kubectl top` on k8s).

A reader returns {"cpu_usec": cumulative CPU microseconds | None, "cpu_cores": instantaneous
cores | None, "mem_bytes": current memory, "rss_bytes": anon memory, "gpu_util": %, "vram_mb"}.
Summary converts to mean/peak cores and MB. Missing values are skipped, never raised.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

Reader = Callable[[], dict]

_CGROUP_CANDIDATES = (
    "/sys/fs/cgroup/system.slice/docker-{id}.scope",
    "/sys/fs/cgroup/docker/{id}",
    "/sys/fs/cgroup/system.slice/docker-{id}.scope/",
)


def _read_int(path: Path) -> int | None:
    with contextlib.suppress(OSError, ValueError):
        return int(path.read_text().split()[0])
    return None


_TICKS = os.sysconf("SC_CLK_TCK")
# first match wins; matched against the process cmdline (Ray sets ray::<Actor> process titles)
_COMPONENTS = (
    ("haproxy", "haproxy"), ("ray::ProxyActor", "ray_proxy"), ("ray::ServeReplica", "ray_replica"),
    ("ray::ServeController", "ray_controller"), ("raylet", "ray_raylet"), ("gcs_server", "ray_gcs"),
    ("tritonserver", "triton"), ("uvicorn", "uvicorn"), ("ray::", "ray_other"),
    ("ray/dashboard", "ray_other"), ("ray/_private", "ray_other"), ("python", "python_other"),
)


def classify(cmdline: str) -> str:
    for needle, comp in _COMPONENTS:
        if needle in cmdline:
            return comp
    return "other"


def _proc_cpu_usec(pid: int) -> tuple[str, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return None
    f = stat[stat.rindex(")") + 2:].split()      # fields after "(comm)"; utime=idx 11, stime=idx 12
    usec = (int(f[11]) + int(f[12])) * 1_000_000 // _TICKS
    return classify(cmd or stat[stat.index("(") + 1:stat.rindex(")")]), usec


def docker_reader(container_ids: list[str]) -> Reader:
    dirs: list[Path] = []
    for cid in container_ids:
        for cand in _CGROUP_CANDIDATES:
            p = Path(cand.format(id=cid))
            if (p / "cpu.stat").exists():
                dirs.append(p)
                break

    def read() -> dict:
        cpu = mem = rss = 0
        comp: dict[str, int] = {}
        for d in dirs:
            with contextlib.suppress(OSError):
                for pid in (d / "cgroup.procs").read_text().split():
                    r = _proc_cpu_usec(int(pid))
                    if r:
                        comp[r[0]] = comp.get(r[0], 0) + r[1]
            with contextlib.suppress(OSError):
                for line in (d / "cpu.stat").read_text().splitlines():
                    if line.startswith("usage_usec"):
                        cpu += int(line.split()[1])
            mem += _read_int(d / "memory.current") or 0
            with contextlib.suppress(OSError):
                for line in (d / "memory.stat").read_text().splitlines():
                    if line.startswith("anon "):
                        rss += int(line.split()[1])
        return {"cpu_usec": cpu, "cpu_cores": None, "mem_bytes": mem, "rss_bytes": rss,
                "gpu_util": None, "vram_mb": None, "comp_usec": comp}

    return read


def nvidia_smi_reader(inner: Reader) -> Reader:
    """Wrap a reader with host nvidia-smi GPU util / VRAM (laptop or a GPU VM)."""
    def read() -> dict:
        d = inner()
        with contextlib.suppress(Exception):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
            util, vram = out.strip().split("\n")[0].split(",")
            d["gpu_util"], d["vram_mb"] = float(util), float(vram)
        return d
    return read


_QTY = re.compile(r"^(\d+(?:\.\d+)?)([a-zA-Z]*)$")
_MEM_UNITS = {"": 1, "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "k": 1000, "M": 1e6, "G": 1e9}


def parse_cpu(s: str) -> float:
    m = _QTY.match(s.strip())
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2)
    return v / 1000 if u == "m" else v


def parse_mem(s: str) -> float:
    m = _QTY.match(s.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _MEM_UNITS.get(m.group(2), 1)


def kubectl_top_reader(namespace: str, selector: str) -> Reader:
    def read() -> dict:
        cores = mem = 0.0
        with contextlib.suppress(Exception):
            out = subprocess.run(
                ["kubectl", "top", "pod", "-n", namespace, "-l", selector, "--no-headers"],
                capture_output=True, text=True, timeout=10, check=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    cores += parse_cpu(parts[1])
                    mem += parse_mem(parts[2])
        return {"cpu_usec": None, "cpu_cores": cores, "mem_bytes": mem, "rss_bytes": mem,
                "gpu_util": None, "vram_mb": None}
    return read


class Sampler:
    def __init__(self, reader: Reader, interval_s: float = 1.0):
        self.reader, self.interval = reader, interval_s
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Sampler:
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.time()
            with contextlib.suppress(Exception):
                self.samples.append((t, self.reader()))
            self._stop.wait(max(0.0, self.interval - (time.time() - t)))

    def stop(self) -> None:
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=5)

    def summary(self, since: float | None = None) -> dict:
        s = [(t, d) for t, d in self.samples if since is None or t >= since]
        cores: list[float] = []
        for (t0, d0), (t1, d1) in itertools.pairwise(s):
            if d0.get("cpu_usec") is not None and d1.get("cpu_usec") is not None and t1 > t0:
                cores.append((d1["cpu_usec"] - d0["cpu_usec"]) / 1e6 / (t1 - t0))
            elif d1.get("cpu_cores") is not None:
                cores.append(d1["cpu_cores"])
        comp_cores: dict[str, float] = {}
        if len(s) >= 2 and s[0][1].get("comp_usec") is not None:
            dt = s[-1][0] - s[0][0]
            a, b = s[0][1]["comp_usec"], s[-1][1]["comp_usec"]
            comp_cores = {k: round(max(0, b.get(k, 0) - a.get(k, 0)) / 1e6 / dt, 3)
                          for k in set(a) | set(b)} if dt > 0 else {}
        mem = [d["mem_bytes"] for _, d in s if d.get("mem_bytes")]
        rss = [d["rss_bytes"] for _, d in s if d.get("rss_bytes")]
        gpu = [d["gpu_util"] for _, d in s if d.get("gpu_util") is not None]
        vram = [d["vram_mb"] for _, d in s if d.get("vram_mb") is not None]
        mb = 1024**2
        return {
            "samples": len(s),
            "cpu_by_component": comp_cores,
            "cpu_core_seconds": round(sum(cores), 2) if cores else None,   # 1 Hz samples ~ seconds
            "cpu_cores_mean": round(sum(cores) / len(cores), 3) if cores else None,
            "cpu_cores_peak": round(max(cores), 3) if cores else None,
            "mem_mean_mb": round(sum(mem) / len(mem) / mb, 1) if mem else None,
            "mem_peak_mb": round(max(mem) / mb, 1) if mem else None,
            "rss_mean_mb": round(sum(rss) / len(rss) / mb, 1) if rss else None,
            "rss_peak_mb": round(max(rss) / mb, 1) if rss else None,
            "gpu_util_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "vram_peak_mb": round(max(vram), 1) if vram else None,
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.samples))
