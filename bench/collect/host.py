"""Host guards: CPU frequency / temperature during a step, and the `preflight` checklist.

A step is flagged invalid when the CPU throttled, the load generator saturated its cores, or a
container restarted. Reading sysfs never raises; missing files just give None.
"""
from __future__ import annotations

import contextlib
import glob
import os
import platform
import shutil
import subprocess
from pathlib import Path

import psutil

CPU0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")


def _read(p: str | Path) -> str | None:
    with contextlib.suppress(OSError):
        return Path(p).read_text().strip()
    return None


def cpu_freq_mhz(cpus: list[int]) -> float | None:
    vals = []
    for c in cpus:
        v = _read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
        if v and v.isdigit():
            vals.append(int(v) / 1000)
    return sum(vals) / len(vals) if vals else None


def package_temp_c() -> float | None:
    best = None
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        if (_read(f"{zone}/type") or "").startswith(("x86_pkg_temp", "TCPU", "acpitz")):
            v = _read(f"{zone}/temp")
            if v and v.lstrip("-").isdigit():
                t = int(v) / 1000
                best = t if best is None else max(best, t)
    return best


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_arm() -> bool:
    return platform.machine().lower() in ("arm64", "aarch64")


def docker_vm() -> tuple[int | None, float | None]:
    """(CPUs, GB of RAM) of the machine Docker runs containers on (the Docker Desktop VM on macOS)."""
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.NCPU}}|{{.MemTotal}}"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
        n, mem = out.split("|")
        return int(n), int(mem) / 1e9
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def auto_cpu_env(env_cfg: dict) -> dict:
    """Size the serving CPUs from the Docker VM (macOS): all but 2 CPUs serve; k6 runs on the host, unpinned."""
    n, _ = docker_vm()
    n = n or os.cpu_count() or 4
    serving = max(1, n - 2) if n > 3 else max(1, n - 1)
    return {**env_cfg, "total_cpus": serving, "serving_cpuset": f"0-{serving - 1}", "loadgen_cpuset": ""}


def loadgen_cores(loadgen_cpuset: str) -> int:
    """Cores available to k6: its pinned set, or (unpinned, macOS) the host CPUs the Docker VM does not use."""
    pinned = parse_cpuset(loadgen_cpuset)
    if pinned:
        return len(pinned)
    n, _ = docker_vm()
    return max(1, (os.cpu_count() or 2) - (n or 0))


def parse_cpuset(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class HostWatch:
    """Samples frequency/temperature next to the resource sampler; summary() -> flags."""

    def __init__(self, serving_cpuset: str):
        self.cpus = parse_cpuset(serving_cpuset)
        self.freq: list[float] = []
        self.temp: list[float] = []

    def sample(self) -> None:
        f, t = cpu_freq_mhz(self.cpus), package_temp_c()
        if f is not None:
            self.freq.append(f)
        if t is not None:
            self.temp.append(t)

    def summary(self) -> dict:
        max_f = _read(CPU0 / "cpuinfo_max_freq")
        out = {
            "freq_mean_mhz": round(sum(self.freq) / len(self.freq), 0) if self.freq else None,
            "freq_min_mhz": round(min(self.freq), 0) if self.freq else None,
            "temp_max_c": round(max(self.temp), 1) if self.temp else None,
        }
        out["throttled"] = bool(
            (out["temp_max_c"] or 0) >= 95
            or (max_f and max_f.isdigit() and out["freq_min_mhz"]
                and out["freq_min_mhz"] < 0.5 * int(max_f) / 1000 and _read(CPU0 / "scaling_governor") != "powersave"))
        return out


def preflight(min_free_gb: float = 8.0) -> list[tuple[str, bool, str]]:
    """(check, ok, hint). Checks that make results noisy, not just failing runs. OS-aware."""
    res: list[tuple[str, bool, str]] = []
    mac = is_macos()
    if is_linux():
        gov = _read(CPU0 / "scaling_governor")
        res.append((f"cpu governor = {gov}", gov in (None, "performance"),
                    "run: sudo cpupower frequency-set -g performance"))
        nt = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        res.append((f"turbo {'off' if nt == '1' else 'on'}", nt in (None, "1"),
                    "optional, steadier clocks: echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"))
    avail = psutil.virtual_memory().available / 1e9
    res.append((f"free RAM {avail:.1f} GB", avail >= min_free_gb, "close browsers / other containers"))
    tools = ["docker", "k6"] + (["taskset"] if is_linux() else [])
    for tool in tools:
        hint = {"k6": "brew install k6" if mac else "install k6 (see `just setup`)",
                "docker": "install and start Docker Desktop or colima" if mac else "install Docker Engine"}.get(tool, f"install {tool}")
        res.append((f"{tool} installed", shutil.which(tool) is not None, hint))
    if shutil.which("docker"):
        ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
        others = [n for n in ps.stdout.split() if not n.startswith("raybench")]
        res.append((f"{len(others)} unrelated containers running", not others, "docker stop " + " ".join(others)))
    if mac and shutil.which("docker"):
        n, mem = docker_vm()
        if n is not None:
            res.append((f"Docker VM has {n} CPUs", n >= 8, "Docker Desktop > Settings > Resources: give it 8+ CPUs"))
            res.append((f"Docker VM has {mem:.0f} GB RAM", (mem or 0) >= 12, "Docker Desktop > Settings > Resources: give it 12+ GB"))
    if is_arm():
        res.append(("arm64 host: Triton images are x86-64 only", False,
                    "Triton stacks are skipped (emulation would give meaningless numbers); set RAYBENCH_EMULATE=1 to force"))
    return res
