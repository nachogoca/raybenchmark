"""Host guards: CPU frequency / temperature during a step, and the `preflight` checklist.

A step is flagged invalid when the CPU throttled, the load generator saturated its cores, or a
container restarted. Reading sysfs never raises; missing files just give None.
"""
from __future__ import annotations

import contextlib
import glob
import shutil
import subprocess
from pathlib import Path

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
    """(check, ok, hint). Checks that make results noisy, not just failing runs."""
    res: list[tuple[str, bool, str]] = []
    gov = _read(CPU0 / "scaling_governor")
    res.append((f"cpu governor = {gov}", gov in (None, "performance"),
                "run: sudo cpupower frequency-set -g performance"))
    nt = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    res.append((f"turbo {'off' if nt == '1' else 'on'}", nt in (None, "1"),
                "optional, steadier clocks: echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"))
    mem = _read("/proc/meminfo") or ""
    avail = next((int(line.split()[1]) / 1e6 for line in mem.splitlines()
                  if line.startswith("MemAvailable")), None)
    res.append((f"free RAM {avail:.1f} GB" if avail else "free RAM unknown",
                avail is None or avail >= min_free_gb, "close browsers / other containers"))
    for tool in ("docker", "k6", "taskset"):
        res.append((f"{tool} installed", shutil.which(tool) is not None, f"install {tool}"))
    if shutil.which("docker"):
        ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
        others = [n for n in ps.stdout.split() if not n.startswith("raybench")]
        res.append((f"{len(others)} unrelated containers running", not others,
                    "docker stop " + " ".join(others)))
    return res
