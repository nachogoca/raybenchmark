"""Disk space helpers: the run must never fill the disk."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def free_gb(path: str | Path = "/") -> float:
    return shutil.disk_usage(path).free / 1e9


def used_pct(path: str | Path = "/") -> float:
    u = shutil.disk_usage(path)
    return 100 * u.used / u.total


def dir_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def docker_usage() -> dict[str, str]:
    """{'Images': '33GB', 'Build Cache': '17GB', ...} from `docker system df`; {} if docker is unavailable."""
    try:
        out = subprocess.run(["docker", "system", "df", "--format", "{{.Type}}|{{.Size}}"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split("|", 1) for line in out.splitlines() if "|" in line)


def snapshot() -> dict:
    return {"free_gb": round(free_gb(), 1), "used_pct": round(used_pct(), 1),
            "results_mb": round(dir_mb(ROOT / "bench" / "results"), 1), "docker": docker_usage()}
