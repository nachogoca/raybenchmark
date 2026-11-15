"""Is the benchmark really running? One screen: process, progress, last log line, disk, containers.

  uv run python -m bench.monitor              # print once
  uv run python -m bench.monitor --watch 30   # refresh every 30 s, append to bench/results/runs/monitor.log
  uv run python -m bench.monitor --alerts     # print one line only when something is wrong (for a notifier)

Alerts: process gone before the last test finished, run log silent too long, disk low, container
restarting or unhealthy, steps marked invalid, a stopped-by-guard row.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

from bench.collect import disk

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "bench" / "results" / "runs"
CFG = yaml.safe_load((ROOT / "config" / "matrix.yaml").read_text())
GUARDS = CFG.get("guards", {})
CONFIG_LINE = re.compile(r"\] \[([A-Za-z0-9_]+)\] (\S+) variant=(\S*) replicas=(\d+)")
STEP_LINE = re.compile(r"\]\s+(grid|search|closed|overload|burst|failure|fanout|family_fixed|refine)\s")
HEADER = re.compile(r"run (\S+): profile=(\S+) env=(\S+) scenarios=\[(.*?)\]")
DONE = re.compile(r"== (\S+) done in ([\d.]+) min")


def find_sweep() -> psutil.Process | None:
    for p in psutil.process_iter(["pid", "cmdline"]):
        c = " ".join(p.info["cmdline"] or [])
        if "-m bench.load.sweep" in c and "monitor" not in c and "pgrep" not in c and not c.startswith(("sh", "/usr/bin/zsh", "zsh")):
            return p
    return None


def latest_log() -> Path | None:
    logs = sorted(RUNS.glob("*.log"), key=lambda f: f.stat().st_mtime)
    return logs[-1] if logs else None


def parse_log(text: str) -> dict:
    lines = text.splitlines()
    out: dict = {"scenarios": [], "done": [], "steps": 0, "config": None, "invalid": 0, "errors": 0, "stopped": False,
                 "finished": False, "last": lines[-1] if lines else ""}
    for line in lines:
        if (m := HEADER.search(line)):
            out["run"], out["profile"] = m.group(1), m.group(2)
            out["scenarios"] = [x.strip(" '") for x in m.group(4).split(",")]
        if (m := CONFIG_LINE.search(line)):
            out["config"] = {"test": m.group(1), "stack": m.group(2), "variant": m.group(3)}
        if (m := DONE.search(line)):
            out["done"].append(m.group(1))
        if STEP_LINE.search(line):
            out["steps"] += 1
        out["invalid"] += "INVALID" in line
        out["errors"] += " ERROR " in line
        out["stopped"] |= "DISK LOW" in line
    out["finished"] = bool(out["scenarios"]) and out["done"][-1:] == out["scenarios"][-1:]
    return out


def containers() -> list[tuple[str, str]]:
    r = subprocess.run(["docker", "ps", "-a", "--filter", "name=raybench", "--format", "{{.Names}}|{{.Status}}"],
                       capture_output=True, text=True)
    return [tuple(x.split("|", 1)) for x in r.stdout.splitlines() if "|" in x]


def assess() -> tuple[list[str], list[str]]:
    """-> (report lines, alerts)."""
    lines, alerts = [], []
    log, proc = latest_log(), find_sweep()
    now = time.time()
    snap = disk.snapshot()
    if log is None:
        return ["No run log found."], ["no run has started"]
    info = parse_log(log.read_text())
    age = now - log.stat().st_mtime
    state = "FINISHED" if info["finished"] else ("RUNNING" if proc else "NOT RUNNING")
    elapsed = (now - proc.create_time()) / 60 if proc else None
    lines.append(f"{datetime.now():%H:%M:%S}  {state}" + (f"  pid {proc.pid}  up {elapsed:.0f} min" if proc else ""))
    lines.append(f"tests done: {len(info['done'])}/{len(info['scenarios'])}  ({', '.join(info['done'][-3:]) or '-'})   steps run: {info['steps']}")
    if info["config"]:
        c = info["config"]
        lines.append(f"now: test {c['test']} · stack {c['stack']} · variant {c['variant'] or '-'}")
    lines.append(f"last log line ({age:.0f} s ago): {info['last'][:110]}")
    d = snap["docker"]
    lines.append(f"disk: {snap['free_gb']} GB free ({snap['used_pct']} % used) · results {snap['results_mb']} MB · "
                 f"docker images {d.get('Images', '?')}, build cache {d.get('Build Cache', '?')}")
    cs = containers()
    lines.append("containers: " + (", ".join(f"{n} [{s.split(' (')[0]}]" for n, s in cs) or "none"))
    if not info["finished"] and not proc:
        alerts.append("the run process is gone before the last test finished")
    if proc and age > 60 * float(GUARDS.get("stale_minutes", 8)):
        alerts.append(f"run log silent for {age / 60:.0f} min (the run may be stuck)")
    if snap["free_gb"] < float(GUARDS.get("min_free_gb", 15)) + 5:
        alerts.append(f"disk low: {snap['free_gb']} GB free")
    if info["stopped"]:
        alerts.append("the run stopped itself because the disk was low")
    for n, s in cs:
        if "Restarting" in s or "unhealthy" in s or ("Exited" in s and proc and not info["finished"] and "triton_repo" not in n):
            alerts.append(f"container {n}: {s}")
    if info["invalid"]:
        alerts.append(f"{info['invalid']} invalid step(s) so far (see the log)")
    if info["errors"]:
        alerts.append(f"{info['errors']} config error(s) in the log")
    lines.append("ALERTS: " + ("; ".join(alerts) if alerts else "none"))
    return lines, alerts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    ap.add_argument("--alerts", action="store_true", help="print one line only when the alert set changes")
    a = ap.parse_args()
    logf = RUNS / "monitor.log"
    seen: set[str] = set()
    while True:
        lines, alerts = assess()
        if a.alerts:
            new = set(alerts) - seen
            for x in sorted(new):
                print(f"{datetime.now():%H:%M:%S} ALERT: {x}", flush=True)
            seen = set(alerts)
        else:
            print("\n".join(lines) + "\n", flush=True)
        if a.watch:
            RUNS.mkdir(parents=True, exist_ok=True)
            with logf.open("a") as f:
                f.write("\n".join(lines[:1] + lines[-3:]) + "\n")
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
