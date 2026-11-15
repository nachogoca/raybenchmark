"""Per-thread CPU of the serving processes, from /proc. Runs inside the container (the image
has no procps/perf):  docker exec -w /app <ctr> python -m bench.profile.threads --secs 10

Shows which threads burn cores. Ray renames its processes, so unnamed native threads (e.g. ONNX
Runtime workers that busy-wait) inherit names like 'ray::IDLE'; judge them by their CPU.
"""
from __future__ import annotations

import argparse
import os
import time

MATCH = ("ray::", "raylet", "gcs_server", "haproxy", "python")


def procs() -> dict[str, str]:
    out = {}
    for p in filter(str.isdigit, os.listdir("/proc")):
        try:
            c = open(f"/proc/{p}/cmdline").read().replace("\0", " ")
        except OSError:
            continue
        if p != str(os.getpid()) and any(m in c.split(" ")[0] or c.startswith(m) for m in MATCH):
            out[p] = c[:40]
    return out


def snap(pids) -> dict:
    d = {}
    for p in pids:
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                f = open(f"/proc/{p}/task/{t}/stat").read()
                a = f[f.rfind(")") + 2:].split()
                d[(p, t)] = (int(a[11]), int(a[12]), open(f"/proc/{p}/task/{t}/comm").read().strip())
        except OSError:
            pass
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=10)
    ap.add_argument("--top", type=int, default=18)
    a = ap.parse_args()
    pr = procs()
    s0 = snap(pr)
    time.sleep(a.secs)
    s1 = snap(pr)
    hz = os.sysconf("SC_CLK_TCK")
    rows = sorted(((((s1[k][0] - s0[k][0]) + (s1[k][1] - s0[k][1])) / hz / a.secs,
                    (s1[k][0] - s0[k][0]) / hz / a.secs, (s1[k][1] - s0[k][1]) / hz / a.secs,
                    pr[k[0]], s1[k][2]) for k in s1 if k in s0), reverse=True)
    print(f"{'cores':>6} {'user':>5} {'sys':>5}  process / thread")
    for c, u, s, pn, tn in rows[:a.top]:
        if c > 0.01:
            print(f"{c:6.2f} {u:5.2f} {s:5.2f}  {pn:<40} {tn}")


if __name__ == "__main__":
    main()
