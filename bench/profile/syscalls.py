"""Per-request syscall and context-switch counts for the serving processes, from /proc. Use it
where strace cannot attach (Yama ptrace_scope=1). Run inside the container while load is applied:
  docker exec -w /app <ctr> python -m bench.profile.syscalls --rps 87 --match ProxyActor ServeReplica

--rps is the request rate you are driving (probe prints it); counts are divided by rps*secs.
Many context switches per request means many thread/process hops, the typical cost of a framework.
"""
from __future__ import annotations

import argparse
import os
import time


def pids_matching(names: list[str]) -> dict[str, str]:
    out = {}
    for p in filter(str.isdigit, os.listdir("/proc")):
        try:
            c = open(f"/proc/{p}/cmdline").read().replace("\0", " ")
        except OSError:
            continue
        for n in names:
            if n in c.split(" ")[0] and p != str(os.getpid()):
                out[p] = n
    return out


def snap(pid: str) -> dict:
    d = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            io = dict(line.split(": ") for line in open(f"/proc/{pid}/task/{t}/io").read().strip().split("\n"))
            cs = sum(int(line.split()[1]) for line in open(f"/proc/{pid}/task/{t}/status") if "ctxt_switches" in line)
            d[t] = (int(io["syscr"]), int(io["syscw"]), cs, open(f"/proc/{pid}/task/{t}/comm").read().strip())
        except (OSError, KeyError):
            pass
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rps", type=float, required=True)
    ap.add_argument("--secs", type=float, default=6)
    ap.add_argument("--match", nargs="+", default=["ProxyActor", "ServeReplica", "ray::IDLE"])
    a = ap.parse_args()
    pm = pids_matching([f"ray::{m}" if not m.startswith("ray") else m for m in a.match] + ["haproxy"])
    s0 = {p: snap(p) for p in pm}
    time.sleep(a.secs)
    n = a.rps * a.secs
    for p, label in pm.items():
        s1 = snap(p)
        tot = [0, 0, 0]
        for t in s1:
            if t in s0[p]:
                for i in range(3):
                    tot[i] += s1[t][i] - s0[p][t][i]
        print(f"{label:<22} pid {p:>5}: per request reads={tot[0]/n:5.1f} writes={tot[1]/n:5.1f} ctx-switches={tot[2]/n:6.1f}")


if __name__ == "__main__":
    main()
