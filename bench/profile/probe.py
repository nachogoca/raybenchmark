"""Closed-loop latency probe: N client threads, each sends its next request when the last one
returns. Measures per-request cost (not capacity; open-loop k6 does that). Pin it away from the
serving cores:  taskset -c 6,7 uv run python -m bench.profile.probe --conc 1 --dur 12

c=1 isolates fixed per-request overhead; c=4 shows how it scales. Payload is bench/load/payloads.
"""
from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from pathlib import Path

PAYLOADS = Path(__file__).resolve().parents[1] / "load" / "payloads"


def run(port: int, path: str, conc: int, dur: float, rows: int) -> dict:
    bodies = [json.dumps(b).encode() for b in json.loads((PAYLOADS / f"rows{rows}.json").read_text())]
    lat: list[float] = []
    errs = [0]
    stop = time.time() + dur

    def worker(i: int) -> None:
        c, k = http.client.HTTPConnection("127.0.0.1", port, timeout=30), i
        while time.time() < stop:
            t = time.perf_counter()
            try:
                c.request("POST", path, bodies[k % len(bodies)], {"Content-Type": "application/json"})
                r = c.getresponse()
                r.read()
                errs[0] += r.status != 200
            except Exception:
                errs[0] += 1
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            lat.append((time.perf_counter() - t) * 1000)
            k += conc

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lat = sorted(lat[len(lat) // 10:])  # drop warm-up
    n = len(lat)
    return {"conc": conc, "n": n, "rps": n / (dur * 0.9), "p50": lat[n // 2], "p90": lat[int(n * .9)],
            "p99": lat[int(n * .99)], "err": errs[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="deepfm_small")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--dur", type=float, default=12)
    ap.add_argument("--rows", type=int, default=200, choices=[1, 50, 200, 500])
    a = ap.parse_args()
    r = run(a.port, f"/v2/models/{a.model}/infer", a.conc, a.dur, a.rows)
    print(f"c={r['conc']} rows={a.rows} n={r['n']} rps={r['rps']:.1f} p50={r['p50']:.1f} "
          f"p90={r['p90']:.1f} p99={r['p99']:.1f} err={r['err']}")


if __name__ == "__main__":
    main()
