# Results and learnings (laptop, 2026-11-15)

Environment: Intel i7-6700HQ (4 cores / 8 threads), 12 GB RAM, Ubuntu, Docker 29.9. Serving
containers pinned to cores 0–5, load generator (k6) to 6–7. Single repeat, search steps only.
Capacity = highest req/s on the ×1.5 ladder with p99 under the SLO and zero errors.
Model: DeepFM small (ONNX), 200 rows per request, 1 replica. Full data: `bench/results/*.jsonl`
(committed); tables: `REPORT.md`, `bench/report/APPENDIX.md`; interactive charts:
`bench/report/report.html`.

## Capacity (req/s)

| Stack | Environment | p99 ≤ 20 ms | p99 ≤ 50 ms | p99 ≤ 100 ms |
|---|---|---|---|---|
| Triton (standalone) | default | 168 | 252 | 378 |
| FastAPI + ORT | default | 168 | 168 | 252 |
| Ray Serve + ORT | default (Python proxy, batching on, 6 threads) | 0 | 22 | 50 |
| Ray Serve + ORT | tuned (batching off, 4 threads) | 75 | 75 | 112 |
| Ray Serve → Triton (gRPC) | tuned | 50 | 75 | 112 |
| Ray Serve + in-process Triton | tuned | 75 | 112 | 112 |
| Ray Serve + ORT | HAProxy, batching off | 252 | 252 | 378 |
| Ray Serve + FastAPI | HAProxy, batching off | 112 | 252 | 252 |
| Ray Serve → Triton (gRPC) | HAProxy, batching off | 168 | 252 | 252 |
| Ray Serve + in-process Triton | HAProxy, batching off | 252 | 252 | 252 |

Rows are not comparable across environments. Triton and FastAPI were only measured in the default
environment; a same-environment table is the first item of the plan.

## Learnings

1. **Ray's cost is per-request overhead, not compute.** The model takes about 1.5 ms per request
   (JSON decode 0.8, ORT 0.7). Ray's Python proxy added roughly 8 ms of fixed cost per request,
   independent of payload size (1 row vs 200 rows: 8.6 vs 10.6 ms).
2. **Batching trades latency for throughput.** Ray's `@serve.batch` waits up to 5 ms. At these
   rates that wait costs more than it saves: turning it off raised capacity at 50 ms from 22 to 50.
3. **Thread budget matters on a shared box.** ORT's intra-op threads spin between requests. With 6
   threads on 6 cores, Ray's own processes compete for CPU. 4 threads raised capacity to 75.
4. **The HAProxy front end removes most of the proxy cost.** Ray + ORT went from 75 (tuned Python
   proxy) to 252 at 50 ms. Routing (per the Ray source): HAProxy asks Ray's router for a replica
   over `/internal/route`, then forwards to the replica directly. See `docs/HAPROXY.md`.
5. **The FastAPI layer inside a Ray replica costs capacity at tight SLOs.** Ray + FastAPI reached
   112 at 20 ms where Ray + ORT reached 252.
6. **Ray's Python proxy profile:** about 39% of proxy CPU in the Python interpreter, 27% in the
   kernel, 19% in Ray's native core. Context switches: about 125 per request in the proxy.
7. **Spinning off was no gain** (hypothesis tested and dropped). Direct ingress without HAProxy
   changed nothing on a single node.

## Mistakes and bugs found (fixed)

- Docker 29 would not start after the old 18.09 install: a legacy BuildKit cache blocked `dockerd`.
- Images missing `config/matrix.yaml`: the Triton repo renderer crashed.
- `ray_triton_grpc` called `run_until_complete` inside Ray's running loop: the replica crashed.
- The sweep kept waiting after a container exited: it now needs watching for `docker ps -a`.
- Interrupted sweep runs left `error` rows in the results, and one stale image produced an invalid
  Triton-in-process number (22/50/75 that did not reproduce, now 252). Invalid runs are archived in
  `bench/results/archive/`, not deleted.
- `/tmp` (12 GB RAM-backed) filled up from Ray session folders written by test runs. Fixed: Ray now
  writes to `~/.rbray`, and the test suite removes its sessions afterwards (`just clean-tmp`).
- The FastAPI-in-Ray test failed when the app used `from __future__ import annotations`.

## Not yet tested

Multi-model families, pipelines, bursts, pod failure, DCNv2, large models, Kubernetes, GPU,
binary protocols. See `docs/PLAN.md`.
