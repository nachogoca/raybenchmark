# Benchmark design

Goal: compare ways to serve ML models (FastAPI + ONNX Runtime, Triton, Ray Serve in several
configurations) on latency, throughput, fleet capacity, resource use and cost, and exercise the
Ray Serve features that a plain HTTP replica does not use: multiplexing, composition,
autoscaling, backpressure, gRPC ingress and fault tolerance. The benchmark records what each
stack delivers; it does not presume a winner. Parameters live in `config/matrix.yaml`, the scenario
code in `bench/load/scenarios.py`, the report in `bench/report/`.

## Rules (every stack)
- Same ONNX artifacts, same CPU budget, same k6 scripts, same SLOs; every row records tool and
  image versions (`bench/versions.py`) and the settings used.
- Load is open-loop (constant arrival rate): latency is measured against the schedule, so a slow
  server cannot hide queueing by slowing the client (no coordinated omission).
- **Capacity (MST)** at an SLO = highest grid rate whose median-of-3 p99 meets the SLO with zero
  errors and ≤ 0.1 % dropped arrivals. 95 % CI by bootstrapping the repetitions.
- **Validity:** a step is invalid, excluded from capacity and listed in the report appendix, when
  k6 used more than 80 % of its cores, the serving CPUs throttled, or a container restarted
  (except where a scenario kills one on purpose).
- Stack order is shuffled per repetition so thermal drift does not favour one stack.
- Negative results are reported as they are. Any claim worth acting on gets re-run.

## Scenarios (laptop, `just overnight`, about 9 h)
Model: DeepFM small, 200 rows per request, KServe v2 JSON, unless noted. Ray runs with its native
HAProxy front end except in I1.

| ID | Question | What runs |
|---|---|---|
| S1 | Latency at each rate; capacity per SLO | All six stacks on a common rate grid (25 … 650 req/s), 3 repetitions, rows=1 latency points, overload at 1.5×/3×, closed-loop maximum, refine steps around the knee |
| S1_scale | How does capacity grow with replicas? | Ray + ORT, FastAPI, Triton at 1/2/4 replicas on the same 6 cores |
| S1_batch | When does batching help? | Ray `@serve.batch` (2/5 ms), Triton dynamic batching, at rows 1 and 200 |
| I1 | What does the ingress cost? | Ray Python proxy vs HAProxy vs gRPC; Triton HTTP vs gRPC |
| F1 | Capacity and memory of a model family under Zipf(1.1) traffic | Ray `@serve.multiplexed` (LRU 4/16), one deployment per variant, Triton 16-model repository, FastAPI 16 sessions; Ray mux at 4/64 variants |
| F2 | First-request latency of a cold or evicted variant | Ray mux, FastAPI lazy load, Triton explicit load |
| F3 | Cost of scoring with K variants | Ray `DeploymentHandle` gather, FastAPI `asyncio.gather`, Triton ensemble; K = 1/4/8 |
| F5 | Cost of a pipeline | featurize → routed variant → postprocess: Ray composed deployments, FastAPI in-process, Triton ensemble with python backend |
| A1 | How fast does Ray absorb a 10× burst? | 30 → 300 req/s for 90 s; static 1 and 4 replicas vs autoscaling (`target_ongoing_requests` 2/8, upscale delay 0/10 s); fresh stack per repetition |
| R1 | What happens beyond capacity? | Ray `max_queued_requests`, uvicorn `--limit-concurrency`, Triton `max_queue_size` vs unbounded, at 1.5×/3× capacity: goodput, rejects, p99 of successful requests |
| R2 | What happens when a serving process dies? | Kill one Ray replica / uvicorn worker / the Triton container under steady load: errors, recovery time |

Metrics: p50/p90/p99/p99.9/max and HDR-style percentile spectrum; MST per SLO; achieved vs
offered; knee; core-ms per request by component (proxy, replica, raylet, Triton, uvicorn); RSS and
memory per loaded model; cold start; latency breakdown (queue / compute / proxy+network) from
Prometheus; $ per 1M requests, allocated and consumed (`config/prices.yaml`).

## Limits of the laptop run
Serving and load generator share one CPU package and memory bus, so absolute numbers do not
transfer to a cluster; rankings and ratios are the product. The laptop "fleet" splits the same
six cores across replicas, so it measures per-replica overhead, not scale-out. Prices are
reference list rates applied to measured CPU time. Cluster scale-out, HPA and KServe are in
`docs/PLAN.md`.

## Kubernetes topology (cloud phase)
- Ray Serve: `proxy_location: EveryNode`, the Service selects worker pods, the head has no
  replicas. `deploy/k8s/render.py` currently routes only through the head; fix before any cluster
  Ray result is valid.
- Triton, FastAPI, KServe: Deployment + Service with HPA. Serving pods and k6 run on separate
  node pools.
- Gateway logic placement and tradeoffs: `docs/HAPROXY.md`.
