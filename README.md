# raybenchmark

Benchmark of model-serving stacks for CTR/ranking models (DeepFM and DCNv2, exported to ONNX):
FastAPI + ONNX Runtime, Triton, and several Ray Serve configurations. It measures latency,
capacity, resource use and cost, and exercises Ray Serve features a plain HTTP replica does not
use: multiplexing, composition, autoscaling, backpressure, gRPC ingress and failure handling.

All stacks serve the same KServe v2 JSON route (`POST /v2/models/<name>/infer`) from the same
seeded-weight ONNX files, and are loaded open-loop with k6. Every result row records tool and image
versions (`bench/versions.py`). The output is one self-contained HTML report.

## Tests

| Test | Question | `--scenarios` id |
|---|---|---|
| Latency and capacity | How do latency and capacity compare as offered load grows? Includes overload and closed-loop max throughput. | `S1` |
| Replica scaling | How does capacity scale with replica count on a fixed CPU budget? | `S1_scale` |
| Batching | When does request batching pay off (Ray `@serve.batch`, Triton dynamic batching)? | `S1_batch` |
| Ingress | What does the ingress cost: Ray Python proxy, HAProxy, gRPC; Triton HTTP vs gRPC? | `I1` |
| Model family (multiplexing) | How do layouts compare for 16 models under Zipf traffic: Ray multiplexing, one deployment per model, Triton, FastAPI? | `F1` |
| Cold model load | What is the first-request latency of a model that is not loaded? | `F2` |
| Ensemble fan-out | What does scoring one request with K models cost as K grows? | `F3` |
| Multi-stage pipeline | What does a feature, model, calibration pipeline cost on each stack? | `F5` |
| Autoscaling under a 10× burst | How fast does each setup absorb a 10× step in load? | `A1` |
| Overload and backpressure | How does each stack behave past capacity, with bounded and unbounded queues? | `R1` |
| Failure recovery | How does each stack recover when a serving process is killed? | `R2` |

Method, rules (capacity definition, validity) and limits: `docs/BENCHMARK_DESIGN.md`.

## Stacks

| Stack | What it is | Compose profile |
|---|---|---|
| FastAPI + ORT | uvicorn workers, ONNX Runtime in each | `fastapi_ort` |
| Ray Serve + ORT | Ray replicas with ONNX Runtime, optional `@serve.batch` | `ray_ort` |
| Ray Serve + FastAPI | Ray replicas behind a FastAPI ingress | `ray_fastapi` |
| Ray Serve → Triton (gRPC) | Ray ingress, separate Triton container | `ray_triton_grpc` |
| Ray Serve + in-process Triton | Ray replicas each embedding a Triton server (NGC image) | `ray_triton_inproc` |
| Triton | standalone reference | `triton` |
| Ray family | multiplexing, per-model deployments, fan-out, pipeline | `ray_family` |

Python stacks share the image `raybench-py`; the in-process stack needs `raybench-triton-ray`
because `libtritonserver.so` only exists in the NGC image. **The images hold a frozen copy of the
code**: rebuild them after any code change (`just build-serving`; `just smoke` and `just overnight`
do it for you).

## Run it

```bash
just setup          # uv sync, print the version table, fail if docker / k6 / kind are missing
just models         # export DeepFM / DCNv2 ONNX files and the 64-member DeepFM family
just payloads       # generate seeded k6 request bodies
just preflight      # CPU governor, free RAM, stray containers
just smoke          # every config of every test with short steps, about an hour
just overnight      # full run, about 9 h, resume-safe; builds the report at the end
just monitor        # is the run alive? progress, last log line, disk, containers, alerts
just report         # bench/report/report.html and REPORT.md from the latest results
```

Other recipes: `just estimate` (time estimate, nothing started), `just up|down|logs <stack>`,
`just test`, `just lint`. One test only: `uv run python -m bench.load.sweep --profile full --scenarios S1,A1`.

**Safety.** `just overnight` starts a monitor beside the run (`bench/results/runs/monitor.log`). The
run stops itself, and can resume, when free disk drops below 15 GB (`guards` in `config/matrix.yaml`).
Container logs are capped at 2 × 20 MB. A step is marked invalid (and excluded from capacity) when
k6 saturates its CPUs, the serving CPUs throttle, or a container restarts.

## Run on macOS

`just setup` detects the OS and prints the right install commands (`brew install k6`, Docker Desktop or
colima). The recipes use a separate results environment, `mac`, and everything else is the same:

```bash
brew install uv just k6                          # plus Docker Desktop (8+ CPUs, 12+ GB) or colima
just setup && just models payloads preflight
just smoke      # then: just overnight
```

What differs from Linux, and why:

- **No CPU pinning.** Containers run in the Docker VM and k6 runs on the host. Serving CPUs are sized
  automatically (VM CPUs minus 2); k6 is not pinned. Results are **not comparable with Linux runs**.
- **Resource numbers come from `docker stats`.** cgroups live inside the VM, so there is no per-process CPU
  split (the "CPU by component" chart is empty).
- **Triton is skipped on Apple Silicon.** The NGC Triton images are x86-64 only, and emulation would give
  meaningless numbers. Tests that use Triton run without it. `RAYBENCH_EMULATE=1` forces them anyway.
- **Linux-only tools.** `perf`, `strace` and `taskset` recipes (`setup-profiling`, `perf-top`, `probe`) say so
  and exit; `just preflight` skips the CPU-governor and turbo checks.

## Report

`bench/report/report.html` is a single file (Plotly inlined, works offline):

- **Summary:** headline cards, scorecard, and the list of tests.
- **Methodology.**
- **One section per test:** Question, Method and Metrics first, then the charts. Latency vs offered
  load, percentile curves and CDFs, capacity with bootstrap CIs, CPU by component, latency
  breakdown, cost, replica scaling, ingress, multiplexing, fan-out, pipeline, autoscaling timeline,
  goodput under overload, error timeline around a kill.
- **Software versions**, then an **appendix** with coverage, invalid steps and every measured step.

`REPORT.md` holds the headline tables.

## Layout

- `bench/models`, `bench/protocol`: model export, KServe v2 JSON and gRPC codecs, payloads.
- `bench/stacks/*`: the servers; `common/ort_session.py` is the one place ORT options live.
- `bench/load`: k6 scripts (`infer.js`, `infer_grpc.js`), `sweep.py` (run bookkeeping, one k6 step), `scenarios.py` (the tests).
- `bench/collect`: k6 CSV parser, resource and Prometheus samplers, host and disk guards.
- `bench/monitor.py`: run monitor. `bench/report`: analysis and the HTML report.
- `deploy/docker`: Dockerfiles and `compose.yaml` (one profile per stack). `deploy/k8s`, `deploy/cloud`: cluster manifests and scripts (not run yet).
- `config/matrix.yaml`: SLOs, load grid, test parameters, guards. `config/prices.yaml`: cloud list prices.

## Notes

- A single host: serving containers are pinned to 6 CPUs (`SERVING_CPUSET`, default cores 0-5) with the same `MEM_LIMIT`; k6 gets the other 2 (cores 6-7). Identical for every stack.
- `Dockerfile.k6` uses `grafana/k6:latest`; the digest is recorded by `bench.versions` once built.
- GPUs are not used; GPU runs are planned for the cloud.
- Dollar figures use reference cloud prices in `config/prices.yaml`. They are marked `verified: false` until checked against the price pages.
- Ports: 8000 HTTP, 8001 Triton gRPC, 8002 Triton metrics, 8080 Ray metrics, 9000 Ray gRPC, 8265 Ray dashboard.

## Documents

- `docs/BENCHMARK_DESIGN.md`: questions, rules and the single-host tests.
- `docs/PLAN.md`: GKE/EKS plan, including KServe and the cluster scale-out tests (not yet run).
- `docs/HAPROXY.md`: how Ray's HAProxy ingress routes requests, and gateway tradeoffs.
