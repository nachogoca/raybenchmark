# raybenchmark

A benchmark of ways to serve CTR/ranking models (DeepFM and DCNv2, exported to ONNX) on
latency, throughput, fleet capacity, resource use and cost per request, exercising Ray Serve's
multiplexing, composition, autoscaling, backpressure, gRPC ingress and failure handling. Models are generated here with random weights. Every
stack serves the same KServe v2 JSON route, `POST /v2/models/<arch>_<size>/infer`, and is loaded
with k6. Every result row records the tool and image versions (`bench/versions.py`).

## What it measures

It compares ways to serve a model: FastAPI, Triton and Ray Serve.

- **Measures:** latency, capacity, CPU, memory and cost.
- **Speed test:** more and more load.
- **Replicas and batching:** does adding copies or grouping requests help?
- **Ingress:** Python proxy, HAProxy or gRPC.
- **Many models:** shared in one service.
- **Steps in a row:** one request through several models.
- **Auto-scaling:** the load jumps 10×.
- **Too much load, and a dead process.**

## Stacks

| Stack | What it is | Compose profile |
|---|---|---|
| FastAPI + ORT | uvicorn workers, ONNX Runtime in each | `fastapi_ort` |
| Ray Serve + ORT | Ray replicas with ONNX Runtime, optional `@serve.batch` | `ray_ort` |
| Ray Serve -> Triton (gRPC) | Ray front end, separate Triton container | `ray_triton_grpc` |
| Ray Serve + in-process Triton | Ray replicas each embedding a Triton server (NGC image) | `ray_triton_inproc` |
| Triton | standalone reference | `triton` |
| Ray family | Ray multiplexing / per-variant deployments / fan-out / pipeline | `ray_family` |

Python stacks share the image `raybench-py`; the in-process stack needs `raybench-triton-ray`
because `libtritonserver.so` only exists in the NGC image.

## Workflow

```bash
just setup          # uv sync, print the version table, fail if docker / k6 / kind are missing
just models         # export DeepFM / DCNv2 ONNX files and the 64-member DeepFM family
just payloads       # generate seeded k6 request bodies
just build          # build raybench-py, raybench-triton-ray, raybench-models, raybench-k6
just preflight      # CPU governor, free RAM, stray containers
just smoke          # every scenario shrunk to a pipeline check (~25 min)
just overnight      # full run (~9 h, resume-safe), then the report
just report         # bench/report/report.html and REPORT.md from bench/results/laptop.jsonl
```

Other recipes: `just estimate`, `just up <stack>`, `just down <stack>`, `just logs <stack>`,
`just test`, `just lint`. Individual scenarios: `uv run python -m bench.load.sweep --profile full --scenarios S1,A1`.

## Layout

- `bench/models`, `bench/protocol`: model export, KServe v2 codec, payloads.
- `bench/stacks/*`: the servers; `common/ort_session.py` is the one place ORT options live.
- `bench/load`: k6 scripts (`infer.js`, `infer_grpc.js`), sweep driver, scenarios. `bench/collect`: k6 parser, resource and Prometheus samplers, host guards. `bench/report`: analysis and the HTML report.
- `deploy/docker`: Dockerfiles, `compose.yaml` (one profile per stack), `.env.example`.
- `config/matrix.yaml`, `config/prices.yaml`: the test matrix and list prices.

## Notes

- Serving containers are pinned to `SERVING_CPUSET` (default cores 0-5, k6 uses 6-7) and
  `MEM_LIMIT` (default 8g), identical for every stack.
- `Dockerfile.k6` uses `grafana/k6:latest` (tag not verified offline); the digest is recorded
  by `bench.versions` once built.
- The laptop GPU (940MX, 2 GB) is not used; GPU runs are cloud-only.
- Ports: 8000 HTTP, 8001 Triton gRPC, 8002 Triton metrics, 8080 Ray metrics, 8265 Ray dashboard.

## Documents

- `docs/BENCHMARK_DESIGN.md`: questions, rules (MST, validity) and the laptop scenarios.
- `docs/PLAN.md`: GKE/EKS plan, including KServe and the fleet scenarios (not yet run).
- `docs/HAPROXY.md`: how Ray's HAProxy front end routes requests, and gateway tradeoffs.
