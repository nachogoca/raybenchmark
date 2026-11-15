# raybenchmark

A benchmark of five ways to serve CTR/ranking models (DeepFM and DCNv2, exported to ONNX) on
latency, throughput and cost per request. Models are generated here with random weights. Every
stack serves the same KServe v2 JSON route, `POST /v2/models/<arch>_<size>/infer`, and is loaded
with k6. Every result row records the tool and image versions (`bench/versions.py`).

## The five stacks

| Stack | What it is | Compose profile |
|---|---|---|
| FastAPI + ORT | uvicorn workers, ONNX Runtime in each | `fastapi_ort` |
| Ray Serve + ORT | Ray replicas with ONNX Runtime, optional `@serve.batch` | `ray_ort` |
| Ray Serve -> Triton (gRPC) | Ray front end, separate Triton container | `ray_triton_grpc` |
| Ray Serve + in-process Triton | Ray replicas each embedding a Triton server (NGC image) | `ray_triton_inproc` |
| Triton | standalone reference | `triton` |

Python stacks share the image `raybench-py`; the in-process stack needs `raybench-triton-ray`
because `libtritonserver.so` only exists in the NGC image.

## Workflow

```bash
just setup          # uv sync, print the version table, fail if docker / k6 / kind are missing
just models         # export DeepFM / DCNv2 ONNX files into artifacts/onnx
just payloads       # generate seeded k6 request bodies
just build          # build raybench-py, raybench-triton-ray, raybench-models, raybench-k6
cp deploy/docker/.env.example deploy/docker/.env   # optional: pick model, replicas, threads
just up fastapi_ort # or ray_ort | ray_triton_grpc | ray_triton_inproc | triton
just bench smoke    # k6 sweep against whatever is up (env defaults to laptop-cpu)
just down fastapi_ort
just report         # analysis tables and charts from bench/results
```

Other recipes: `just logs <stack>`, `just test`, `just lint`.

## Layout

- `bench/models`, `bench/protocol`: model export, KServe v2 codec, payloads.
- `bench/stacks/*`: the five servers; `common/ort_session.py` is the one place ORT options live.
- `bench/load`: k6 script and sweep driver. `bench/report`: analysis.
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

- `docs/RESULTS.md`: results so far and learnings.
- `docs/PLAN.md`: cloud benchmark plan, including the model-family scenarios.
- `docs/BENCHMARK_DESIGN.md`: questions, pass/fail rules and scenarios.
- `docs/HAPROXY.md`: how Ray's HAProxy front end routes requests, and gateway tradeoffs.
