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

## Phase 0: upgrade this machine (needs sudo)

Docker here is 18.09 and fails on this kernel. Replace it:

```bash
# remove the old engine and the stale "cosmic" apt source
sudo apt-get remove -y docker-ce docker-ce-cli containerd.io
sudo rm -f /etc/apt/sources.list.d/docker*.list
# add Docker's official repo for the running Ubuntu codename
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker

# k6 v2.4.0
curl -LO https://github.com/grafana/k6/releases/download/v2.4.0/k6-v2.4.0-linux-amd64.deb
sudo dpkg -i k6-v2.4.0-linux-amd64.deb

# kind v0.33.0
mkdir -p ~/.local/bin
curl -Lo ~/.local/bin/kind https://kind.sigs.k8s.io/dl/v0.33.0/kind-linux-amd64
chmod +x ~/.local/bin/kind

# current kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && mv kubectl ~/.local/bin/

# NGC (Triton images)
docker login nvcr.io
```

If Docker's repo does not yet have a package for your codename, use the newest codename it
does publish. You also need [`just`](https://github.com/casey/just) and `uv`.

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

## Learnings so far (first smoke run, 2026-11-15, laptop-cpu)

Smoke numbers prove the pipeline works; they are single-repeat search results, not publishable.
deepfm_small, 200 rows/request, 1 replica. Capacity = highest req/s with p99 under the SLO, 0 errors.

| Stack | @20 ms | @50 ms | @100 ms | Peak RSS |
|---|---|---|---|---|
| triton | 168 | 252 | 378 | 47 MB |
| fastapi_ort | 168 | 168 | 252 | 86 MB |
| ray_ort | 0 | 22 | 50 | 933 MB |
| ray_triton_grpc | 0 | 22 | 50 | 1004 MB |
| ray_triton_inproc | 0 | 15 | 22 | 971 MB |

Capacities land on rungs of the x1.5 ladder (10, 15, 22, 33, 50, 75, 112, 168, 252, 378), so
"22 vs 50" is one or two rungs, not a precise ratio. Bisection is off in smoke.

### Getting it to run (host and harness gotchas)

- Docker 29 would not start after the old 18.09 install: `dockerd` refused the legacy BuildKit
  cache (`/var/lib/docker/buildkit/metadata.db`). Moving that file aside fixed it. Existing
  containers and networks were unaffected.
- The CLI was on the `colima` context but Lima is not installed; `docker context use default`.
- `compose up` straight after a `down -v` of the same project can fail transiently (network not
  yet removed). Wait a few seconds or retry.
- Both Python images must contain `config/` (`bench/stacks/triton/repo.py` reads
  `config/matrix.yaml`). Missing, the `triton_repo` init container crashed and took down every
  Triton-based stack. The in-process image needs it too.
- A Ray Serve replica's `__init__` already runs inside an event loop. `run_until_complete` there
  crashes the replica; build the client in `__init__` and wait for the backend on first use.
- The sweep did not notice a crashed container: it sat idle waiting. Watch `docker ps -a` for
  exited containers when running it unattended.
- Resume only skips configs with a `done` row. Failed attempts leave `error` rows in
  `bench/results/<env>.jsonl` that also show up in the appendix; drop them after a failed run.

### Why is Ray 5-10x slower here? (investigation, not solved)

Ray p50 is ~18-20 ms even at 10 req/s, against ~3 ms for FastAPI and Triton, at every rate. The
two Ray stacks with different backends (ORT in the replica, Triton over gRPC) land on the same
numbers, so the cost is in Ray Serve's ingress path, not the model backend.

Measured with a closed-loop probe pinned to cores 6-7 (200 rows unless noted):

| Change | c=1 p50 | Note |
|---|---|---|
| FastAPI, ORT 6 threads | 2.9 ms | reference |
| ray_ort, batching on, 6 threads | 20.5 ms | the benchmark config |
| batching off | 13.9 ms | `@serve.batch` waits up to 5 ms at low rates, ~6 ms total |
| batching off, 4 ORT threads | 10.2 ms | leaves cores for Ray's own processes |
| batching off, 1 ORT thread | 10.9 ms | compute is not the cost |
| 1 row instead of 200 | 8.6 vs 10.6 ms | ~8 ms is fixed per request, only ~2 ms scales with size |

- Compute is ~1.5 ms (JSON decode 0.8, ORT 0.7, encode 0.01). Ray's replica processing metric
  averages ~7.9 ms, so ~6 ms is plumbing inside Ray, not the model.
- Router wait is ~1 ms per request. Process CPU at ~90 req/s: replica 3.7 cores (mostly ORT
  threads), proxy 0.69 cores (~7.7 ms CPU per request), everything else idle.
- py-spy on the proxy shows Python request handling is only ~5% of a core. The rest of the
  proxy's CPU is native (Ray core worker gRPC/asio threads) which py-spy cannot see; next step
  is `perf`.
- No effect, tested: `RAY_SERVE_RUN_USER_CODE_IN_SEPARATE_THREAD=0`,
  `RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP=0`, `RAY_SERVE_ENABLE_DIRECT_INGRESS=1` (not confirmed
  that it engaged), and turning ORT spinning off (no gain; hypothesis dropped).
- Run-to-run noise on this laptop is about +-3 ms (identical configs gave 10.0, 11.1 and
  14.1 ms). Differences smaller than that need repeated, interleaved runs.
- Cheap wins available now: batching off for Ray at low rates, and fewer ORT threads than cores
  for any stack that shares a cpuset with Ray. Closed-loop throughput at c=4 rose from ~99 to
  ~147 req/s with both. Not yet confirmed through the sweep.

### Kubernetes routing

`deploy/k8s/render.py` points the `ray-*` Service at the head pod only, and the head has
`num-cpus: 0`. So every request goes client -> head proxy -> replica on a worker (one extra
pod-to-pod hop, one proxy process for all HTTP parsing), while FastAPI and Triton Services hit
the serving pod directly. That penalises Ray twice. Route through worker proxies instead
(`proxy_location: EveryNode` and a Service selecting `ray.io/serve=true` pods, as KubeRay's
`<name>-serve-svc` does; check against the Ray version in use).

## How to improve this benchmark

- **Noise control**: pin the CPU governor, run each config 3+ times interleaved, report spread.
  Cores 0-5 of this i7-6700HQ (4C/8T) include hyperthread siblings; pin by physical core.
- **Finer capacity**: turn bisection on in smoke, or the ladder rungs hide real differences.
- **Isolate fixed overhead**: add a no-op Ray Serve echo and a no-op FastAPI echo (no model) to
  measure each framework's per-request cost directly, and a 1-row payload run per stack.
- **Matrix coverage**: add batching off for Ray, and an ORT intra-thread sweep (threads below
  core count). Right now `batching: [true]` hides the largest single Ray lever.
- **Fail loudly**: detect exited containers in the sweep and abort the config instead of
  waiting; retry `compose up` once; do not write `error` rows into the results used by reports.
- **Observability**: optional profile mode with py-spy and `perf` in a throwaway image, and a
  Prometheus + Grafana compose profile scraping Ray (:8080) and Triton (:8002). Worth it for the
  multi-hour `core`/`full` runs so Ray's internals can be lined up against k6 results.
- **Fair Ray ingress**: compare Ray's HTTP proxy against its gRPC proxy and against direct
  ingress with a flag check that it engaged; consider reporting Ray with and without the proxy.
- **Report**: show the p50 floor at low rate, not only capacity; filter error rows from the
  appendix; label smoke results as unconfirmed.
- **Cost**: fill `config/prices.yaml` for the laptop (or skip $/req there) so the cost column is
  not n/a.

## Documents

- `docs/RESULTS.md`: results so far and learnings.
- `docs/PLAN.md`: cloud benchmark plan, including the model-family scenarios.
- `docs/BENCHMARK_DESIGN.md`: questions, pass/fail rules and scenarios.
- `docs/HAPROXY.md`: how Ray's HAProxy front end routes requests, and gateway tradeoffs.
