# raybenchmark v2: full Ray Serve benchmark, laptop overnight run plus GKE/EKS plan

## Context
`raybenchmark` compares 5 serving stacks for DeepFM/DCNv2 ONNX models (FastAPI+ORT, Ray+ORT,
Ray+FastAPI, Ray→Triton gRPC, Ray+in-process Triton, Triton) using k6 on the laptop. It has
real limits today:
- **Statistics:** one repeat per step; only p50/p90/p99/max; no histograms or time series; no
  confidence intervals.
- **Capacity:** a coarse ×1.5 ladder; no common load grid across stacks, so latency curves don't line up.
- **Resources:** 1 Hz cgroup CPU/memory per container; no split between proxy, router and
  replica; Ray and Triton Prometheus metrics are not collected; no check that the load
  generator kept up or that the CPU throttled.
- **Ray Serve:** only HAProxy and `@serve.batch` are exercised. Autoscaling, multiplexing,
  composition, gRPC ingress, backpressure and fault tolerance are untested (F1–F6 in
  `docs/PLAN.md` exist only on paper).
- **Cost:** every entry in `config/prices.yaml` is still `null`. The report is hand-drawn SVG.

**Goal:** a comprehensive, repeatable benchmark that uses the full set of Ray Serve features,
produces one self-contained interactive HTML report comparable to industry tools (MLPerf
Inference / Gatling / k6 style), and covers latency at many request rates and percentiles,
throughput, fleet capacity, resource use and cost efficiency.

**Decisions (from the user):**
- Run on the **laptop CPU only** (Docker Compose; 6 serving cores + 2 for k6). GKE and EKS are
  **planned in full, not run**.
- Stacks: keep the current 5 (plus `ray_fastapi`). **KServe goes into the cloud plan only.**
- Ray features to cover: autoscaling + bursts, multiplexing/families, composition/pipelines,
  ingress + resilience.
- Workload: CTR/ranking only (DeepFM/DCNv2).
- Cost: core-seconds priced at a dated reference cloud vCPU/GB rate.
- Budget: one **overnight run (~9–10 h)**.
- Report: **one self-contained interactive HTML file**.
- Order: plan → implement → commit and push to `origin` (github.com/nachogoca/raybenchmark) →
  the user starts the overnight run.

---

## 1. Measurement upgrades (applied to every scenario)

| Area | Change | Files |
|---|---|---|
| Per-request data | k6 `--out csv` (gz) per step. After the step, a parser builds a log-linear latency histogram (HDR-style, 1 µs–10 s), a percentile spectrum (p50, 75, 90, 95, 99, 99.9, 99.99, max), per-second series (rps, p50, p99, errors), and a breakdown of error codes. The parser stores aggregates in the jsonl row and the raw data as parquet (gitignored). | `bench/load/infer.js`, new `bench/collect/k6parse.py`, `bench/load/backends.py` (`run_k6`) |
| Common RPS grid | Every stack is measured at the same offered rates (e.g. 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 650 req/s; stop 2 points after saturation), so latency-vs-RPS curves line up. Separately, capacity search uses a ×1.25 ladder plus 3 bisection steps. | `config/matrix.yaml` (`grid_rps`, `ladder.factor`), `bench/load/sweep.py` |
| Capacity definition | Max sustainable throughput (MST) at each SLO: p99 ≤ SLO, errors = 0, k6 dropped iterations ≤ 0.1 %, achieved ≥ 99 % of offered. Also report the knee (where p99 rises above 2× its low-load value) and closed-loop max throughput. | `bench/report/analyze.py` |
| Repeats + CI | Confirm steps 3 reps; **stack order shuffled per rep** so slow drift (thermals) does not favour one stack; bootstrap 95 % CI on p99 and capacity. | `sweep.py`, `analyze.py` |
| Server-side metrics | 1 Hz Prometheus scraper: Ray (`:8080`: proxy and replica processing latency, queued and ongoing requests, replica count, multiplexed model load/unload/hits), Triton (`:8002`: queue / compute_input / compute_infer / compute_output µs, batch sizes), FastAPI (add `prometheus_client` middleware: in-handler time). Feeds a **latency breakdown** (client − server = network + proxy; queue vs compute). | new `bench/collect/prom.py`, `bench/stacks/fastapi_ort/app.py` |
| Per-component CPU | Extend `docker_reader`: read the container's `cgroup.procs`, sum `/proc/<pid>/stat` CPU per role (ProxyActor, HAProxy, ServeReplica, raylet, GCS, controller, tritonserver, uvicorn) → stacked CPU by component. | `bench/collect/sampler.py` |
| Validity guards | Sample k6 container CPU (flag > 80 % of its 2 cores), CPU frequency + package temperature from `/sys` (flag steps with throttling), container restarts. A row is `valid: false` with a reason; the report shows these, never hides them. `just preflight` checks governor, turbo, free RAM and Docker state, and prints the commands (`sudo cpupower frequency-set -g performance`, turbo off) that the user runs with `!`. | `sampler.py`, new `bench/collect/host.py`, `justfile` |
| Cost model | `config/prices.yaml` gets dated reference rates (GKE n2-standard and EKS m6i per vCPU-h and GB-h, source URL + `as_of`; the user fills or confirms them before the run). Report: $/1M req **allocated** (provisioned cores × time) and **consumed** (measured core-s), req/s per vCPU, req/s per GB, plus the replicas/vCPUs needed for target loads (1k / 10k / 100k rps) extrapolated from the scaling curve. | `config/prices.yaml`, `analyze.py` |

Already in place and reused: `Sweep` resume and run-id bookkeeping (`bench/load/sweep.py`),
`ComposeBackend` up/down/cold-start (`bench/load/backends.py`), `ServeConfig`
(`bench/stacks/common/config.py`), `make_session` (`bench/stacks/common/ort_session.py`), the
Triton repo writer (`bench/stacks/triton/repo.py`), versions (`bench/versions.py`), and the
export pipeline (`bench/models/export.py`).

## 2. Scenarios (laptop). Each answers one question and maps to a report section.

Model: DeepFM small, 200 rows/request unless noted. Ray runs with **HAProxy on** by default (the
best laptop result so far); the ingress comparison is its own scenario.

| ID | Question | Stacks | What's run | Est. |
|---|---|---|---|---|
| **S1** latency/capacity | Latency at each RPS, capacity per SLO | all 6 | common RPS grid + capacity search + confirm (3 reps × SLO 20/50/100) + overload 1.5×/3× + closed loop; rows=1 grid only | 3.0 h |
| **S1-scale** fleet scaling | How capacity grows with replicas (laptop "fleet") | ray_ort, ray_triton_inproc, fastapi_ort, triton | replicas 1/2/4 (cores split evenly), search only → scaling efficiency, req/s per core | 1.0 h |
| **S1-batch** batching | When batching helps | Ray (`@serve.batch` 2/5 ms), Triton dynamic batching | on/off at rows 1 and 200, search only | 0.5 h |
| **I1** ingress | Proxy cost | ray_ort: Python proxy vs HAProxy vs **gRPC ingress** (KServe v2 proto); Triton HTTP vs gRPC as reference | grid + search | 0.6 h |
| **F1** multiplexing | Capacity and memory for a 16-variant family under Zipf(1.1) traffic | Ray `@serve.multiplexed` (`max_num_models_per_replica` 4 / 16), Ray 16 deployments, Triton 16-model repo, FastAPI 16 sessions | grid + search; then Ray mux at 4/16/64 variants | 1.1 h |
| **F2** cold variant | First-request latency for a cold model | Ray mux (evicted), Triton explicit load, FastAPI lazy | 5 trials each | 0.3 h |
| **F3** fan-out | Cost of scoring with K variants | Ray `DeploymentHandle` gather, Triton ensemble, FastAPI `asyncio.gather` | K = 1/4/8 at a fixed rate, 2 reps | 0.5 h |
| **F5** pipeline | Cost of composition | feature stage → routed variant → postprocess: Ray composed deployments, Triton ensemble (Python backend for the feature step), FastAPI in-process chain | grid to SLO | 0.5 h |
| **A1** autoscaling burst | How fast Ray absorbs a 10× step | Ray autoscaling (`target_ongoing_requests` 2 vs 8, upscale_delay 0 vs 10 s, min 1 / max 4 replicas) vs static fleets of 1 and 4 replicas | base → 10× for 120 s → base, 3 reps; replicas over time, time-to-SLO, errors, core-s | 1.0 h |
| **R1** backpressure | Behaviour beyond capacity | Ray `max_queued_requests` (off / 50), Triton `max_queue_size`, uvicorn `--limit-concurrency` | 1.5/2/3× capacity: goodput, 503 rate, p99 of successful requests | 0.3 h |
| **R2** failure | Recovery when a replica dies | Ray: kill -9 a ServeReplica pid (2 replicas); FastAPI: kill a worker; Triton: kill the container (restart policy) | steady 60 % of capacity, 3 reps: error count, recovery time, p99 timeline | 0.4 h |

Total ≈ 9.2 h. The run order is the priority order above; resume keeps a partial night useful.
`sweep.py --dry-run` prints the real estimate before starting.

The laptop has no HPA. Autoscaling for non-Ray stacks is compared through static
under- and over-provisioned fleets; HPA and KServe autoscaling are in the cloud plan.

### Implementation of the new scenarios
- **Scenario runner:** generalise `sweep.py` so a scenario (in `config/matrix.yaml` under
  `scenarios:`) declares stacks, load shape (`grid` | `search` | `fixed` | `step` | `trace`) and
  hooks (`before_step`: kill pid, evict model). The existing ladder logic becomes one load shape.
- **k6:** `infer.js` gains `MODEL_MIX` (Zipf weights → `serve_multiplexed_model_id` header or
  model in path), a `step` executor (`ramping-arrival-rate`), and a gRPC variant (`infer_grpc.js`,
  KServe v2 `grpc_predict_v2.proto`).
- **Family models:** `bench/models/export.py` gets `--variants N` (seeded `deepfm_small_v00..`);
  `just family`.
- **New stack apps** (reuse `ray_runner.handle`, `make_runner`):
  `bench/stacks/ray_mux/`, `bench/stacks/ray_fanout/`, `bench/stacks/ray_pipeline/`,
  `bench/stacks/ray_grpc/`; FastAPI and Triton family/ensemble/pipeline modes go in the
  existing `fastapi_ort/app.py` and `triton/repo.py` behind env flags. Ray autoscaling and
  `max_queued_requests` go through `ServeConfig` → `.options(autoscaling_config=…)`.
- **Compose:** new profiles in `deploy/docker/compose.yaml`, same cpuset/mem limits.
- **Tests:** extend `tests/test_sweep.py` (scenario expansion, MST rule), new `tests/test_k6parse.py`
  (histogram and percentile accuracy on synthetic data), `tests/test_stacks_local.py` (mux,
  fan-out, pipeline answer correctly).

## 3. Report: `bench/report/report.html` (single file, Plotly inlined, works offline)
Rewrite `bench/report/html.py` + `charts.py` on Plotly (add `plotly` to the `analysis` group). Sections:
1. **Executive summary:** verdict cards per question (winner, margin, CI, caveats); scorecard
   table (capacity@50 ms, p99@100 rps, $/1M req, req/s/vCPU, RSS, cold start).
2. **Methodology & environment:** hardware, pinning, versions, MST rule, validity results,
   invalid-row list.
3. **Latency:** p50/p90/p99/p99.9 vs offered RPS per stack with CI bands; percentile spectrum
   (HDR plot, log-x "nines") at 3 fixed RPS; CDFs; latency heatmap over time; per-step time series.
4. **Throughput & capacity:** MST per SLO with error bars; achieved vs offered (saturation);
   closed-loop max; knee markers.
5. **Fleet capacity:** capacity vs replicas, scaling efficiency, req/s per core, replicas/vCPU
   needed for 1k/10k/100k rps.
6. **Latency breakdown:** stacked client / proxy / queue / compute per stack at a common RPS.
7. **Resources:** CPU by component (stacked) vs RPS; core-ms per request; memory and RSS per loaded model.
8. **Cost efficiency:** $/1M req (allocated vs consumed) per SLO; cost-vs-p99 Pareto frontier.
9. **Ray Serve features:** ingress (I1), multiplexing (F1/F2: capacity vs #variants,
   memory/model, cold latency, hit rate), composition (F3/F5), autoscaling (A1: load, replicas
   and p99 on one shared timeline), backpressure (R1: goodput curve), failure (R2: error/p99
   timeline around the kill).
10. **Overload & cold start.**
11. **Appendix:** coverage matrix, every row (sortable table), raw-data paths, reproduction commands.

Also add the report to `REPORT.md` and `docs/RESULTS.md` (markdown headline tables).

## 4. Cloud plan (written, not run): rewrite `docs/PLAN.md`
- **Clusters:** GKE (n2-standard-8 serving pool, c2/n2 load-gen pool) and EKS (m6i.2xlarge
  serving, separate load-gen node group), same vCPU per node; one GPU pool option (L4) for later.
  Prices with date and source in `config/prices.yaml`.
- **Stacks:** the 6 above + **KServe** (RawDeployment mode, Triton runtime for ONNX, HPA; the
  family scenario as one InferenceService per variant), all on Kubernetes so the topology is
  identical.
- **Ray on k8s:** KubeRay RayService, `proxy_location: EveryNode`, Service selects worker pods,
  head has no replicas (fix in `deploy/k8s/render.py`), HAProxy per node, Ray autoscaler +
  cluster autoscaler for node-level scale-out.
- **Fleet scenarios added on cloud:** capacity vs nodes (1/2/4/8), cluster-autoscaler burst
  (time to new node), node failure, HPA vs Ray autoscaling vs KServe autoscaling, cross-cloud $/1M req.
- **Gates:** same validity rules; re-run anything that differs from the laptop by more than one
  ladder step; estimated cost per run, listed per cloud.

## 5. Docs and delivery
- Update `docs/BENCHMARK_DESIGN.md` (questions, MST rule, validity rules, scenario table) and
  `README.md` (new recipes).
- New `just` recipes: `preflight`, `family`, `overnight` (= preflight → build → sweep all scenarios → report).
- Commit on a branch (`bench-v2`), push to `origin`, open a PR if the user wants; then the
  user starts `just overnight` (backgrounded, resumable).

## Verification (before pushing)
1. `just test` and `just lint` pass, including the new parser and scenario tests.
2. `just overnight --profile smoke`: every scenario at 1 short step per stack (~25 min), all
   rows `valid`, report renders every section.
3. Sanity-check against the existing results: S1 capacities for triton / ray_ort+HAProxy are
   within one ladder step of `docs/RESULTS.md` (252 req/s at 50 ms).
4. Open `report.html` offline (no network) and confirm the charts are interactive; check the size (< 15 MB).
5. `sweep.py --dry-run --profile overnight` estimate ≤ 10 h.
