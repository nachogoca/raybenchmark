# Cloud benchmark plan

Continues `docs/RESULTS.md` (laptop) on a Kubernetes cluster. Questions and pass/fail rules are
in `docs/BENCHMARK_DESIGN.md`; this file defines the runs.

## 0. Before running anything on the cloud
- Fix `deploy/k8s/render.py`: Ray Serve with `proxy_location: EveryNode`, the serve Service selects
  worker pods, the head has no replicas. Without this, cluster Ray numbers are invalid.
- Regenerate artifacts on the new machine: `just models` (ONNX is gitignored, ~1.3 GB) and `just payloads`.
- Rebuild images (`just build`); build the torch image only if the torch runtime is in scope.
- Node pools: a serving pool (CPU, one node type) and a separate load-generator pool. Serving and
  k6 must never share nodes.
- Record in every row: node type, cores, memory, Kubernetes and Ray versions (`bench/versions.py`).

## 1. Stacks (same artifacts, same CPU per node, same SLOs)
| Stack | Serving pattern |
|---|---|
| FastAPI + ORT | Deployment, uvicorn workers = replicas, HPA |
| Triton | Deployment, model repository, HPA |
| Ray Serve + FastAPI | RayService, `@serve.ingress`, HAProxy per node |
| Ray Serve + Triton (gRPC) | RayService + Triton Deployment, HAProxy per node |

Each stack runs its grid: batching on/off, ORT threads, replicas. Report the best config and keep
every row.

## 2. Model family (the workload Ray Serve is built for)
A *family* is many related models sharing an architecture and feature schema, differing in weights:
per-market, per-tenant or per-segment variants of one ranking model. Realistic counts are tens to
hundreds; the plan starts at 16 and scales up.

Generate the family: 16 DeepFM small variants (different seeds, same schema), exported with
`bench.models.export` under distinct names (`deepfm_small_v00` … `v15`). Add 4 DCNv2 variants for
heterogeneity. Each variant's ONNX file is ~5–30 MB (small size).

## 3. Scenarios

### F1 — family under skewed traffic (capacity and density)
- Traffic: Zipf over 16 models (s = 1.1: one hot model, long tail), 200 rows per request.
- Ladder to the SLO, as in S1, measured against the whole family.
- Metrics: capacity per SLO, p99, cores busy, RSS per **loaded** model, total memory.
- Ray: one deployment for the family with `@serve.multiplexed` (LRU cache of loaded variants per
  replica; requests carry `serve_multiplexed_model_id`). Also measure one deployment per variant
  (16 deployments, each autoscaled) to show the cost of that choice.
- Triton: one model repository with 16 models. Compare `--model-control-mode=explicit` (all loaded)
  with on-demand loading.
- FastAPI: one process holding all 16 sessions, and the lazy variant (load on first request).

### F2 — cold variant (scale-to-zero, first-request latency)
- A variant untouched for 10 minutes receives one request, then steady load on it.
- Metrics: first-request latency, time until p99 < SLO, memory after load.
- Compare: Ray multiplexing (load into a replica that has room), Triton on-demand load,
  FastAPI lazy load.

### F3 — family fan-out (one request scored by K variants)
- A request is scored by K = 4 variants of the family and the scores are averaged (ensemble
  across markets or segments). K tested at 1, 4, 8.
- Ray: parallel fan-out with `DeploymentHandle`s inside one request. Triton: ensemble with four
  model steps. FastAPI: `asyncio.gather` over four sessions in one process.
- Metrics: p99 at a fixed rate, cores per 1M requests. Shows how each stack scales with K.

### F4 — canary rollout inside the family
- Variant `v03` at 100% of its traffic, then 95/5 with the new version `v03b`, then 50/50, then 100%.
- Metrics: p99 per version during the shift, error rate, time to complete the shift without
  dropped requests.
- Ray: traffic split at the ingress; Triton: model version policy; FastAPI: gateway or
  application-level split.

### F5 — pipeline on top of the family
- CPU feature stage (decode, normalise) → one of the family variants (routed by a segment field
  in the request) → postprocess.
- Same definition as S3 in `docs/BENCHMARK_DESIGN.md`, with the family routing added.

### F6 — burst on the family
- Steady load on F1's mix, then a 10× step for 2 minutes. Same metrics as S4, reported per stack.

### S1–S5 (from the design doc)
Single model (S1), pipeline (S3), burst (S4), and pod failure (S5) run on the cluster too, so the
family results sit next to the single-model baseline.

## 4. Run order and gates
1. Smoke: each stack, F1 at 1 variant, to prove the pipeline end to end.
2. Single-model baseline (S1) on the cluster, all four stacks, repeated 3×.
3. F1 across all four stacks; repeat the best config 3×.
4. F2–F6 in that order, only on configs that passed F1.
5. Re-run any result that differs from the laptop by more than one ladder step, and check why.

A run is valid only if: no errors, load generator not CPU-bound (k6 cpu below 80% of its cores),
and serving pods were not restarted during the step.

## 5. Output
- Same tables and charts as the laptop (`just report`, `just report-html`), per environment.
- Family scenarios get their own section in `report.html`: capacity vs number of variants, memory
  per loaded model, cold-start latency.
- Results land in `bench/results/<env>.jsonl` (committed) and the summary is regenerated.

## 6. Open decisions for the cloud phase
- Node type and count (start with the cheapest CPU node that fits 6 serving cores).
- Whether to include the GPU variants of F1 and F3 (after CPU results are repeatable).
- Whether the torch runtime is in scope for the family runs (not planned for now).
