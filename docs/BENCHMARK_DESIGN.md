# Serving benchmark design

Goal: compare ways to serve ML models on a Kubernetes platform: FastAPI + ONNX Runtime,
Triton, Ray Serve + FastAPI, and Ray Serve + Triton. The benchmark records what each one
delivers; it does not presume a winner.

## Rules (apply to every stack)
- Same ONNX artifacts, same CPU budget per node, same k6 load script, same SLOs.
- Each stack runs a documented grid (batching on/off, ORT threads, replicas); the best config
  is reported, and every row records its settings.
- Single runs are search results. Any claim is re-run with repeats before it is reported.
- Negative results are reported as-is.

## Questions and pass/fail criteria
| # | Question | Scenario | Measured by |
|---|---|---|---|
| Q1 | Which stack has the best latency and capacity for one model? | S1 | capacity at p99 ≤ 20/50/100 ms |
| Q2 | How do stacks behave when many models share a cluster with skewed traffic? | S2 | capacity and memory per loaded model; first-request latency for a cold model |
| Q3 | What does a multi-stage pipeline cost on each stack? | S3 | p99 and cores at a fixed rate |
| Q4 | How quickly does each stack absorb a traffic burst? | S4 | time until p99 < SLO; errors during scale-up; core-seconds |
| Q5 | What happens when a serving pod fails under load? | S5 | error rate; recovery time |
| Q6 | How much of the gap between Ray and Triton is the HTTP front end? | S1 with Ray HAProxy on/off | capacity difference, in ladder steps |

## Scenarios
- **S1 single model:** DeepFM small, 200 rows per request, rate ladder.
- **S2 multi-model:** 8 model variants (4 DeepFM, 4 DCNv2, small) with Zipf traffic (one hot
  model, long tail). Each stack serves the same 8 models.
- **S3 pipeline:** CPU feature stage (decode, normalise) → model A → model B. Each stack
  implements the same three stages in its native way (FastAPI call chain, Triton ensemble,
  Ray composed deployments).
- **S4 burst:** 10× step in load for 2 minutes.
- **S5 failure:** kill one serving pod during steady load.

## Metrics
capacity at each SLO · p50 and p99 at capacity · cores busy · peak memory · memory per model (S2)
· core-seconds per 1M requests · time-to-recover (S4) · error rate (S5) · $ per 1M requests
(from `config/prices.yaml`).

## Kubernetes topology
- Ray Serve: `proxy_location: EveryNode`, the serve Service selects worker pods (the head has
  no replicas). The current `deploy/k8s/render.py` routes only through the head; this must be
  fixed before any cluster Ray result is valid.
- Triton and FastAPI: Deployment + Service, scaled by HPA (requests per second or CPU).
- Serving pods and the load generator run on separate nodes so they do not share cores.

## Open items
- Gateway logic placement and tradeoffs: see `docs/HAPROXY.md`.
- GPU variants of S1–S3, after CPU results are repeatable.
