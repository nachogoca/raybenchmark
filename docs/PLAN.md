# Cloud benchmark plan (GKE and EKS): planned, not yet run

Extends the laptop benchmark (`docs/BENCHMARK_DESIGN.md`) to a Kubernetes fleet. The scenarios,
SLOs, MST rule and validity rules are identical, so laptop and cloud results sit side by side in
one report (`just report` per environment).

## 0. Before running anything
- Fix `deploy/k8s/render.py`: Ray Serve with `proxy_location: EveryNode`, Service selecting worker
  pods, head without replicas. Without this, cluster Ray numbers are invalid.
- `just models payloads build`, push images (`deploy/cloud/{gke,eks}/push-images.sh`).
- Node pools: a serving pool (one CPU node type) and a separate load-generator pool; serving pods
  and k6 never share nodes. Pods use requests == limits (Guaranteed QoS, `pod_sizing` in the matrix).
- Fill `config/prices.yaml` with the cluster's real on-demand rates (set `verified: true`).
- Record in every row: node type, cores, memory, Kubernetes and Ray versions.

## 1. Clusters
| | GKE | EKS |
|---|---|---|
| Serving pool | n2-standard-8 | m6i.2xlarge |
| Load-gen pool | c2-standard-8 | c6i.2xlarge |
| GPU option (later) | g2-standard-8 (L4) | g6.2xlarge (L4) |
Scripts: `deploy/cloud/gke/*`, `deploy/cloud/eks/*`. Same vCPU per node on both clouds.

## 2. Stacks
FastAPI + ORT, Triton, Ray Serve + ORT/FastAPI, Ray → Triton (gRPC), Ray + in-process Triton
(all as on the laptop), plus **KServe**: raw-deployment mode, Triton runtime serving the same
ONNX files, HPA; the family scenario as one InferenceService per variant (and ModelMesh as a
variant if time permits). Ray runs as a KubeRay `RayService` with HAProxy on every node.

## 3. Fleet scenarios (in addition to S1–R2 on the cluster)
| ID | Question | What runs |
|---|---|---|
| C1 | Capacity vs nodes | 1/2/4/8 serving nodes, MST per stack; scaling efficiency and $/1M req |
| C2 | Scale-out under a 10× burst | HPA (RPS and CPU) vs Ray autoscaler + cluster autoscaler vs KServe autoscaler; time to p99 < SLO, time to a new node, errors, core-seconds |
| C3 | Node failure | Drain/kill a serving node under steady load: errors, recovery |
| C4 | Rolling update | Deploy a new model version under load: dropped requests, p99 during the rollout (Ray in-place update vs Deployment rolling update vs KServe canary) |
| C5 | Cross-cloud cost | Same grid on GKE and EKS: $/1M req at each SLO |
| C6 | Family at scale | 256-variant family: memory density, cold variant, cache hit rate vs replicas |

## 4. Run order and gates
1. Smoke on each cluster (`just smoke` equivalent against the cluster env).
2. S1 on all stacks, 3 repetitions. Re-run any result that differs from the laptop ranking by more than one grid step, and find out why.
3. S1_scale, I1, F1–F5, A1, R1, R2 as on the laptop.
4. C1–C6.
A run is valid only if there are no unexplained errors, k6 stays below 80 % of its cores, and
serving pods did not restart (except where a scenario kills them).

## 5. Estimated spend
Each full pass is ~9 h of 2–8 serving nodes plus two load-generator nodes. Compute the exact figure
from `config/prices.yaml` before launching; budget one full pass per cloud plus one re-run.

## 6. Open decisions
- Whether to include the GPU variants of S1, F1 and F3 once CPU results are repeatable.
- Whether to include the PyTorch runtime (not planned).
