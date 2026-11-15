# deploy/cloud

Pick one provider (GKE or EKS); the benchmark manifests (`deploy/k8s/render.py`) are identical.

| Script | Does |
|---|---|
| `gke/create.sh` | GKE: 2x n2-standard-8 serving + 1x n2-standard-4 loadgen, KubeRay, namespace |
| `gke/add-gpu-pool.sh` | GKE: 1x g2-standard-8 (L4) labeled `raybench/accel=l4` |
| `gke/push-images.sh` | tag + push the four raybench images to Artifact Registry |
| `gke/delete.sh` | delete the cluster (all pools) |
| `eks/cluster.yaml` | eksctl config: 2x m6i.2xlarge, 1x m6i.xlarge, GPU group g6.2xlarge (0 nodes) |
| `eks/create.sh` | create cluster, KubeRay, metrics-server; `GPU=1` scales GPU group + NVIDIA plugin |
| `eks/push-images.sh` | tag + push the four images to ECR |
| `eks/delete.sh` | delete the cluster |

All scripts read env vars (PROJECT/REGION/ZONE/CLUSTER/TAG/REGISTRY) with defaults at the top.
Flow: create -> push-images -> `render.py --registry "$REGISTRY/" --tag $TAG ... | kubectl apply -f -`.

Prices: after creating the cluster, look up the hourly on-demand price of the exact node
set (serving x2 + loadgen, and GPU node when used) and hand-copy it into `config/prices.yaml`
with the date you looked it up. Do not infer it from memory.

Tear down the same day: `gke/delete.sh` or `eks/delete.sh`. Check the console afterwards for
orphaned load balancers, disks and Artifact Registry/ECR storage.
