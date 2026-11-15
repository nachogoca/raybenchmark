#!/usr/bin/env bash
# Add a single-L4 node (g2-standard-8) for the k8s-gpu env. Delete the cluster when done.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-europe-west4-a}"   # must offer L4 GPUs
CLUSTER="${CLUSTER:-raybench}"

gcloud container node-pools create gpu --project "$PROJECT" --zone "$ZONE" \
  --cluster "$CLUSTER" --num-nodes 1 --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --node-labels raybench/role=serving,raybench/accel=l4 --no-enable-autoupgrade
# GKE taints GPU nodes with nvidia.com/gpu=present:NoSchedule; rendered GPU pods tolerate it.
