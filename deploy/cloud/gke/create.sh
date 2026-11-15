#!/usr/bin/env bash
# Create a GKE cluster: 2x n2-standard-8 (serving) + 1x n2-standard-4 (loadgen), install KubeRay.
# Costs money: run delete.sh the same day.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-europe-west4}"
ZONE="${ZONE:-europe-west4-a}"
CLUSTER="${CLUSTER:-raybench}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
  --release-channel regular --num-nodes 2 --machine-type n2-standard-8 \
  --node-labels raybench/role=serving --enable-ip-alias --no-enable-autoupgrade

gcloud container node-pools create loadgen --project "$PROJECT" --zone "$ZONE" \
  --cluster "$CLUSTER" --num-nodes 1 --machine-type n2-standard-4 \
  --node-labels raybench/role=loadgen --no-enable-autoupgrade

gcloud container clusters get-credentials "$CLUSTER" --project "$PROJECT" --zone "$ZONE"

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update
helm repo update kuberay
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.7.0 \
  -n ray-system --create-namespace

# metrics-server is built into GKE: nothing to install (`kubectl top` works after ~1 min).
kubectl apply -f "$HERE/../../k8s/namespace.yaml"
echo "next: ./push-images.sh, optionally ./add-gpu-pool.sh"
