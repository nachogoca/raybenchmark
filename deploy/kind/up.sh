#!/usr/bin/env bash
# Create the kind cluster and install KubeRay, metrics-server, namespace, and images.
# Usage: TAG=local deploy/kind/up.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${TAG:-local}"
CLUSTER="${CLUSTER:-raybench}"

if kind get clusters | grep -qx "$CLUSTER"; then
  echo "kind cluster $CLUSTER exists, skipping create"
else
  kind create cluster --config "$HERE/cluster.yaml"
fi
kubectl config use-context "kind-$CLUSTER"

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update
helm repo update kuberay
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.7.0 \
  -n ray-system --create-namespace

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
if ! kubectl -n kube-system get deploy metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | grep -q kubelet-insecure-tls; then
  kubectl -n kube-system patch deploy metrics-server --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
fi

kubectl apply -f "$HERE/../k8s/namespace.yaml"

for img in raybench-py raybench-triton-ray raybench-models raybench-k6; do
  kind load docker-image "$img:$TAG" --name "$CLUSTER"
done
# Standalone Triton is large; load it too if present locally (otherwise the node pulls it).
if docker image inspect nvcr.io/nvidia/tritonserver:26.10-py3 >/dev/null 2>&1; then
  kind load docker-image nvcr.io/nvidia/tritonserver:26.10-py3 --name "$CLUSTER"
fi
kubectl -n ray-system rollout status deploy/kuberay-operator --timeout=180s
echo "cluster ready"
