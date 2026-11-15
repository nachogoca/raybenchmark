#!/usr/bin/env bash
# Create an EKS cluster via eksctl and install KubeRay + metrics-server (+ NVIDIA plugin if GPU=1).
# Costs money: run delete.sh the same day.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${CLUSTER:-raybench}"
REGION="${REGION:-eu-central-1}"
GPU="${GPU:-0}"

eksctl create cluster -f "$HERE/cluster.yaml"
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION"

helm repo add kuberay https://ray-project.github.io/kuberay-helm/ --force-update
helm repo update kuberay
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.7.0 \
  -n ray-system --create-namespace

# EKS has no metrics-server by default.
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

if [ "$GPU" = "1" ]; then
  eksctl scale nodegroup --cluster "$CLUSTER" --region "$REGION" --name gpu --nodes 1
  # The EKS GPU AMI ships the driver but NOT the device plugin: install it. Pin a version
  # from https://github.com/NVIDIA/k8s-device-plugin/releases (set NVIDIA_PLUGIN_VERSION).
  helm repo add nvdp https://nvidia.github.io/k8s-device-plugin --force-update
  helm upgrade --install nvdp nvdp/nvidia-device-plugin -n kube-system \
    ${NVIDIA_PLUGIN_VERSION:+--version "$NVIDIA_PLUGIN_VERSION"}
fi
kubectl apply -f "$HERE/../../k8s/namespace.yaml"
