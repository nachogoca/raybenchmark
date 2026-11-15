#!/usr/bin/env bash
# Tag and push the four raybench images to ECR (one repo per image).
# Then render manifests with: --registry "$REGISTRY/" --tag "$TAG"
set -euo pipefail
REGION="${REGION:-eu-central-1}"
TAG="${TAG:-local}"
ACCOUNT="${ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
REGISTRY="${REGISTRY:-$ACCOUNT.dkr.ecr.$REGION.amazonaws.com}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"
for img in raybench-py raybench-triton-ray raybench-models raybench-k6; do
  aws ecr describe-repositories --repository-names "$img" --region "$REGION" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "$img" --region "$REGION" >/dev/null
  docker tag "$img:$TAG" "$REGISTRY/$img:$TAG"
  docker push "$REGISTRY/$img:$TAG"
done
echo "REGISTRY=$REGISTRY"
