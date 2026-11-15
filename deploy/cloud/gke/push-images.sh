#!/usr/bin/env bash
# Tag and push the four raybench images to Artifact Registry.
# Then render manifests with: --registry "$REGISTRY/" --tag "$TAG"
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-europe-west4}"
REPO="${REPO:-raybench}"
TAG="${TAG:-local}"
REGISTRY="${REGISTRY:-$REGION-docker.pkg.dev/$PROJECT/$REPO}"

gcloud artifacts repositories describe "$REPO" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --project "$PROJECT" --location "$REGION" \
       --repository-format docker
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
for img in raybench-py raybench-triton-ray raybench-models raybench-k6; do
  docker tag "$img:$TAG" "$REGISTRY/$img:$TAG"
  docker push "$REGISTRY/$img:$TAG"
done
echo "REGISTRY=$REGISTRY"
