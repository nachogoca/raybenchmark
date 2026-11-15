#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-europe-west4-a}"
CLUSTER="${CLUSTER:-raybench}"
gcloud container clusters delete "$CLUSTER" --project "$PROJECT" --zone "$ZONE" --quiet
