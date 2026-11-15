#!/usr/bin/env bash
set -euo pipefail
eksctl delete cluster --name "${CLUSTER:-raybench}" --region "${REGION:-eu-central-1}" --wait
