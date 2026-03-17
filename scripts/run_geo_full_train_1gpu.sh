#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

bash "$SCRIPT_DIR/run_geo_full_train.sh"
