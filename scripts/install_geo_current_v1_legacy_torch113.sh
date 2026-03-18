#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-geo-current-v1-torch113}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_BIN="${CONDA_BIN:-conda}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
PYTORCH_SPEC="${PYTORCH_SPEC:-torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers}"
COMMON_PKGS="${COMMON_PKGS:-peft accelerate datasets pillow numpy pyproj rasterio pyyaml scipy sentencepiece einops}"
INSTALL_LLF="${INSTALL_LLF:-1}"

echo "[GeoCurrentV1] create conda env: $ENV_NAME"
"$CONDA_BIN" create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

CONDA_BASE="$("$CONDA_BIN" info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

echo "[GeoCurrentV1] install legacy torch stack: $PYTORCH_SPEC"
python -m pip install $PYTORCH_SPEC --extra-index-url "$PIP_EXTRA_INDEX_URL"

echo "[GeoCurrentV1] install common packages"
python -m pip install $TRANSFORMERS_SPEC $COMMON_PKGS

if [[ "$INSTALL_LLF" == "1" ]]; then
  echo "[GeoCurrentV1] install LLaMAFactory"
  python -m pip install llamafactory
fi

echo "[GeoCurrentV1] run preflight"
python "$ROOT/scripts/check_geo_current_v1_env.py" || true

cat <<EOF
[GeoCurrentV1] install finished.
env: $ENV_NAME

Next:
  conda activate $ENV_NAME
  python $ROOT/scripts/check_geo_current_v1_env.py --require-qwen25-vl

Note:
  This script targets a legacy torch 1.13 environment and forces training/rollout scripts to fall back to fp16.
  Qwen2.5-VL runtime support still depends on whether your installed transformers build can import Qwen2_5_VLForConditionalGeneration.
EOF
