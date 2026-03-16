#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/geo_patch_v1_adapter}"
DATASET_ROOT="${DATASET_ROOT:-/home/zsy/Downloads/dataset-extracted}"
MODEL_PATH="${MODEL_PATH:-$ROOT/../ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PREFIX="${DATASET_PREFIX:-geomapgen_geo_patch_v1}"
TRAIN_DATASET_NAME="${DATASET_PREFIX}_train"
CONFIG_PATH="$OUTPUT_ROOT/qwen2_5vl_3b_lora_sft.yaml"

mkdir -p "$OUTPUT_ROOT"

echo "[V1Adapter] export dataset to $OUTPUT_ROOT"
"$PYTHON_BIN" "$ROOT/scripts/export_llamafactory_from_geo_patch_dataset.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --splits train val \
  --use-system-prompt \
  --dataset-prefix "$DATASET_PREFIX"

cat > "$CONFIG_PATH" <<EOF
### model
model_name_or_path: $MODEL_PATH
trust_remote_code: true
image_max_pixels: 1048576
video_max_pixels: 16384

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: $OUTPUT_ROOT
media_dir: $OUTPUT_ROOT
dataset: $TRAIN_DATASET_NAME
template: qwen2_vl
cutoff_len: 8192
val_size: 0.02
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 4

### output
output_dir: $OUTPUT_ROOT/llamafactory_qwen2_5vl_3b_lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000

### eval
eval_strategy: steps
eval_steps: 200
EOF

echo "[V1Adapter] train config written to $CONFIG_PATH"
"$LLAMAFACTORY_BIN" train "$CONFIG_PATH"
