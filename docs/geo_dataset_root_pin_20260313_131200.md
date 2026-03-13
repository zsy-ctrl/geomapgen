# Geo Dataset Root Pin 20260313_131200

## Default dataset root

The default real-dataset root for `geomapgen` has been pinned to:

`/home/zsy/Downloads/dataset-extracted`

This change was applied to the main real-data configs and launch scripts:

- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`
- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_lora_train.ps1`
- `scripts/run_geo_full_train.sh`
- `scripts/run_geo_full_train.ps1`
- `scripts/run_geo_predict.sh`
- `scripts/run_geo_predict.ps1`
- `scripts/run_geo_eval.sh`
- `scripts/run_geo_eval.ps1`

## Scope

This pin applies to the real train/eval/predict path only.

Smoke configs remain unchanged on purpose:

- `configs/geo_vector_smoke_current.yaml`
- `configs/geo_vector_smoke_multi.yaml`

Those files are still reserved for local smoke and synthetic tests.
