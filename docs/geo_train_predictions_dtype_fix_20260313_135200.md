# Geo Train Predictions And Dtype Fix

Date: 2026-03-13

This update addressed two active issues in the `geomapgen` training path.

## 1. Training-time prediction snapshots

The LoRA training defaults now export model predictions during training so patch-level learning progress can be inspected directly.

Default behavior:

- `artifact_export.save_train_batch_predictions = true`
- `artifact_export.save_val_batch_predictions = true`
- `artifact_export.max_batches_per_epoch = 1`
- `artifact_export.max_samples_per_batch = 1`

This keeps the export volume small while still producing:

- `Lane.pred.geojson`
- `Intersection.pred.geojson`

under the epoch batch artifact folders.

## 2. Half/Float dtype mismatch fix

The older inference/training path could fail with:

- `GEO-1003 RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`

The main risk point was the visual prefix path:

- `sat_encoder` output could be `Half`
- `sat_proj` weights remained `Float`
- the projection was applied before explicit dtype alignment

The fix now aligns `sat_tokens` to `self.sat_proj.weight.dtype` before the linear projection in:

- `unimapgen/models/qwen_map_generator.py`

This keeps the projection input and weight dtype consistent before converting the projected tokens to the LLM embedding dtype.

## Files changed

- `unimapgen/models/qwen_map_generator.py`
- `configs/geo_vector_lora.yaml`
- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_lora_train.ps1`
