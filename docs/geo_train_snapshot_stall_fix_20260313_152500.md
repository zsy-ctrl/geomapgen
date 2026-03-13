# Geo Train Snapshot Stall Fix

Timestamp: 2026-03-13 15:25

## Symptom

Training repeatedly appeared to hang around early patch progress such as `2/102`.

## Root Cause

Training artifact export was still running `model.generate()` inside the train loop for multiple patch batches.

Even though the tqdm bar showed patch progress, the loop had not actually finished that iteration yet because it was still exporting prediction snapshots.

This made the progress bar appear frozen for a long time.

## Fix

### Train loop guard

`unimapgen/train_geo_model.py`

Train-time artifact export is now limited by:

- `artifact_export.max_batches_per_epoch`

The train loop now respects that limit before calling `export_batch_geojson_snapshots(...)`.

### Safer defaults

`configs/geo_vector_lora.yaml`

- `artifact_export.save_train_batch_predictions` now defaults to `false`

`scripts/run_geo_lora_train.sh`
`scripts/run_geo_lora_train.ps1`

- training prediction export defaults to `false`

This keeps:

- train inputs / GT artifacts
- val prediction artifacts

while avoiding slow autoregressive generation inside every train patch step.

### Visibility when manually enabled

If train prediction snapshots are re-enabled manually, the train loop now prints:

- when train prediction snapshot export starts
- when it finishes

so the process no longer looks like a silent hang.

## Recommended Use

For normal training on limited VRAM:

- keep train prediction snapshots disabled
- inspect model outputs from validation artifacts instead

If patch-level train predictions are needed for debugging, enable them temporarily and keep:

- `artifact_export.max_batches_per_epoch = 1`
