# Geo All Patch Artifacts And Cut Exports

Timestamp: 2026-03-13 17:22:04

## Change

Training artifact export now keeps every patch-level training snapshot instead of only the first batch in each epoch.

This is enabled by treating:

- `artifact_export.max_batches_per_epoch <= 0`

as:

- export all batches in the epoch

## New / Clarified Outputs

For each patch batch under:

- `outputs/.../artifacts/train/epoch_XXXX/batch_YYYYY/...`

you will now see:

- `Lane.gt.geojson`
- `Intersection.gt.geojson`
- `Lane.pred.geojson`
- `Intersection.pred.geojson`
- `Lane.pred.raw.txt`
- `Intersection.pred.raw.txt`
- `Lanecut.pred.geojson`
- `Intersectioncut.pred.geojson`
- `Lanecut.txt`
- `Intersectioncut.txt`

Notes:

- `*cut.pred.geojson` is exported from the patch state / cut-anchor feature records used for that patch.
- `*cut.txt` is the plain-text state input used by the model for that patch.

## Stitched Whole-Map Outputs

After one map sample finishes an epoch, the stitched directory now contains:

- `Lane.stitched.pred.geojson`
- `Intersection.stitched.pred.geojson`
- `Lane.geojson`
- `Intersection.geojson`

The new `Lane.geojson` / `Intersection.geojson` are duplicates of the stitched predictions with simpler names for inspection.

## Default Script Behavior

`run_geo_lora_train.sh` and `run_geo_lora_train.ps1` now default to:

- `SAVE_TRAIN_BATCH_PREDICTIONS=true`
- `ARTIFACT_MAX_BATCHES_PER_EPOCH=0`

which means:

- keep train prediction artifacts for every patch

## Tradeoff

This is much heavier than the previous debug-light setup.

Expected impact:

- slower training
- more disk usage
- more generated artifacts per epoch
