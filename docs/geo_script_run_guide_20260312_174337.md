# Geo Script Run Guide 20260312_174337

This document updates the run guide with the latest trusted-mask behavior.

## Two-TIF Input Assumption
Each sample is expected to provide:
- `patch_tif/0.tif`
  - full map GeoTIFF
- `patch_tif/0_edit_poly.tif`
  - black/white trusted-region mask GeoTIFF

Meaning of the mask:
- white pixels: trusted human-reviewed region
- black pixels: untrusted region

## Training Behavior With The Trusted Mask
Training currently uses the mask in these ways:
- train crop can be limited around the review mask bounding box
- train tiles with insufficient white-mask area are dropped
- empty train tiles are dropped
- train line targets are filtered to trusted mask regions

Relevant config keys:
- `data.train_crop_to_review_mask`
- `data.train_filter_features_by_review_mask`
- `data.feature_mask_min_inside_ratio`
- `tiling.train.min_review_ratio`
- `tiling.train.min_review_pixels`
- `tiling.train.fallback_to_all_if_empty`
- `tiling.train.allow_empty_tiles`

## Current Default Train Mask Settings
Current defaults in the YAML configs:

```yaml
data:
  train_crop_to_review_mask: true
  train_filter_features_by_review_mask: true
  feature_mask_min_inside_ratio: 0.5

tiling:
  train:
    min_review_ratio: 0.002
    min_review_pixels: 2048
    fallback_to_all_if_empty: false
    allow_empty_tiles: false
```

## How To Change Mask-Related Behavior
Edit either:
- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`

Recommended interpretations:
- increase `min_review_ratio` if too many black-heavy patches still survive
- increase `min_review_pixels` if tiny trusted slivers should be ignored
- keep `fallback_to_all_if_empty: false` if you do not want all-black samples to silently re-enter training
- keep `allow_empty_tiles: false` if you do not want blank target tiles in training
- increase `feature_mask_min_inside_ratio` if polygons/features should be more strictly inside white regions

## Token To GeoJSON Summary
Prediction/eval/training snapshot export all use the same structured decode chain:

```text
token ids
-> structured item decode
-> dequantized uv
-> absolute pixel features
-> GeoJSON export
```

The final output files written for users remain:
- `Lane.geojson`
- `Intersection.geojson`

## Artifact Export Reminder
The scripts already expose artifact switches at the top.

For training:
- `ArtifactExportEnabled`
- `SaveKeptPatches`
- `SaveDiscardedPatches`
- `SaveResizedPatchInputs`
- `SaveTrainBatchGeojson`
- `SaveValBatchGeojson`

For predict/eval:
- `ArtifactExportEnabled`
- `SaveKeptPatches`
- `SaveDiscardedPatches`
- `SaveResizedPatchInputs`
- `SavePredictTileGeojson`
- `SaveEvalSampleGeojson`

These control whether patch audit images, tile snapshots, and intermediate GeoJSON files are saved.
## Feature Count Parameters

The startup scripts expose:

- `LaneMaxFeatures`
- `IntersectionMaxFeatures`

Current rule:

- set either value to `0` to disable the explicit object-count cap

Important detail:

- `0` means "do not truncate by feature counter"
- it does not mean infinite effective capacity, because token limits still apply:
  - `TargetMaxTokens`
  - `MaxNewTokens`
  - model context length
