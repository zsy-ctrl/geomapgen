# Geo Patch Text Cut State 20260313_145200

## This update changes

- Default `image_size` is now `256` for the main train, eval, and predict configs/scripts.
- The active training/inference path is still text-based GeoJSON, not hard grammar-constrained coordinate tokens.
- `state_text` now contains explicit cut-anchor metadata lines before the state GeoJSON body.
- `target_text` now contains explicit cut metadata lines before the target GeoJSON body.
- Training can now optimize once per large-image sample while still computing patch-level losses for each patch.

## DINO to Qwen bridge

- Satellite imagery is encoded by `SatelliteEncoder`, which loads DINO and produces pooled patch tokens.
- The DINO token tensor is projected into the Qwen hidden size by `sat_proj`, a linear layer in `unimapgen/models/qwen_map_generator.py`.
- The projected visual tokens are concatenated with prompt tokens and state tokens before the Qwen causal LM forward pass.
- Current bridge design:
  - `sat_tokens = sat_encoder(image)`
  - cast to `sat_proj.weight.dtype`
  - `sat_tokens = sat_proj(sat_tokens)`
  - cast to Qwen embedding dtype
  - concatenate with prompt/state embeddings

## Cut-aware text supervision

- Patch supervision still uses cropped, mask-filtered local geometry.
- The cropped supervision is no longer only compact GeoJSON text.
- `target_text` now has this structure:
  - `PatchTargetMeta ...`
  - zero or more `CutFeature ...` lines
  - `GeoJSON:`
  - compact GeoJSON FeatureCollection
- `state_text` now has this structure:
  - `StateAnchorMeta ...`
  - zero or more `StateAnchor ...` lines
  - `StateGeoJSON:`
  - compact GeoJSON FeatureCollection

## What is already true

- Patch images are cropped before training.
- Review-mask filtering happens before local text supervision is built.
- Line features carry explicit cut flags through crop/mask clipping.
- Polygon features carry `clipped` state through crop/mask clipping.
- Training order is sample-sequential: one large image's patch sequence is processed before the next sample.
- With `optimize_per_sample=true`, gradients accumulate across all patch batches of one large image and one optimizer step is applied after the full sample is processed.

## What is not yet implemented

- There is not yet a differentiable stitched whole-map loss that:
  - autoregressively predicts all patches,
  - stitches the whole map,
  - compares stitched full-map text against the original full-map text,
  - backpropagates that stitched-text loss.
- Current "large-image granularity" optimization is the accumulated mean patch loss over one sample, stepped once per sample.

