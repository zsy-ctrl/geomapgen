# Geo Structured Token UV Plan

Timestamp: 2026-03-13 15:38

## Goal

Replace the current free-text GeoJSON training/prediction path with a structured coordinate-token path:

`GeoJSON -> patch-local uv -> quantized coordinate tokens -> model -> dequantize uv -> GeoJSON`

The goal is to improve:

- geometric stability
- parse stability
- cut-point continuity across patches
- large-map inference controllability

## Scope

This plan is for the `geomapgen` repo and is intended to become the main training / validation / inference path.

This is not a small patch. It changes:

- supervision format
- tokenizer / vocabulary
- dataset serialization
- inference decoding
- cut-point / state-update representation

## Non-Goals

- Do not train on whole-map stitched text directly.
- Do not rely on a second learned converter model for token-to-GeoJSON conversion.
- Do not encode absolute longitude / latitude directly as vocab tokens.

## Core Design

### 1. Coordinate Representation

Use patch-local coordinates, not global lon/lat, as the supervised geometry target.

For each patch:

- convert GT GeoJSON to raster pixel coordinates
- crop to current patch
- map to patch-local `u, v`
- quantize `u, v` into `coord_bins`

Recommended initial form:

- `u, v` each quantized to `[0, coord_bins - 1]`
- default initial `coord_bins`: `640`

### 2. Structured Vocabulary

Add atomic geometry tokens to the tokenizer vocabulary.

Minimum required token families:

- object structure
  - `<map_bos>`
  - `<obj>`
  - `<obj_end>`
- geometry type
  - `<line>`
  - `<poly>`
  - `<ring>`
  - `<pt>`
- source / continuity
  - `<src_local>`
  - `<src_state>`
  - `<cut_in_left>`
  - `<cut_in_top>`
  - `<cut_in_right>`
  - `<cut_in_bottom>`
  - `<cut_in_none>`
  - `<cut_out_left>`
  - `<cut_out_top>`
  - `<cut_out_right>`
  - `<cut_out_bottom>`
  - `<cut_out_none>`
- state anchors
  - `<anchor>`
  - `<anchor_end>`
  - `<side_left>`
  - `<side_top>`
  - `<side_right>`
  - `<side_bottom>`
  - `<side_none>`
- coordinates
  - `<coord_0000>` ... `<coord_0639>` or equivalent count

Optional later:

- structured property tokens for fixed schema fields

### 3. Supervision Target

Do not train Qwen to emit raw GeoJSON text.

Train it to emit structured map sequences, for example:

```text
<map_bos> <obj> <line> <src_local> <cut_in_left> <cut_out_right>
<pt> <coord_0123> <coord_0448>
<pt> <coord_0130> <coord_0455>
<prop> {"LaneType":14,...} <prop_end>
<obj_end>
```

Initial plan:

- geometry is structured tokens
- properties remain compact JSON span inside `<prop> ... <prop_end>`

Later hardening option:

- move properties to structured key/value tokens too

### 4. State Update

State update must be centered on cut-points and local increments, not full historical GeoJSON text.

For each patch:

- collect left/top neighboring boundary anchors from GT during training
- collect left/top predicted boundary anchors during inference
- feed only local boundary state, not the full previous patch contents

State sequence should carry:

- side
- local anchor points
- object type
- cut direction
- optional lightweight object id / continuation id

### 5. GeoJSON Export

The model does not need to output GeoJSON text directly.

Final output pipeline:

1. decode structured token sequence
2. dequantize `u, v`
3. recover patch-local pixel coordinates
4. map back to absolute raster pixel coordinates
5. stitch across patches
6. convert to final GeoJSON

This converter is deterministic code, not a second model.

## Required Pipeline Order

Training data for each patch must be built in this order:

1. load source GeoJSON
2. reorder GT features deterministically
3. apply trusted-mask filtering
4. crop to patch
5. mark crop / mask cut points
6. convert to patch-local `uv`
7. quantize `uv`
8. serialize to structured token target

Image path in parallel:

1. crop patch from tif
2. zero-out masked invalid pixels
3. resize / pad to model input size
4. send to DINO

## Reordering Rules

Need deterministic feature ordering before tokenization.

Recommended first-pass ordering:

- primary key: geometry type
- then by patch-local min-x
- then by min-y
- then by length / area
- then stable fallback by original index

Per-feature point ordering:

- lines: canonical direction
- polygons: canonical ring start
- polygons: outer ring first, inner rings after

## DINO -> Qwen Fusion

Keep the current high-level design:

- DINO image encoder
- linear projection `sat_proj` from DINO hidden size to Qwen hidden size
- projected visual tokens become a prefix to Qwen

Initial design remains:

- `SatelliteEncoder -> sat_proj -> prefix tokens`
- concatenate with prompt / state embeddings
- train Qwen as causal LM over structured map tokens

Do not change to cross-attention in phase 1.

Reason:

- prefix-token fusion already exists
- lower implementation risk
- easier migration from current code

## Patch and Whole-Map Training Semantics

Training semantics should remain:

- `1 epoch = 1 big map sample`
- all patches from that map are trained in sequence
- each patch computes one patch-level loss
- optimizer step is per patch

Whole-map stitched output after finishing all patches in the map:

- export stitched prediction
- compare to GT for monitoring
- do not backprop whole-map stitched loss in phase 1

Reason:

- stitched decode is non-differentiable
- patch-level CE loss is stable and already supported

## File-Level Implementation Plan

### Phase 1: Vocabulary and Serialization

Files:

- `unimapgen/geo/tokenizer.py`
- `unimapgen/geo/schema.py`
- `unimapgen/geo/coord_sequence.py`

Tasks:

- restore / harden structured token vocabulary
- define coordinate quantization / dequantization
- define structured object serialization
- keep compact prop JSON span inside token stream

Acceptance:

- can round-trip multiple lanes / multiple intersections without model involvement

### Phase 2: Dataset Supervision Rewrite

Files:

- `unimapgen/geo/dataset.py`
- `unimapgen/geo/prompting.py`
- `unimapgen/geo/io.py`

Tasks:

- remove pure-GeoJSON text target as main supervision
- output `state_items`, `target_items`, `map_input_ids`
- ensure cut-point labels are explicit in target
- ensure trusted-mask filtering happens before token target build

Acceptance:

- batch artifacts show structured target with cut markers
- GT patch export and structured target agree

### Phase 3: Model and Loss Path

Files:

- `unimapgen/models/qwen_map_generator.py`
- `unimapgen/train_geo_model.py`

Tasks:

- train on structured map token ids
- keep DINO -> `sat_proj` -> prefix path
- remove dependence on final GeoJSON text parsing for training

Acceptance:

- one-patch overfit can reconstruct target token sequence

### Phase 4: Inference Rewrite

Files:

- `unimapgen/geo/inference.py`
- `unimapgen/predict_geo_vector.py`
- `unimapgen/geo/artifacts.py`

Tasks:

- decode model output as structured sequence
- recover `uv`
- dequantize to absolute pixel coordinates
- stitch patch outputs
- export final GeoJSON
- keep raw token / decoded-item debug artifacts

Acceptance:

- no parse-failed dependence on free-text GeoJSON
- final output can contain multiple features reliably

### Phase 5: Evaluation

Files:

- `unimapgen/eval_geo_vector.py`
- `unimapgen/geo/metrics.py`

Tasks:

- evaluate final stitched GeoJSON as before
- add token-level decode success stats
- add per-patch / per-map continuity diagnostics

## Debug Outputs Required

For every stage keep these artifacts:

- raw patch image
- resized patch image
- trusted mask crop
- state token text dump
- target token text dump
- decoded prediction token text dump
- dequantized per-patch GeoJSON
- stitched whole-map GeoJSON
- parse / decode failure diagnostics

## Acceptance Tests

### A. Serializer Round-Trip

- multi-lane input survives:
  - GeoJSON -> tokens -> GeoJSON
- multi-intersection input survives
- polygon rings survive
- properties survive for known fields

### B. Golden Predict Replay

- inject fixed structured token sequence
- run full prediction export
- verify multiple lanes / intersections are written correctly

### C. One-Patch Overfit

- single patch
- target token sequence reproduced nearly exactly
- exported GeoJSON visually matches GT

### D. One-Map Overfit

- one big map with many patches
- patch predictions decode successfully
- stitched output is non-empty and topologically continuous enough to inspect

## Risks

- property JSON span can still be a weak point until fully structured
- very dense patches may exceed context if token budgets are too small
- polygon clipping / ring recovery needs careful testing
- backward compatibility with old checkpoints should not be assumed

## Recommended Migration Strategy

1. Keep current text-GeoJSON branch intact as fallback.
2. Implement structured-token branch behind a config switch.
3. Pass serializer round-trip and golden replay before any long training.
4. Run one-patch overfit before real training.
5. Only then switch the main train / predict scripts to the new branch.

## Decisions Already Implied By This Plan

- yes: structured coordinate tokens should be added to the vocabulary
- yes: GT GeoJSON should be preprocessed into patch-local `uv`
- yes: model output should be structured `uv` token sequences
- yes: final `.geojson` should be produced by deterministic postprocessing
- no: phase 1 should not use stitched whole-map loss for backprop

## Open Decisions For Later

- whether properties should remain compact JSON span or become fully structured tokens
- exact `coord_bins`
- whether to add lightweight continuation ids for stronger cross-patch stitching
- whether to keep prompt-side GeoJSON examples after moving fully to structured tokens
