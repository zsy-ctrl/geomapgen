# Geo Text IO Soft Prompt

Date: 2026-03-13

This update switches the geo training / inference path away from hard structured coordinate-token generation and into plain text GeoJSON generation.

## What changed

- Input state to the LLM is now text GeoJSON, not structured state tokens.
- Output target for supervision is now text GeoJSON, not quantized coordinate-token sequences.
- Prediction decoding now:
  - decodes plain text
  - extracts the first JSON object
  - coerces it into a FeatureCollection when possible
  - converts that GeoJSON back into pixel-space features

## Soft format control only

The previous grammar helper / structured token constraint path is no longer used in the active train / inference flow.

- `grammar_helper=None` during generation
- prompt examples are used as soft guidance
- no hard structured coordinate-token decoding is required for the main path

## Prompt examples

Prompt construction now injects compact multi-feature examples for:

- `Lane.geojson`
- `Intersection.geojson`

These are used only as soft format examples so the model sees:

- multiple lane features
- multiple intersection features
- valid FeatureCollection layout
- closed polygon rings for intersections

## GeoTIFF geospatial context

Prompts also include compact `GeoMeta` text:

- `crs`
- `origin`
- `pixel_size`
- `patch_px`
- `patch_world`

So the model can explicitly read patch location metadata from the GeoTIFF context.

## Important training note

Because the model target is now final GeoJSON text, spatial augmentation such as rotation / flip would break the consistency between image orientation and world-coordinate supervision.

Therefore the default was changed to:

- `train_augment: false`

## Compatibility note

Old checkpoints trained with the structured coordinate-token target should not be used to judge this new text-GeoJSON path.

The active semantics of the target sequence have changed materially.
