# GeoTIFF Prompt Context

Date: 2026-03-13

This update makes the model explicitly read geospatial metadata from each GeoTIFF patch during both training and inference.

## Previous behavior

Before this change, GeoTIFF geospatial metadata was used for:

- projecting GeoJSON into raster pixel coordinates
- converting predictions back to GeoJSON coordinates

But the model prompt itself did not explicitly include:

- CRS
- origin
- pixel size
- patch pixel bbox
- patch world bbox

So the model could not directly condition on that geospatial context as text.

## New behavior

Training and inference prompts now prepend a compact `GeoMeta` string containing:

- `crs`
- `origin`
- `pixel_size`
- `patch_px`
- `patch_world`

This is produced by:

- `unimapgen/geo/prompting.py`

And is used by:

- `unimapgen/geo/dataset.py`
- `unimapgen/geo/inference.py`

## Config

Added under `prompt`:

- `include_geospatial_context`
- `geospatial_precision`

Defaults:

- `include_geospatial_context = true`
- `geospatial_precision = 3`

## LoRA prompt budget

Because the prompt now carries extra `GeoMeta` content, the LoRA training default prompt budget was raised from:

- `64 -> 96`

to reduce the chance that geospatial context gets truncated out of the prompt.
