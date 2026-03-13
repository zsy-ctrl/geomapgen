# Paper Geo Rewrite Status

This repository now runs the GeoTIFF + `Lane.geojson` / `Intersection.geojson` pipeline on top of the original `QwenSatelliteMapGenerator` model framework.

## Current Runtime Entry Points

- `unimapgen/train_geo_lora.py`
- `unimapgen/train_geo_full.py`
- `unimapgen/train_geo_model.py`
- `unimapgen/predict_geo_vector.py`
- `unimapgen/eval_geo_vector.py`

## Current Runtime Model Path

- `unimapgen/models/qwen_map_generator.py`

The new geo pipeline now uses:

- image tokens from DINOv2
- prompt tokens from Qwen tokenizer
- state-update prefix tokens
- target GeoJSON text tokens

## State Update

The rewritten geo pipeline now includes patch-wise state update:

- training uses ground-truth state tokens from previously scanned patch borders
- inference scans tiles in raster order
- inference feeds previously predicted border state into the next tile

## Important Note About Older Docs

Several older code-reading documents still reference deleted files such as `unimapgen/models/qwen_geo_generator.py`, `unimapgen/check_geo_data.py`, and the earlier structured-token geo implementation. Those references are now historical and no longer describe the active runtime path.
