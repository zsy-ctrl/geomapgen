# Geo Raw Text Debug

Timestamp: 2026-03-13 15:13

## Purpose

When a prediction ends up as an empty GeoJSON or parse failure, keep the raw model text before JSON conversion so we can inspect what the model actually generated.

## New Outputs

### Predict sample outputs

Under each sample output directory:

- `Lane.raw_tiles.json`
- `Lane.raw_before_json.txt`
- `Intersection.raw_tiles.json`
- `Intersection.raw_before_json.txt`

`*.raw_before_json.txt` concatenates every tile's raw prediction text before `extract_first_json_object(...)` and GeoJSON conversion.

If parsing fails completely, `*.parse_failed.json` now also records:

- `raw_tiles_path`
- `raw_text_path`

### Predict tile artifact outputs

Under:

- `artifacts/tile_geojson/lane/`
- `artifacts/tile_geojson/intersection/`

Each tile now also saves:

- `tile_XXXX.pred.raw.txt`

This is the raw text emitted by the model for that tile before JSON extraction/conversion.

## How To Use

When final output is empty:

1. Open `*.raw_before_json.txt`
2. Check whether the model produced:
   - empty text
   - non-JSON text
   - partial JSON
   - valid GeoJSON with wrong structure
3. If needed, inspect per-tile files in `artifacts/tile_geojson/.../tile_XXXX.pred.raw.txt`

This separates:

- model generation failure
- JSON extraction failure
- GeoJSON conversion failure

## Scope

This change is debug-only output. It does not alter model training targets or prediction semantics.
