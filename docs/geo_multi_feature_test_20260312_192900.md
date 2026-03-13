# Geo Multi Feature Test 20260312 192900

## Purpose

Verify that the current structured-coordinate pipeline can preserve multiple objects in one GeoJSON `FeatureCollection`, not only a single lane or a single intersection.

This test checks the deterministic round-trip path:

`GeoJSON -> pixel features -> patch-local uv -> quantized coordinate tokens -> decode -> GeoJSON`

It does **not** measure learned prediction quality. It only verifies that the serialization / decode / export chain supports multiple features.

## Script

- Script: `scripts/test_geo_multi_feature_roundtrip.py`

## Command

```bash
python scripts/test_geo_multi_feature_roundtrip.py \
  --config configs/geo_vector_smoke_current.yaml \
  --split train \
  --output_dir outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_192500
```

## What The Script Does

1. Reads one sample from the configured dataset split.
2. Loads `Lane.geojson` and `Intersection.geojson`.
3. Converts them to internal pixel features.
4. Duplicates one base feature several times with small pixel offsets, producing a synthetic multi-feature sample.
5. Runs the full coordinate-token round-trip.
6. Writes:
   - synthetic input GeoJSON
   - round-trip GeoJSON
   - `report.json`

## Verified Result

Output directory:

- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_192500`

Key result:

- lane:
  - input features: 3
  - decoded items: 3
  - round-trip features: 3
- intersection:
  - input features: 3
  - decoded items: 3
  - round-trip features: 3

Files to inspect:

- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_192500/report.json`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_192500/Lane.roundtrip.geojson`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_192500/Intersection.roundtrip.geojson`

## Interpretation

The current pipeline can represent and export multiple lane features and multiple intersection features inside one `features` array.

This does **not** prove that a tiny smoke checkpoint will reliably predict multiple objects from imagery. It only proves that the internal representation and GeoJSON export path are not limited to a single object.
