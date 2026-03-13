# Geo Multi Feature Test 20260312 193600

## Purpose

Run a larger synthetic round-trip test to verify that the current structured-coordinate pipeline can preserve more than a handful of objects inside one GeoJSON `FeatureCollection`.

This test validates:

`GeoJSON -> pixel -> uv -> quantized coordinate tokens -> decode -> GeoJSON`

It does not measure learned image-to-map quality. It verifies that the representation, serialization, decode, and GeoJSON export path can handle multiple lane and intersection objects.

## Script

- `scripts/test_geo_multi_feature_roundtrip.py`

## Command Used

```bash
python scripts/test_geo_multi_feature_roundtrip.py \
  --config configs/geo_vector_smoke_current.yaml \
  --split train \
  --lane_count 8 \
  --intersection_count 6 \
  --offset_step_x 3 \
  --offset_step_y 3 \
  --grid_cols 4 \
  --output_dir outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400
```

## Result

- lane:
  - input features: 8
  - decoded items: 8
  - round-trip features: 8
- intersection:
  - input features: 6
  - decoded items: 6
  - round-trip features: 6

## Output Files

- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400/report.json`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400/Lane.synthetic_input.geojson`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400/Lane.roundtrip.geojson`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400/Intersection.synthetic_input.geojson`
- `outputs/geo_vector_smoke/multi_feature_roundtrip_20260312_193400/Intersection.roundtrip.geojson`

## Interpretation

The current pipeline is not limited to a single lane or a single intersection per GeoJSON output. In this test it preserved:

- `8` lane features in one `Lane.geojson`
- `6` intersection features in one `Intersection.geojson`

This confirms that the internal coordinate-token representation and the final GeoJSON export support multiple objects in the same `features` array.
