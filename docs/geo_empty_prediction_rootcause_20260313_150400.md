# Geo Empty Prediction Root Cause 20260313_150400

## What was wrong

- The active training target had drifted away from the user's requirement.
- The model was being trained to emit extra metadata text before the final GeoJSON body.
- Prediction parsing only consumed the first valid JSON object.
- If the model failed to produce a valid GeoJSON object, the pipeline still wrote an empty FeatureCollection in some paths, which looked like a valid success but was not.

## Fixes applied

- `target_text` is now pure compact GeoJSON again.
- Cut-aware supervision metadata is still generated, but it is saved separately as `target_meta.txt` for inspection instead of being mixed into the output target text.
- `state_text` keeps explicit anchor metadata because it is model input context, not final output.
- Batch artifact export now always saves raw prediction text as:
  - `Lane.pred.raw.txt`
  - `Intersection.pred.raw.txt`
- If prediction parsing fails during training snapshot export, the code now writes:
  - `Lane.pred.parse_failed.json`
  - `Intersection.pred.parse_failed.json`
  instead of silently pretending the model produced a valid non-empty prediction.
- Final `predict` output no longer writes task GeoJSON for a task when `decoded_ok_count == 0`; it writes a `.parse_failed.json` marker and keeps the raw tile outputs.

## Current practical meaning

- If you still see empty final output after this change, it should no longer be ambiguous:
  - either the model explicitly produced an empty GeoJSON,
  - or the parse failed and the raw prediction text will be saved for inspection.

