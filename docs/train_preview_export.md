# Train Preview Export

The geo training loop can now export preview GeoJSON files after each epoch.

Enable it in `configs/geo_vector_lora.yaml` or `configs/geo_vector_full.yaml`:

```yaml
train:
  preview_export:
    enabled: true
    split: val
    max_samples: 1
    sample_ids: []
    output_subdir: preview_geojson
```

Behavior:

- `enabled`: turn per-epoch preview export on or off.
- `split`: dataset split used for preview inference. Default is `val`.
- `max_samples`: maximum number of samples exported each epoch. Use `0` for all samples.
- `sample_ids`: optional fixed sample ids. You can also provide a comma-separated string.
- `output_subdir`: subdirectory created under the current training run directory.

Outputs are written to:

- `outputs/<run>/preview_geojson/epoch_0001/<sample_id>/Lane.geojson`
- `outputs/<run>/preview_geojson/epoch_0001/<sample_id>/Intersection.geojson`
- `outputs/<run>/preview_geojson/epoch_0001/summary.json`

Notes:

- Preview export runs after checkpoint save and metrics logging for the epoch.
- It uses the current in-memory model weights and the existing `predict` tiling/decode settings.
- If preview export is enabled but the configured split or sample ids cannot be resolved, training exits with an error before the main loop starts.
