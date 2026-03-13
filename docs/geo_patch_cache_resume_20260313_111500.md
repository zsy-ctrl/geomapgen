# Geo Patch Cache Resume 20260313 111500

## What Changed

Training can now reuse cached patch samples.

- If cache is enabled and a patch cache file already exists, the dataset loads it directly.
- If cache is missing, the dataset falls back to reading the original `0.tif`, cropping, resizing, and preparing target/state features.
- When cache writing is enabled, the newly prepared sample is written back to cache automatically.

The cache stores the pre-augmentation sample base:

- resized `image_chw`
- `raster_meta`
- `resize_ctx`
- `target_features_uv`
- `state_features_uv`

Random augmentation is still applied online, so resumed training keeps augmentation behavior.

## Config Keys

Added under `data`:

- `cache_enabled`
- `cache_write_enabled`
- `cache_dir`
- `cache_namespace`

## Quick Start

In `scripts/run_geo_lora_train.sh` or `scripts/run_geo_lora_train.ps1`:

- set `CACHE_ENABLED="true"`
- set `CACHE_WRITE_ENABLED="true"`
- set `CACHE_DIR="your_cache_directory"`

Recommended first run:

- `CACHE_ENABLED="true"`
- `CACHE_WRITE_ENABLED="true"`

Recommended resume run:

- `CACHE_ENABLED="true"`
- `CACHE_WRITE_ENABLED="true"`

This way existing cache is reused and any missing entries are filled automatically.

## Notes

- Cache keys include split, sample id, task name, tile index, crop bbox, image size, tile config, sample interval, mask settings, and state flags.
- Changing these settings produces different cache files instead of silently reusing stale data.
- Current cache is only used by training/validation dataset loading, not by `predict`.
