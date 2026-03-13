# Geo OOM Safe LoRA Defaults 20260313 120100

## Reason

Training on the current geo pipeline hit `CUDA out of memory` under the previous LoRA defaults.

## Updated LoRA Defaults

- `data.image_size = 336`
- `serialization.tasks.lane.max_features = 64`
- `serialization.tasks.intersection.max_features = 32`
- `text.prompt_max_tokens = 128`
- `text.state_max_tokens = 512`
- `text.target_max_tokens = 1024`
- `state_update.max_features = 16`
- `model.gradient_checkpointing = true`
- `train.amp = true`

## Script Defaults Updated

Both launchers now match the safer defaults:

- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_lora_train.ps1`

## Notes

These defaults are conservative and intended to reduce OOM risk first.
If training is stable, they can be relaxed gradually.
