# Geo OOM Root-Cause Check 20260313_130500

## Current conclusion

If `LoRA + batch_size=1 + image_size=128` still reports `CUDA out of memory` on a 16 GB GPU, the dominant pressure is no longer the image tensor. The remaining likely root causes are:

1. The Qwen base model is still being loaded at a higher precision than expected.
2. Validation is using more memory than training because it was not using autocast.
3. Actual token lengths in the first batch are still much larger than expected.
4. The remote machine is not running the same tightened config/code path.

## Changes applied

### Explicit LLM memory controls

Added explicit config/script controls for:

- `model.llm_torch_dtype`
- `model.attn_implementation`

Current low-memory defaults:

- `llm_torch_dtype: float16`
- `attn_implementation: sdpa`

### Validation memory reduction

`run_val()` now runs inside CUDA autocast when `train.amp=true`.

### Better startup diagnostics

Training startup now prints:

- GPU name and total VRAM
- CUDA allocated/reserved memory right after `model.to(device)`
- effective LLM dtype
- effective LLM parameter dtype
- `sat_proj` dtype

### Better first-batch diagnostics

At the first batch of each epoch, training now prints:

- image tensor shape
- prompt token lengths
- state token lengths
- target token lengths
- CUDA allocated/reserved/peak allocated/peak reserved after the first optimization step

## What to check on the remote machine

The new logs should include lines similar to:

- `[Init] CUDA after model.to allocated_gb=... reserved_gb=...`
- `[Init] LLM dtype=... llm_param_dtype=... sat_proj_dtype=...`
- `[Epoch 1] Batch0 image_shape=... prompt_lens=... state_lens=... target_lens=...`
- `[Epoch 1] Batch0 CUDA allocated_gb=... peak_alloc_gb=...`

If these lines are missing, the remote machine is still running an older code path.

## Next escalation if OOM still persists

If the new logs confirm:

- `llm_param_dtype=torch.float16`
- `image_size=128`
- small first-batch token lengths

and OOM still happens, the next meaningful step is no longer more resize reduction. The next effective options are:

1. QLoRA / 4-bit base model loading
2. smaller Qwen checkpoint
3. optional validation skipping during early bring-up
