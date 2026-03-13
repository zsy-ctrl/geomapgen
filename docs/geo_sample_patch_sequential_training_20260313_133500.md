# Geo Sample Patch Sequential Training 20260313_133500

## Requirement

Training should process one large map at a time:

- finish all continuous patches from one large image
- then move to the next large image

## What changed

The training and validation loaders now support sample-grouped sequential batching.

Current default for real training:

- `train.sample_patch_sequential = true`
- `train.val_sample_patch_sequential = true`

## Ordering behavior

Within each sample, records are ordered by:

1. `tile_index`
2. task order from config/task schema registration

So the effective traversal is:

- sample A, tile 0, tasks...
- sample A, tile 1, tasks...
- ...
- sample A, last tile, tasks...
- sample B, tile 0, tasks...

## Important note

This is intentionally different from random shuffling.

It improves consistency with the user's desired "one big image, all patches, then next image" training regime, but it also reduces stochastic mixing across images.
