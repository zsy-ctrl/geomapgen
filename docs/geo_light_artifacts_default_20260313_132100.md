# Geo Light Artifacts Default 20260313_132100

## Why intermediate artifacts were missing

To keep 16 GB training stable, training artifacts had previously been disabled entirely:

- `artifact_export.enabled = false`

That removed visibility into:

- selected/discarded patches
- resized inputs
- prompt/state/target intermediate assets

## New default behavior

The default LoRA training path now enables **lightweight** artifacts by default:

- patch audit enabled
- selected/discarded patch previews enabled
- resized patch previews enabled
- train batch input snapshots enabled
- val batch input snapshots enabled

But to avoid extra generation-time memory pressure during training:

- `save_train_batch_predictions = false`
- `save_val_batch_predictions = false`

## Current default caps

- `max_batches_per_epoch = 1`
- `max_samples_per_batch = 1`
- `max_patch_images_per_sample = 1`

This keeps visibility on what the model is being fed without reintroducing heavy decode-time OOM risk.
