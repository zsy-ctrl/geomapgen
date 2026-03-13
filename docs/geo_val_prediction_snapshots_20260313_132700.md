# Geo Val Prediction Snapshots 20260313_132700

## What is now exported by default

To make training progress visible without reintroducing the highest OOM risk path, default LoRA training now exports:

- training inputs and GT snapshots
- validation inputs and GT snapshots
- validation prediction snapshots

Current default:

- `save_train_batch_predictions = false`
- `save_val_batch_predictions = true`
- `max_batches_per_epoch = 1`
- `max_samples_per_batch = 1`

## Where to look

Validation prediction snapshots are written under:

- `outputs/.../artifacts/val/epoch_XXXX/batch_YYYYY/.../Lane.pred.geojson`
- `outputs/.../artifacts/val/epoch_XXXX/batch_YYYYY/.../Intersection.pred.geojson`

## Why validation only

Generating prediction snapshots during training batches is more likely to destabilize a 16 GB setup.

Validation prediction snapshots are a better default compromise:

- you can inspect what the model currently predicts
- memory risk stays lower than enabling both train and val prediction snapshots
