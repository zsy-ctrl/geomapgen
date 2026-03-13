# Geo Memory Risk Log 20260313 120900

## What Was Added

Training startup now prints a memory risk summary before the first epoch.

## Startup Log Includes

- CUDA device name
- total GPU VRAM in GB when CUDA is available
- estimated memory risk level: `low`, `medium`, or `high`
- a simple numeric risk score
- the main reasons contributing to the score
- practical suggestions when the configuration looks risky

## Current Heuristics

The risk estimate looks at:

- `llm_train_mode`
- `data.image_size`
- `text.prompt_max_tokens`
- `text.state_max_tokens`
- `text.target_max_tokens`
- `serialization.tasks.lane.max_features`
- `serialization.tasks.intersection.max_features`
- `state_update.max_features`
- `train.batch_size`
- `train.amp`
- `model.gradient_checkpointing`

## Purpose

This is only a startup warning and debugging aid.
It does not change training behavior by itself.
