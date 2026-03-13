# Geo Script Run Guide 20260312_172817

## Purpose
This guide explains how to launch training, prediction, and evaluation with the current scripts and how to modify their parameters.

## Main Scripts

PowerShell:
- `scripts/run_geo_lora_train.ps1`
- `scripts/run_geo_full_train.ps1`
- `scripts/run_geo_predict.ps1`
- `scripts/run_geo_eval.ps1`

Bash:
- `scripts/run_geo_lora_train.sh`
- `scripts/run_geo_full_train.sh`
- `scripts/run_geo_predict.sh`
- `scripts/run_geo_eval.sh`

## How To Modify Parameters
Each script has editable variables at the top of the file.

You usually change only these top variables:
- dataset path
- checkpoint path
- output path
- image/tile sizes
- token limits
- artifact export switches
- training hyperparameters

The scripts map those variables to environment variables, and the YAML config reads them through `${ENV:-default}`.

## Training

### LoRA Training
PowerShell:

```powershell
cd C:\DevelopProject\VScode\UniMapGenStrongBaseline
powershell -ExecutionPolicy Bypass -File .\scripts\run_geo_lora_train.ps1
```

Bash:

```bash
cd /path/to/UniMapGenStrongBaseline
bash ./scripts/run_geo_lora_train.sh
```

### Full Training
PowerShell:

```powershell
cd C:\DevelopProject\VScode\UniMapGenStrongBaseline
powershell -ExecutionPolicy Bypass -File .\scripts\run_geo_full_train.ps1
```

Bash:

```bash
cd /path/to/UniMapGenStrongBaseline
bash ./scripts/run_geo_full_train.sh
```

### Most Important Training Variables
- `Config`
- `DatasetRoot`
- `DinoModelPath`
- `QwenModelPath`
- `OutputDir`
- `RunName`
- `InitCheckpoint`
- `ImageSize`
- `TileSizePx`
- `OverlapPx`
- `KeepMarginPx`
- `StateBorderMarginPx`
- `SampleIntervalMeter`
- `CoordBins`
- `LaneMaxFeatures`
- `IntersectionMaxFeatures`
- `BatchSize`
- `ValBatchSize`
- `Epochs`
- `LearningRate`
- `PromptMaxTokens`
- `StateMaxTokens`
- `TargetMaxTokens`
- `MaxNewTokens`
- `UseKvCache`

### Artifact Export Variables For Training
- `ArtifactExportEnabled`
- `SaveKeptPatches`
- `SaveDiscardedPatches`
- `SaveResizedPatchInputs`
- `SaveTrainBatchGeojson`
- `SaveValBatchGeojson`
- `ArtifactMaxBatchesPerEpoch`
- `ArtifactMaxSamplesPerBatch`
- `ArtifactMaxPatchImagesPerSample`

### Training Outputs
Training run directory contains:
- `latest.pt`
- `best.pt`
- `metrics.jsonl`
- `config_snapshot.yaml`
- `run_meta.txt`
- `artifacts/train_patch_audit/...`
- `artifacts/val_patch_audit/...`
- `artifacts/train/epoch_xxxx/batch_xxxxx/...`
- `artifacts/val/epoch_xxxx/batch_xxxxx/...`

## Prediction

### PowerShell
Edit `scripts/run_geo_predict.ps1`, especially:
- `Checkpoint`
- either `InputImage`, `InputDir`, or `Split`
- `OutputDir`

Then run:

```powershell
cd C:\DevelopProject\VScode\UniMapGenStrongBaseline
powershell -ExecutionPolicy Bypass -File .\scripts\run_geo_predict.ps1
```

### Bash
Edit `scripts/run_geo_predict.sh`, then run:

```bash
cd /path/to/UniMapGenStrongBaseline
bash ./scripts/run_geo_predict.sh
```

### Prediction Variables
- `Checkpoint` is required.
- `InputImage`: predict one image.
- `InputDir`: predict all images matching `Glob`.
- `Split`: use dataset split input instead of ad hoc paths.
- `OutputDir`: final prediction root.

### Artifact Export Variables For Prediction
- `ArtifactExportEnabled`
- `SaveKeptPatches`
- `SaveDiscardedPatches`
- `SaveResizedPatchInputs`
- `SavePredictTileGeojson`
- `ArtifactMaxPatchImagesPerSample`

### Prediction Outputs
Per sample:
- `Lane.geojson`
- `Intersection.geojson`
- `Lane.raw_tiles.json`
- `Intersection.raw_tiles.json`
- `artifacts/patch_audit/...`
- `artifacts/tile_geojson/<task>/tile_xxxx.pred.geojson`
- `artifacts/tile_geojson/<task>/tile_xxxx.kept.geojson`

## Evaluation

### PowerShell
Edit `scripts/run_geo_eval.ps1`, especially:
- `Checkpoint`
- `Split`
- `Output`

Then run:

```powershell
cd C:\DevelopProject\VScode\UniMapGenStrongBaseline
powershell -ExecutionPolicy Bypass -File .\scripts\run_geo_eval.ps1
```

### Bash
Edit `scripts/run_geo_eval.sh`, then run:

```bash
cd /path/to/UniMapGenStrongBaseline
bash ./scripts/run_geo_eval.sh
```

### Evaluation Variables
- `Checkpoint` is required.
- `Split` chooses dataset split to evaluate.
- `Output` is the final metrics JSON path.
- `MaxSamples` can limit evaluation size.

### Artifact Export Variables For Evaluation
- `ArtifactExportEnabled`
- `SaveKeptPatches`
- `SaveDiscardedPatches`
- `SaveResizedPatchInputs`
- `SaveEvalSampleGeojson`
- `SavePredictTileGeojson`
- `ArtifactMaxPatchImagesPerSample`

### Evaluation Outputs
- metrics JSON at the `Output` path
- artifact directory next to it:

If `Output=outputs/geo_eval_metrics.json`, artifacts go to:

```text
outputs/geo_eval_metrics_artifacts/
```

Per sample:
- `Lane.pred.geojson`
- `Lane.gt.geojson`
- `Intersection.pred.geojson`
- `Intersection.gt.geojson`
- patch audit outputs
- tile-level prediction GeoJSON outputs

## Practical Parameter Notes

### If artifact export is too large
Reduce:
- `ArtifactMaxPatchImagesPerSample`
- `ArtifactMaxBatchesPerEpoch`
- `ArtifactMaxSamplesPerBatch`

Or disable some switches:
- `SaveDiscardedPatches=false`
- `SaveTrainBatchGeojson=false`
- `SavePredictTileGeojson=false`

### If one patch is too dense
Increase carefully:
- `LaneMaxFeatures`
- `IntersectionMaxFeatures`
- `TargetMaxTokens`
- `MaxNewTokens`

### If inference is too slow
Reduce carefully:
- `TileSizePx`
- `LaneMaxFeatures`
- `IntersectionMaxFeatures`
- `MaxNewTokens`

And verify:
- `UseKvCache=true`

## Relationship Between Script Variables And Config
Scripts do not directly replace the YAML file.
They set environment variables consumed by:
- `configs/geo_vector_lora.yaml`
- `configs/geo_vector_full.yaml`

So the normal workflow is:
1. Open the script you want to use.
2. Edit the variables at the top.
3. Run the script.

That is the intended control surface for day-to-day runs.
