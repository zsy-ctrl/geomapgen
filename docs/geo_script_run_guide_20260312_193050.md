# Geo Script Run Guide 2026-03-12 19:30:50

## Defaults Changed In This Revision

The main launcher scripts now default these token limits to `0`:

- `PROMPT_MAX_TOKENS`
- `STATE_MAX_TOKENS`
- `TARGET_MAX_TOKENS`
- `MAX_NEW_TOKENS`
- `MIN_NEW_TOKENS`
- `MAX_PROP_TOKENS`

Meaning:

- For prompt/state/target tokenization: `0` means no proactive truncation in the tokenizer/collator.
- For generation: `MAX_NEW_TOKENS=0` means use the remaining model context budget.
- For property decoding grammar: `MAX_PROP_TOKENS=0` means do not hard-cap the property span.

## Scripts To Edit

- Training LoRA:
  [run_geo_lora_train.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_lora_train.sh)
  [run_geo_lora_train.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_lora_train.ps1)
- Training Full:
  [run_geo_full_train.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_full_train.sh)
  [run_geo_full_train.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_full_train.ps1)
- Predict:
  [run_geo_predict.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_predict.sh)
  [run_geo_predict.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_predict.ps1)
- Eval:
  [run_geo_eval.sh](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_eval.sh)
  [run_geo_eval.ps1](/c:/DevelopProject/VScode/UniMapGenStrongBaseline/scripts/run_geo_eval.ps1)

## Parameters You Can Change Directly In The Scripts

- `IMAGE_SIZE`
- `TILE_SIZE_PX`
- `OVERLAP_PX`
- `KEEP_MARGIN_PX`
- `STATE_BORDER_MARGIN_PX`
- `SAMPLE_INTERVAL_METER`
- `COORD_BINS`
- `LANE_MAX_FEATURES`
- `INTERSECTION_MAX_FEATURES`
- `STATE_MAX_FEATURES`
- `STATE_ANCHOR_MAX_POINTS`
- `PROMPT_MAX_TOKENS`
- `STATE_MAX_TOKENS`
- `TARGET_MAX_TOKENS`
- `MAX_NEW_TOKENS`
- `MIN_NEW_TOKENS`
- `MAX_PROP_TOKENS`
- `USE_KV_CACHE`

## Mask Semantics

- `train` and `eval` use the review mask.
- `predict` ignores the review mask and scans the full image.
