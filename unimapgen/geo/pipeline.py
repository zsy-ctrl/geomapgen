from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch

from unimapgen.utils import ensure_dir

from .dataset import GeoVectorCollator, GeoVectorDataset, GeoVectorDatasetConfig
from .errors import raise_geo_error, wrap_geo_error
from .schema import TaskSchema, load_task_schemas
from .tokenizer import GeoCoordTokenizer


def _parse_cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def load_geo_task_schemas(cfg: Dict) -> Dict[str, TaskSchema]:
    return load_task_schemas(cfg.get("serialization", {}))


def select_enabled_task_schemas(cfg: Dict, task_schemas: Dict[str, TaskSchema]) -> Dict[str, TaskSchema]:
    task_enable_cfg = cfg.get("data", {}).get("task_enable", {})
    enabled = {}
    for name, schema in task_schemas.items():
        flag = task_enable_cfg.get(name, True)
        if str(flag).strip().lower() == "false":
            continue
        enabled[name] = schema
    return enabled


def get_stage_tiling_cfg(cfg: Dict, stage: str) -> Dict:
    tiling_cfg = cfg.get("tiling", {})
    stage_cfg = tiling_cfg.get(stage, {})
    default_enabled = bool(stage in {"train", "eval", "predict"})
    return {
        "enabled": bool(stage_cfg.get("enabled", default_enabled)),
        "tile_size_px": int(stage_cfg.get("tile_size_px", 2048)),
        "overlap_px": int(stage_cfg.get("overlap_px", 256)),
        "keep_margin_px": int(stage_cfg.get("keep_margin_px", max(0, int(stage_cfg.get("overlap_px", 256)) // 2))),
        "min_review_ratio": float(stage_cfg.get("min_review_ratio", 0.0 if stage != "train" else 0.002)),
        "min_review_pixels": int(stage_cfg.get("min_review_pixels", 0 if stage != "train" else 1024)),
        "fallback_to_all_if_empty": bool(stage_cfg.get("fallback_to_all_if_empty", stage != "train")),
        "search_within_review_bbox": bool(stage_cfg.get("search_within_review_bbox", stage == "train")),
        "allow_empty_tiles": bool(stage_cfg.get("allow_empty_tiles", True)),
        "max_tiles_per_sample": stage_cfg.get("max_tiles_per_sample"),
    }


def build_geo_dataset(
    cfg: Dict,
    split: str,
    task_schemas: Dict[str, TaskSchema],
    max_samples=None,
    train_augment: bool = False,
    crop_to_review_mask: Optional[bool] = None,
    stage: str = "train",
):
    data_cfg = cfg["data"]
    enabled_tasks = list(select_enabled_task_schemas(cfg=cfg, task_schemas=task_schemas).keys())
    task_to_label_relpath = {}
    label_cfg = data_cfg.get("label_relpaths", {})
    for task_name in enabled_tasks:
        if task_name not in label_cfg:
            raise_geo_error("GEO-1103", f"data.label_relpaths missing task={task_name}")
        task_to_label_relpath[task_name] = str(label_cfg[task_name])

    tiling_stage_cfg = get_stage_tiling_cfg(cfg=cfg, stage=stage)
    state_cfg = cfg.get("state_update", {})
    prompt_cfg = cfg.get("prompt", {})
    dataset_cfg = GeoVectorDatasetConfig(
        stage=str(stage),
        dataset_root=str(data_cfg["dataset_root"]),
        split=str(split),
        image_relpath=str(data_cfg["image_relpath"]),
        review_mask_relpath=str(data_cfg.get("review_mask_relpath")) if data_cfg.get("review_mask_relpath") else None,
        task_to_label_relpath=task_to_label_relpath,
        image_size=int(data_cfg["image_size"]),
        band_indices=[int(x) for x in data_cfg.get("band_indices", [1, 2, 3])],
        mask_threshold=int(data_cfg.get("mask_threshold", 127)),
        crop_to_review_mask=bool(
            data_cfg.get("crop_to_review_mask", True) if crop_to_review_mask is None else crop_to_review_mask
        ),
        review_crop_pad_px=int(data_cfg.get("review_crop_pad_px", 32)),
        sample_interval_meter=float(cfg["serialization"]["sample_interval_meter"])
        if cfg.get("serialization", {}).get("sample_interval_meter") is not None
        else None,
        max_samples=max_samples,
        train_augment=bool(train_augment),
        aug_rot90_prob=float(data_cfg.get("aug_rot90_prob", 0.0)) if train_augment else 0.0,
        aug_hflip_prob=float(data_cfg.get("aug_hflip_prob", 0.0)) if train_augment else 0.0,
        aug_vflip_prob=float(data_cfg.get("aug_vflip_prob", 0.0)) if train_augment else 0.0,
        tiling_enabled=bool(tiling_stage_cfg["enabled"]),
        tile_size_px=int(tiling_stage_cfg["tile_size_px"]),
        tile_overlap_px=int(tiling_stage_cfg["overlap_px"]),
        tile_keep_margin_px=int(tiling_stage_cfg["keep_margin_px"]),
        tile_min_mask_ratio=float(tiling_stage_cfg["min_review_ratio"]),
        tile_min_mask_pixels=int(tiling_stage_cfg["min_review_pixels"]),
        tile_fallback_to_all_if_empty=bool(tiling_stage_cfg["fallback_to_all_if_empty"]),
        tile_search_within_review_bbox=bool(tiling_stage_cfg["search_within_review_bbox"]),
        tile_allow_empty=bool(tiling_stage_cfg["allow_empty_tiles"]),
        tile_max_per_sample=tiling_stage_cfg.get("max_tiles_per_sample"),
        feature_filter_by_review_mask=bool(data_cfg.get("train_filter_features_by_review_mask", True)) if str(stage) == "train" else bool(data_cfg.get("eval_filter_features_by_review_mask", False)),
        feature_mask_min_inside_ratio=float(data_cfg.get("feature_mask_min_inside_ratio", 0.5)),
        state_enabled=bool(state_cfg.get("enabled", True)),
        state_border_margin_px=int(state_cfg.get("border_margin_px", 96)),
        state_max_features=int(state_cfg.get("max_features", 32)),
        state_anchor_max_points=int(state_cfg.get("anchor_max_points", 6)),
        prompt_with_state=str(
            prompt_cfg.get(
                "with_state_suffix",
                "Previous cut-point state anchors are provided below. Continue local increments only.",
            )
        ).strip(),
        prompt_without_state=str(
            prompt_cfg.get(
                "without_state_suffix",
                "No previous cut-point state anchors are available for this patch.",
            )
        ).strip(),
        prompt_include_geospatial_context=_parse_cfg_bool(
            prompt_cfg.get("include_geospatial_context", True),
            default=True,
        ),
        prompt_geospatial_precision=int(prompt_cfg.get("geospatial_precision", 3)),
        cache_enabled=_parse_cfg_bool(data_cfg.get("cache_enabled", False), default=False),
        cache_write_enabled=_parse_cfg_bool(data_cfg.get("cache_write_enabled", True), default=True),
        cache_dir=str(data_cfg.get("cache_dir", "")).strip() or None,
        cache_namespace=str(data_cfg.get("cache_namespace", "geo_patch_cache_v1")).strip(),
    )
    selected_task_schemas = {name: task_schemas[name] for name in enabled_tasks}
    return GeoVectorDataset(
        cfg=dataset_cfg,
        task_schemas=selected_task_schemas,
    )


def build_geo_components(cfg: Dict):
    task_schemas = load_geo_task_schemas(cfg)
    text_tokenizer = GeoCoordTokenizer(
        qwen_model_path=str(cfg["model"]["qwen_model_path"]),
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        trust_remote_code=True,
        coord_bins=int(cfg.get("serialization", {}).get("coord_bins", 1024)),
    )
    from unimapgen.models.qwen_map_generator import QwenSatelliteMapGenerator

    model_cfg = cfg["model"]
    model = QwenSatelliteMapGenerator(
        dino_model_path=str(model_cfg["dino_model_path"]),
        qwen_model_path=str(model_cfg["qwen_model_path"]),
        vocab_size=int(text_tokenizer.vocab_size),
        allowed_map_token_ids=list(text_tokenizer.allowed_output_token_ids),
        map_eos_token_id=int(text_tokenizer.eos_token_id),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        freeze_satellite=bool(model_cfg.get("freeze_satellite", True)),
        freeze_llm=bool(model_cfg.get("freeze_llm", False)),
        llm_train_mode=str(model_cfg.get("llm_train_mode", "full")),
        lora_r=int(model_cfg.get("lora", {}).get("r", 16)),
        lora_alpha=int(model_cfg.get("lora", {}).get("alpha", 32)),
        lora_dropout=float(model_cfg.get("lora", {}).get("dropout", 0.05)),
        lora_target_modules=list(model_cfg.get("lora", {}).get("target_modules", [])),
        sat_token_hw=tuple(model_cfg.get("sat_token_hw", [4, 4])),
        sat_patch_size=int(model_cfg.get("sat_patch_size", 14)),
        sat_drop_cls_token=bool(model_cfg.get("sat_drop_cls_token", True)),
        sat_normalize_input=bool(model_cfg.get("sat_normalize_input", True)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        llm_torch_dtype=str(model_cfg.get("llm_torch_dtype", "float16")),
        attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
    )
    collator = GeoVectorCollator(
        map_tokenizer=text_tokenizer,
        image_size=int(cfg["data"]["image_size"]),
        prompt_max_tokens=cfg.get("text", {}).get("prompt_max_tokens"),
        state_max_tokens=cfg.get("text", {}).get("state_max_tokens"),
        target_max_tokens=cfg.get("text", {}).get("target_max_tokens"),
    )
    return task_schemas, text_tokenizer, collator, model


def compute_shift_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    if int(valid.sum().item()) == 0:
        return 0, 0
    pred = shift_logits.argmax(dim=-1)
    correct = int((pred.eq(shift_labels) & valid).sum().item())
    total = int(valid.sum().item())
    return correct, total


def maybe_load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> Dict:
    if not checkpoint_path:
        return {}
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(state, dict):
            model_state = model.state_dict()
            filtered_state = {}
            skipped_shape = []
            for key, value in state.items():
                if key not in model_state:
                    filtered_state[key] = value
                    continue
                if tuple(model_state[key].shape) != tuple(value.shape):
                    skipped_shape.append(
                        {
                            "key": str(key),
                            "checkpoint_shape": tuple(value.shape),
                            "model_shape": tuple(model_state[key].shape),
                        }
                    )
                    continue
                filtered_state[key] = value
            state = filtered_state
        else:
            skipped_shape = []
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[Init] Loaded checkpoint={checkpoint_path} "
            f"(missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped_shape)})",
            flush=True,
        )
        if skipped_shape:
            preview = skipped_shape[:8]
            print(f"[Init] Skipped shape-mismatched keys: {preview}", flush=True)
        return ckpt if isinstance(ckpt, dict) else {}
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1405",
            message=f"failed to load checkpoint: {checkpoint_path}",
            exc=exc,
        )


def maybe_resume_training_state(
    optimizer,
    scaler,
    checkpoint_obj: Dict,
    load_optimizer: bool,
    load_scaler: bool,
) -> Dict:
    state = {"epoch": 0, "global_step": 0, "best_val": None}
    if not isinstance(checkpoint_obj, dict):
        return state
    try:
        if load_optimizer and optimizer is not None and "optimizer" in checkpoint_obj:
            optimizer.load_state_dict(checkpoint_obj["optimizer"])
        if load_scaler and scaler is not None and "scaler" in checkpoint_obj and checkpoint_obj["scaler"] is not None:
            scaler.load_state_dict(checkpoint_obj["scaler"])
        state["epoch"] = int(checkpoint_obj.get("epoch", 0))
        state["global_step"] = int(checkpoint_obj.get("global_step", 0))
        if checkpoint_obj.get("best_val") is not None:
            state["best_val"] = float(checkpoint_obj["best_val"])
        return state
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1406",
            message="failed to restore optimizer/scaler training state from checkpoint",
            exc=exc,
        )


def make_output_dir(train_cfg: Dict, resume_checkpoint: str) -> str:
    explicit = str(train_cfg.get("output_dir", "")).strip()
    if explicit:
        ensure_dir(explicit)
        return explicit
    resume_in_place = bool(train_cfg.get("resume_in_place", False))
    if resume_checkpoint and resume_in_place:
        out_dir = os.path.dirname(os.path.abspath(resume_checkpoint))
        ensure_dir(out_dir)
        return out_dir
    base_dir = str(train_cfg.get("base_output_dir", "outputs/geo_vector"))
    run_name = str(train_cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, run_name)
    ensure_dir(out_dir)
    return out_dir


def save_json(path: str, obj) -> None:
    try:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1106",
            message=f"failed to save JSON output: {path}",
            exc=exc,
        )


def atomic_torch_save(obj, path: str) -> None:
    try:
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1107",
            message=f"failed to atomically save checkpoint: {path}",
            exc=exc,
        )


def build_checkpoint_obj(
    model: torch.nn.Module,
    optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_val,
    cfg: Dict,
) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": None if best_val is None else float(best_val),
        "cfg": cfg,
    }
