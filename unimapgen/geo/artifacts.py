from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from unimapgen.utils import ensure_dir

from .geometry import ResizeContext, build_resize_context
from .io import (
    RasterMeta,
    coerce_feature_collection,
    extract_first_json_object,
    geojson_dumps,
    geojson_to_pixel_features,
    pixel_features_to_geojson,
    read_raster_meta,
    read_rgb_geotiff,
    save_text,
)


def _as_bool(value, default: bool = False) -> bool:
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


def get_artifact_export_cfg(cfg: Dict) -> Dict:
    section = cfg.get("artifact_export", {})
    return {
        "enabled": _as_bool(section.get("enabled", False), default=False),
        "save_kept_patches": _as_bool(section.get("save_kept_patches", True), default=True),
        "save_discarded_patches": _as_bool(section.get("save_discarded_patches", True), default=True),
        "save_resized_patch_inputs": _as_bool(section.get("save_resized_patch_inputs", True), default=True),
        "save_train_batch_inputs": _as_bool(section.get("save_train_batch_inputs", True), default=True),
        "save_val_batch_inputs": _as_bool(section.get("save_val_batch_inputs", True), default=True),
        "save_train_batch_geojson": _as_bool(section.get("save_train_batch_geojson", True), default=True),
        "save_val_batch_geojson": _as_bool(section.get("save_val_batch_geojson", True), default=True),
        "save_train_batch_predictions": _as_bool(section.get("save_train_batch_predictions", False), default=False),
        "save_val_batch_predictions": _as_bool(section.get("save_val_batch_predictions", False), default=False),
        "save_eval_sample_geojson": _as_bool(section.get("save_eval_sample_geojson", True), default=True),
        "save_predict_tile_geojson": _as_bool(section.get("save_predict_tile_geojson", True), default=True),
        "max_batches_per_epoch": int(section.get("max_batches_per_epoch", 1)),
        "max_samples_per_batch": int(section.get("max_samples_per_batch", 1)),
        "max_patch_images_per_sample": int(section.get("max_patch_images_per_sample", 0)),
    }


def save_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def save_geojson_snapshot(path: str, task_schema, feature_records: Sequence[Dict], raster_meta) -> None:
    meta = raster_meta if isinstance(raster_meta, RasterMeta) else RasterMeta.from_dict(raster_meta)
    geojson_dict = pixel_features_to_geojson(
        task_schema=task_schema,
        feature_records=feature_records,
        raster_meta=meta,
    )
    save_text(path, geojson_dumps(geojson_dict))


def save_patch_preview(
    image_path: str,
    band_indices: Sequence[int],
    crop_bbox: Optional[Sequence[int]],
    raw_output_path: Optional[str] = None,
    resized_output_path: Optional[str] = None,
    image_size: Optional[int] = None,
) -> None:
    image_hwc, _ = read_rgb_geotiff(
        path=image_path,
        band_indices=[int(x) for x in band_indices],
        crop_bbox=None if crop_bbox is None else [int(v) for v in crop_bbox],
    )
    if raw_output_path:
        _save_hwc_png(raw_output_path, image_hwc)
    if resized_output_path and image_size is not None and int(image_size) > 0:
        raster_meta = read_raster_meta(image_path)
        resize_ctx = build_resize_context(
            width=int(raster_meta.width),
            height=int(raster_meta.height),
            target_size=int(image_size),
            crop_bbox=None if crop_bbox is None else tuple(int(v) for v in crop_bbox),
        )
        resized_chw = _resize_crop_to_square(crop_hwc=image_hwc, resize_ctx=resize_ctx)
        _save_chw_png(resized_output_path, resized_chw)


def export_tile_audit_records(
    audit_records: Sequence[Dict],
    output_dir: str,
    band_indices: Sequence[int],
    image_size: Optional[int],
    save_kept_patches: bool,
    save_discarded_patches: bool,
    save_resized_patch_inputs: bool,
    max_patch_images_per_sample: int = 0,
) -> None:
    ensure_dir(output_dir)
    save_json(os.path.join(output_dir, "tile_audit.json"), list(audit_records))

    per_sample_counts: Dict[str, int] = defaultdict(int)
    for record in audit_records:
        sample_id = str(record.get("sample_id", "sample"))
        selected = bool(record.get("selected", False))
        if selected and not bool(save_kept_patches):
            continue
        if (not selected) and not bool(save_discarded_patches):
            continue
        if int(max_patch_images_per_sample) > 0 and per_sample_counts[sample_id] >= int(max_patch_images_per_sample):
            continue
        image_path = str(record.get("image_path", "")).strip()
        if not image_path or not os.path.isfile(image_path):
            continue
        per_sample_counts[sample_id] += 1
        status_dir = "selected" if selected else "discarded"
        sample_out_dir = os.path.join(output_dir, sample_id, status_dir)
        ensure_dir(sample_out_dir)
        bbox = record.get("bbox")
        crop_bbox = None if bbox in (None, []) else [int(v) for v in bbox]
        tile_index = int(record.get("candidate_index", 0))
        reason = str(record.get("reason", "unknown"))
        stem = f"tile_{tile_index:04d}_{reason}"
        raw_path = os.path.join(sample_out_dir, f"{stem}.png")
        resized_path = (
            os.path.join(sample_out_dir, f"{stem}.resized.png")
            if bool(save_resized_patch_inputs) and image_size is not None and int(image_size) > 0
            else None
        )
        save_patch_preview(
            image_path=image_path,
            band_indices=band_indices,
            crop_bbox=crop_bbox,
            raw_output_path=raw_path,
            resized_output_path=resized_path,
            image_size=image_size,
        )


def export_batch_geojson_snapshots(
    cfg: Dict,
    task_schemas: Dict,
    text_tokenizer,
    model,
    batch: Dict,
    device,
    decode_cfg: Dict,
    output_dir: str,
    stage: str,
    epoch: int,
    batch_index: int,
) -> list[dict]:
    artifact_cfg = get_artifact_export_cfg(cfg)
    if not bool(artifact_cfg["enabled"]):
        return []
    if stage == "train" and not bool(artifact_cfg["save_train_batch_geojson"]):
        return []
    if stage == "val" and not bool(artifact_cfg["save_val_batch_geojson"]):
        return []

    save_inputs = bool(artifact_cfg["save_train_batch_inputs"]) if stage == "train" else bool(artifact_cfg["save_val_batch_inputs"])
    save_predictions = bool(artifact_cfg["save_train_batch_predictions"]) if stage == "train" else bool(artifact_cfg["save_val_batch_predictions"])

    max_samples_cfg = int(artifact_cfg["max_samples_per_batch"])
    max_samples = len(batch.get("sample_ids", [])) if max_samples_cfg <= 0 else min(int(max_samples_cfg), len(batch.get("sample_ids", [])))
    if max_samples <= 0:
        return []

    stage_out_dir = os.path.join(output_dir, "artifacts", stage, f"epoch_{int(epoch):04d}", f"batch_{int(batch_index):05d}")
    ensure_dir(stage_out_dir)
    save_json(
        os.path.join(stage_out_dir, "batch_meta.json"),
        {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "sample_ids": list(batch.get("sample_ids", [])),
            "task_names": list(batch.get("task_names", [])),
        },
    )

    was_training = bool(model.training)
    model.eval()
    exported_prediction_records: list[dict] = []
    try:
        for sample_index in range(min(int(max_samples), len(batch.get("sample_ids", [])))):
            sample_id = str(batch["sample_ids"][sample_index])
            task_name = str(batch["task_names"][sample_index])
            task_schema = task_schemas[task_name]
            raster_meta = RasterMeta.from_dict(batch["raster_metas"][sample_index])
            resize_ctx = ResizeContext.from_dict(batch["resize_ctxs"][sample_index])
            sample_out_dir = os.path.join(stage_out_dir, f"{sample_index:02d}_{sample_id}_{task_name}")
            ensure_dir(sample_out_dir)

            if save_inputs:
                raw_patch_path = os.path.join(sample_out_dir, "patch.png")
                resized_patch_path = os.path.join(sample_out_dir, "patch.resized.png")
                save_patch_preview(
                    image_path=str(batch["image_paths"][sample_index]),
                    band_indices=[int(x) for x in cfg["data"].get("band_indices", [1, 2, 3])],
                    crop_bbox=batch["crop_bboxes"][sample_index],
                    raw_output_path=raw_patch_path,
                    resized_output_path=resized_patch_path if artifact_cfg["save_resized_patch_inputs"] else None,
                    image_size=int(cfg["data"]["image_size"]),
                )
                save_text(os.path.join(sample_out_dir, "prompt.txt"), str(batch["prompt_texts"][sample_index]))
                save_text(os.path.join(sample_out_dir, "state.txt"), str(batch.get("state_texts", [""])[sample_index]))
                save_text(
                    os.path.join(sample_out_dir, f"{task_schema.collection_name}cut.txt"),
                    str(batch.get("state_texts", [""])[sample_index]),
                )
                save_text(os.path.join(sample_out_dir, "target.txt"), str(batch.get("target_texts", [""])[sample_index]))
                save_text(
                    os.path.join(sample_out_dir, "target_meta.txt"),
                    str(batch.get("target_meta_texts", [""])[sample_index]),
                )
                save_json(os.path.join(sample_out_dir, "state_items.json"), batch["state_items_list"][sample_index])
                save_json(os.path.join(sample_out_dir, "target_items.json"), batch["target_items_list"][sample_index])

                target_feature_records_list = batch.get("target_feature_records_list", [])
                gt_features_abs = (
                    target_feature_records_list[sample_index]
                    if sample_index < len(target_feature_records_list)
                    else []
                )
                save_geojson_snapshot(
                    path=os.path.join(sample_out_dir, f"{task_schema.collection_name}.gt.geojson"),
                    task_schema=task_schema,
                    feature_records=gt_features_abs,
                    raster_meta=raster_meta,
                )
                state_feature_records_list = batch.get("state_feature_records_list", [])
                cut_features_abs = (
                    state_feature_records_list[sample_index]
                    if sample_index < len(state_feature_records_list)
                    else []
                )
                save_geojson_snapshot(
                    path=os.path.join(sample_out_dir, f"{task_schema.collection_name}cut.pred.geojson"),
                    task_schema=task_schema,
                    feature_records=cut_features_abs,
                    raster_meta=raster_meta,
                )

            if save_predictions:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                use_amp = bool((cfg or {}).get("train", {}).get("amp", False)) and device.type == "cuda"
                with torch.inference_mode():
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred_ids = model.generate(
                            image=batch["image"][sample_index : sample_index + 1].to(device),
                            prompt_input_ids=batch["prompt_input_ids"][sample_index : sample_index + 1].to(device),
                            prompt_attention_mask=batch["prompt_attention_mask"][sample_index : sample_index + 1].to(device),
                            pv_images=None,
                            state_input_ids=batch["state_input_ids"][sample_index : sample_index + 1].to(device),
                            state_attention_mask=batch["state_attention_mask"][sample_index : sample_index + 1].to(device),
                            max_new_tokens=int(decode_cfg.get("max_new_tokens", 512)),
                            min_new_tokens=int(decode_cfg.get("min_new_tokens", 8)),
                            temperature=float(decode_cfg.get("temperature", 1.0)),
                            top_k=int(decode_cfg.get("top_k", 1)),
                            repetition_penalty=float(decode_cfg.get("repetition_penalty", 1.0)),
                            grammar_helper=None,
                            use_kv_cache=_as_bool(decode_cfg.get("use_kv_cache", True), default=True),
                            return_token_meta=False,
                        )
                pred_token_ids = pred_ids[0].detach().cpu().tolist()
                pred_text = text_tokenizer.decode_text(pred_token_ids)
                pred_geojson = coerce_feature_collection(
                    task_schema=task_schema,
                    obj=extract_first_json_object(pred_text),
                )
                pred_features_abs = (
                    geojson_to_pixel_features(
                        geojson_dict=pred_geojson,
                        task_schema=task_schema,
                        raster_meta=raster_meta,
                    )
                    if pred_geojson is not None
                    else []
                )
                save_text(os.path.join(sample_out_dir, f"{task_schema.collection_name}.pred.raw.txt"), pred_text)
                if pred_geojson is not None:
                    save_geojson_snapshot(
                        path=os.path.join(sample_out_dir, f"{task_schema.collection_name}.pred.geojson"),
                        task_schema=task_schema,
                        feature_records=pred_features_abs,
                        raster_meta=raster_meta,
                    )
                    exported_prediction_records.append(
                        {
                            "sample_id": sample_id,
                            "task_name": task_name,
                            "tile_index": int(batch["tile_indices"][sample_index]),
                            "crop_bbox": batch["crop_bboxes"][sample_index],
                            "raster_meta": raster_meta,
                            "feature_records": pred_features_abs,
                        }
                    )
                else:
                    save_json(
                        os.path.join(sample_out_dir, f"{task_schema.collection_name}.pred.parse_failed.json"),
                        {
                            "parsed_geojson": False,
                            "pred_feature_count": int(len(pred_features_abs)),
                            "pred_text": pred_text,
                        },
                    )
                save_json(
                    os.path.join(sample_out_dir, "prediction_debug.json"),
                    {
                        "decode_info": {
                            "parsed_geojson": bool(pred_geojson is not None),
                            "pred_feature_count": int(len(pred_features_abs)),
                        },
                        "token_ids": [int(x) for x in pred_token_ids],
                        "pred_text": pred_text,
                        "pred_item_count": int(len(pred_features_abs)),
                    },
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return exported_prediction_records


def export_prediction_tile_geojsons(
    cfg: Dict,
    sample_id: str,
    image_path: str,
    pred_result: Dict,
    output_dir: str,
) -> None:
    artifact_cfg = get_artifact_export_cfg(cfg)
    if not bool(artifact_cfg["enabled"]) or not bool(artifact_cfg["save_predict_tile_geojson"]):
        return

    ensure_dir(output_dir)
    export_tile_audit_records(
        audit_records=[
            {
                "sample_id": str(sample_id),
                "image_path": str(image_path),
                **record,
            }
            for record in pred_result.get("tile_audit", [])
        ],
        output_dir=os.path.join(output_dir, "patch_audit"),
        band_indices=[int(x) for x in cfg["data"].get("band_indices", [1, 2, 3])],
        image_size=int(cfg["data"]["image_size"]),
        save_kept_patches=bool(artifact_cfg["save_kept_patches"]),
        save_discarded_patches=bool(artifact_cfg["save_discarded_patches"]),
        save_resized_patch_inputs=bool(artifact_cfg["save_resized_patch_inputs"]),
        max_patch_images_per_sample=int(artifact_cfg["max_patch_images_per_sample"]),
    )

    tiles_out_dir = os.path.join(output_dir, "tile_geojson")
    ensure_dir(tiles_out_dir)
    for task_name, tile_records in pred_result.get("raw_outputs", {}).items():
        task_out_dir = os.path.join(tiles_out_dir, str(task_name))
        ensure_dir(task_out_dir)
        for tile_record in tile_records:
            tile_index = int(tile_record.get("tile_index", 0))
            pred_geojson = tile_record.get("pred_geojson")
            kept_geojson = tile_record.get("kept_geojson")
            pred_text = str(tile_record.get("pred_text", ""))
            save_text(
                os.path.join(task_out_dir, f"tile_{tile_index:04d}.pred.raw.txt"),
                pred_text,
            )
            if isinstance(pred_geojson, dict):
                save_text(
                    os.path.join(task_out_dir, f"tile_{tile_index:04d}.pred.geojson"),
                    geojson_dumps(pred_geojson),
                )
            if isinstance(kept_geojson, dict):
                save_text(
                    os.path.join(task_out_dir, f"tile_{tile_index:04d}.kept.geojson"),
                    geojson_dumps(kept_geojson),
                )


def export_eval_sample_geojsons(
    cfg: Dict,
    sample_id: str,
    image_path: str,
    pred_result: Dict,
    task_schemas: Dict,
    gt_by_task: Dict[str, Sequence[Dict]],
    output_dir: str,
) -> None:
    artifact_cfg = get_artifact_export_cfg(cfg)
    if not bool(artifact_cfg["enabled"]) or not bool(artifact_cfg["save_eval_sample_geojson"]):
        return

    ensure_dir(output_dir)
    export_prediction_tile_geojsons(
        cfg=cfg,
        sample_id=sample_id,
        image_path=image_path,
        pred_result=pred_result,
        output_dir=output_dir,
    )
    raster_meta = pred_result["raster_meta"]
    for task_name, task_schema in task_schemas.items():
        pred_features = pred_result.get("task_predictions", {}).get(task_name, [])
        gt_features = gt_by_task.get(task_name, [])
        save_geojson_snapshot(
            path=os.path.join(output_dir, f"{task_schema.collection_name}.pred.geojson"),
            task_schema=task_schema,
            feature_records=pred_features,
            raster_meta=raster_meta,
        )
        save_geojson_snapshot(
            path=os.path.join(output_dir, f"{task_schema.collection_name}.gt.geojson"),
            task_schema=task_schema,
            feature_records=gt_features,
            raster_meta=raster_meta,
        )


def _save_hwc_png(path: str, image_hwc: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    arr = np.asarray(np.clip(image_hwc, 0.0, 255.0), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _save_chw_png(path: str, image_chw: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    arr = np.asarray(image_chw, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected CHW image, got shape={arr.shape}")
    arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.asarray(np.clip(arr, 0.0, 255.0), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _resize_crop_to_square(crop_hwc: np.ndarray, resize_ctx: ResizeContext) -> np.ndarray:
    crop_u8 = np.asarray(np.clip(crop_hwc, 0.0, 255.0), dtype=np.uint8)
    pil = Image.fromarray(crop_u8)
    pil = pil.resize((resize_ctx.resized_width, resize_ctx.resized_height), Image.BILINEAR)
    canvas = np.zeros((resize_ctx.target_size, resize_ctx.target_size, 3), dtype=np.float32)
    resized = np.asarray(pil, dtype=np.float32)
    canvas[
        resize_ctx.pad_y : resize_ctx.pad_y + resize_ctx.resized_height,
        resize_ctx.pad_x : resize_ctx.pad_x + resize_ctx.resized_width,
    ] = resized
    return np.transpose(canvas / 255.0, (2, 0, 1)).astype(np.float32)


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
