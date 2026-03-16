from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .coord_sequence import uv_feature_records_to_state_items
from .geometry import (
    audit_tile_window_selection,
    annotate_tile_windows_with_mask,
    build_resize_context,
    clip_feature_to_bbox,
    clip_polygon_rings_to_bbox,
    compute_mask_bbox,
    expand_bbox,
    feature_center_inside_bbox,
    feature_intersects_bbox,
    generate_tile_windows,
    resample_feature_points,
)
from .metrics import deduplicate_feature_records
from .pipeline import get_stage_tiling_cfg
from .prompting import build_state_text, build_task_prompt_text
from .schema import TaskSchema
from .io import (
    NON_TRAINING_PROPERTY_KEYS,
    assign_incremental_feature_ids,
    coerce_feature_collection,
    extract_first_json_object,
    geojson_dumps_compact,
    geojson_to_pixel_features,
    pixel_features_to_geojson,
    pixel_features_to_uv_geojson,
    read_binary_mask,
    read_raster_meta,
    read_rgb_geotiff,
    strip_non_training_fields_from_feature_collection,
    uv_geojson_to_pixel_features,
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


def prompt_text_for_task(
    cfg: Dict,
    task_schema: TaskSchema,
    task_name: str,
    has_state: bool,
    raster_meta,
    crop_bbox: Optional[Sequence[int]],
    companion_task_name: str = "",
    companion_geojson_text: str = "",
) -> str:
    prompt_cfg = cfg.get("prompt", {})
    return build_task_prompt_text(
        task_name=str(task_name),
        base_prompt=str(task_schema.prompt_template).strip(),
        has_state=bool(has_state),
        with_state_suffix=str(
            prompt_cfg.get(
                "with_state_suffix",
                "Previous cut-point state anchors are provided below. Continue local increments only.",
            )
        ).strip(),
        without_state_suffix=str(
            prompt_cfg.get(
                "without_state_suffix",
                "No previous cut-point state anchors are available for this patch.",
            )
        ).strip(),
        raster_meta=raster_meta,
        crop_bbox=crop_bbox,
        include_geospatial_context=_as_bool(prompt_cfg.get("include_geospatial_context", True), default=True),
        geospatial_precision=int(prompt_cfg.get("geospatial_precision", 3)),
        companion_task_name=str(companion_task_name),
        companion_geojson_text=str(companion_geojson_text),
    )


def _paired_task_name(task_name: str, task_schemas: Dict[str, TaskSchema]) -> str:
    key = str(task_name).strip().lower()
    if key == "lane" and "intersection" in task_schemas:
        return "intersection"
    if key == "intersection" and "lane" in task_schemas:
        return "lane"
    return ""


def _ordered_task_names(task_schemas: Dict[str, TaskSchema]) -> List[str]:
    preferred = ["intersection", "lane"]
    ordered: List[str] = []
    seen = set()
    for name in preferred + list(task_schemas.keys()):
        key = str(name).strip().lower()
        if key in task_schemas and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _infer_sample_dir_from_image(cfg: Dict, image_path: str) -> str:
    rel_parts = [part for part in os.path.normpath(str(cfg["data"]["image_relpath"])).split(os.sep) if part]
    sample_dir = os.path.abspath(image_path)
    for _ in rel_parts:
        sample_dir = os.path.dirname(sample_dir)
    return sample_dir


def _load_optional_review_mask(cfg: Dict, image_path: str, stage: str) -> Optional[np.ndarray]:
    if str(stage).strip().lower() == "predict":
        return None
    review_relpath = str(cfg["data"].get("review_mask_relpath", "")).strip()
    if not review_relpath:
        return None
    sample_dir = _infer_sample_dir_from_image(cfg=cfg, image_path=image_path)
    review_mask_path = os.path.join(sample_dir, review_relpath)
    if not os.path.isfile(review_mask_path):
        return None
    return read_binary_mask(path=review_mask_path, threshold=int(cfg["data"].get("mask_threshold", 127)))


def _build_tile_windows_for_inference(
    cfg: Dict,
    image_path: str,
    raster_meta,
    stage: str,
) -> Tuple[List[Optional[Dict]], List[Dict[str, bool]], Optional[np.ndarray], List[Dict]]:
    tiling_cfg = get_stage_tiling_cfg(cfg=cfg, stage=stage)
    review_mask = _load_optional_review_mask(cfg=cfg, image_path=image_path, stage=stage)
    review_bbox = compute_mask_bbox(review_mask) if review_mask is not None else None
    region_bbox = None
    if bool(tiling_cfg["search_within_review_bbox"]) and review_bbox is not None:
        region_bbox = expand_bbox(
            bbox=review_bbox,
            pad_px=int(cfg["data"].get("review_crop_pad_px", 32)),
            width=int(raster_meta.width),
            height=int(raster_meta.height),
        )
    if not bool(tiling_cfg["enabled"]):
        return [None], [{"left": False, "top": False}], review_mask, [
            {
                "candidate_index": 0,
                "selected": True,
                "reason": "tiling_disabled",
                "bbox": None,
                "keep_bbox": None,
                "mask_ratio": 0.0,
                "mask_pixels": 0,
            }
        ]
    tile_windows = generate_tile_windows(
        width=int(raster_meta.width),
        height=int(raster_meta.height),
        tile_size_px=int(tiling_cfg["tile_size_px"]),
        overlap_px=int(tiling_cfg["overlap_px"]),
        region_bbox=None if region_bbox is None else tuple(int(v) for v in region_bbox),
        keep_margin_px=int(tiling_cfg["keep_margin_px"]),
    )
    tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
    tile_windows, tile_audits = audit_tile_window_selection(
        tile_windows=tile_windows,
        min_mask_ratio=float(tiling_cfg["min_review_ratio"]),
        min_mask_pixels=int(tiling_cfg["min_review_pixels"]),
        max_tiles=tiling_cfg.get("max_tiles_per_sample"),
        fallback_to_all_if_empty=bool(tiling_cfg.get("fallback_to_all_if_empty", True)),
    )
    tile_windows = sorted(tile_windows, key=lambda window: (int(window.y0), int(window.x0)))
    tile_windows_dict = [window.to_dict() for window in tile_windows]
    neighbors: List[Dict[str, bool]] = []
    seen_rows = set()
    seen_cols = set()
    for tile_window in tile_windows_dict:
        row_key = int(tile_window["y0"])
        col_key = int(tile_window["x0"])
        neighbors.append({"left": row_key in seen_rows, "top": col_key in seen_cols})
        seen_rows.add(row_key)
        seen_cols.add(col_key)
    return tile_windows_dict, neighbors, review_mask, tile_audits


def _resize_image_crop(crop_hwc: np.ndarray, resize_ctx) -> np.ndarray:
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


def _state_region_bboxes(
    crop_bbox: Optional[Sequence[int]],
    border_margin_px: int,
    use_left: bool,
    use_top: bool,
) -> List[Tuple[int, int, int, int]]:
    if crop_bbox is None:
        return []
    x0, y0, x1, y1 = [int(v) for v in crop_bbox]
    margin = max(1, int(border_margin_px))
    out: List[Tuple[int, int, int, int]] = []
    if bool(use_left):
        out.append((int(x0), int(y0), int(min(x1, x0 + margin)), int(y1)))
    if bool(use_top):
        out.append((int(x0), int(y0), int(x1), int(min(y1, y0 + margin))))
    return out


def _collect_state_features(
    state_memory: Sequence[Dict],
    task_schema: TaskSchema,
    crop_bbox: Optional[Sequence[int]],
    raster_meta,
    state_region_bboxes: Sequence[Tuple[int, int, int, int]],
    sample_interval_meter: Optional[float],
    max_features: int,
) -> List[Dict]:
    if not state_region_bboxes:
        return []
    out: List[Dict] = []
    max_features_limit = int(max_features)
    pixel_size_meter = max(abs(float(raster_meta.pixel_size_x)), 1e-6)
    interval_px = None
    if sample_interval_meter is not None and float(sample_interval_meter) > 0:
        interval_px = float(sample_interval_meter) / pixel_size_meter
    closed = bool(task_schema.geometry_type == "polygon")
    crop_bbox_tuple = None if crop_bbox is None else tuple(int(v) for v in crop_bbox)
    for feature in state_memory:
        if task_schema.geometry_type == "polygon" and feature.get("rings"):
            rings_original = [
                np.asarray(ring, dtype=np.float32)
                for ring in feature.get("rings", [])
                if np.asarray(ring, dtype=np.float32).ndim == 2
            ]
            if not rings_original or rings_original[0].shape[0] < int(task_schema.min_points_per_feature):
                continue
            mask_points = np.concatenate(rings_original, axis=0)
            if crop_bbox_tuple is not None and not feature_intersects_bbox(mask_points, crop_bbox_tuple):
                continue
            candidate_groups = [rings_original] if crop_bbox_tuple is None else clip_polygon_rings_to_bbox(
                rings_xy=rings_original,
                bbox=crop_bbox_tuple,
            )
            for candidate_group in candidate_groups:
                if not candidate_group or candidate_group[0].shape[0] < int(task_schema.min_points_per_feature):
                    continue
                state_groups = [candidate_group]
                if state_region_bboxes:
                    state_groups = []
                    for state_bbox in state_region_bboxes:
                        if not feature_intersects_bbox(np.concatenate(candidate_group, axis=0), state_bbox):
                            continue
                        state_groups.extend(
                            clip_polygon_rings_to_bbox(
                                rings_xy=candidate_group,
                                bbox=state_bbox,
                            )
                        )
                for state_group in state_groups:
                    if not state_group or state_group[0].shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    if interval_px is not None:
                        resampled_group = []
                        for ring in state_group:
                            resampled_ring = resample_feature_points(
                                points_xy=ring,
                                interval_px=float(interval_px),
                                max_points=max(4, int(ring.shape[0] * 2)),
                                closed=True,
                            )
                            if resampled_ring.shape[0] >= int(task_schema.min_points_per_feature):
                                resampled_group.append(resampled_ring.astype(np.float32))
                        state_group = resampled_group
                    if not state_group or state_group[0].shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    out.append(
                        {
                            "properties": dict(feature.get("properties", {})),
                            "points": state_group[0].astype(np.float32),
                            "rings": [ring.astype(np.float32) for ring in state_group],
                        }
                    )
                    if max_features_limit > 0 and len(out) >= max_features_limit:
                        return out
            continue
        pts_original = np.asarray(feature.get("points", []), dtype=np.float32)
        if pts_original.ndim != 2 or pts_original.shape[0] < int(task_schema.min_points_per_feature):
            continue
        if crop_bbox_tuple is not None and not feature_intersects_bbox(pts_original, crop_bbox_tuple):
            continue
        pieces = [pts_original] if crop_bbox_tuple is None else clip_feature_to_bbox(
            points_xy=pts_original,
            bbox=crop_bbox_tuple,
            geometry_type=task_schema.geometry_type,
        )
        for piece in pieces:
            for state_bbox in state_region_bboxes:
                if not feature_intersects_bbox(piece, state_bbox):
                    continue
                state_pieces = clip_feature_to_bbox(
                    points_xy=piece,
                    bbox=state_bbox,
                    geometry_type=task_schema.geometry_type,
                )
                for state_piece in state_pieces:
                    if state_piece.shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    if interval_px is not None:
                        state_piece = resample_feature_points(
                            points_xy=state_piece,
                            interval_px=float(interval_px),
                            max_points=max(4, int(state_piece.shape[0] * 2)),
                            closed=closed,
                        )
                    out.append({"properties": dict(feature.get("properties", {})), "points": state_piece.astype(np.float32)})
                    if max_features_limit > 0 and len(out) >= max_features_limit:
                        return out
    return out


def _feature_records_to_uv(feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
    out = []
    for feature in feature_records:
        record = {
            "properties": dict(feature.get("properties", {})),
            "points": points_abs_to_uv(np.asarray(feature.get("points", []), dtype=np.float32), resize_ctx=resize_ctx),
        }
        if feature.get("rings"):
            record["rings"] = rings_abs_to_uv(feature.get("rings", []), resize_ctx=resize_ctx)
        out.append(record)
    return out


def _retain_predictions_for_keep_bbox(
    feature_records: Sequence[Dict],
    keep_bbox: Optional[Sequence[int]],
) -> List[Dict]:
    if keep_bbox is None:
        return [dict(feature) for feature in feature_records]
    keep_bbox_tuple = tuple(int(v) for v in keep_bbox)
    out = []
    for feature in feature_records:
        if feature.get("rings"):
            rings = [
                np.asarray(ring, dtype=np.float32)
                for ring in feature.get("rings", [])
                if np.asarray(ring, dtype=np.float32).ndim == 2
            ]
            points = np.concatenate(rings, axis=0) if rings else np.zeros((0, 2), dtype=np.float32)
        else:
            rings = None
            points = np.asarray(feature.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        if feature_center_inside_bbox(points_xy=points, bbox=keep_bbox_tuple):
            record = {"properties": dict(feature.get("properties", {})), "points": np.asarray(feature.get("points", []), dtype=np.float32)}
            if rings:
                record["rings"] = [ring.astype(np.float32) for ring in rings]
            out.append(record)
    return out


def _properties_equal(pred_props: Dict, gt_props: Dict) -> bool:
    pred_keys = {key for key in pred_props.keys() if str(key) not in NON_TRAINING_PROPERTY_KEYS}
    gt_keys = {key for key in gt_props.keys() if str(key) not in NON_TRAINING_PROPERTY_KEYS}
    if pred_keys != gt_keys:
        return False
    for key in pred_keys:
        pred_value = pred_props[key]
        gt_value = gt_props[key]
        if isinstance(pred_value, float) or isinstance(gt_value, float):
            if pred_value is None or gt_value is None:
                if pred_value != gt_value:
                    return False
            elif abs(float(pred_value) - float(gt_value)) > 1e-6:
                return False
        else:
            if pred_value != gt_value:
                return False
    return True


def _merge_line_points(existing: np.ndarray, candidate: np.ndarray, tolerance_px: float) -> Optional[np.ndarray]:
    if existing.shape[0] < 2 or candidate.shape[0] < 2:
        return None
    options = [
        (np.linalg.norm(existing[-1] - candidate[0]), np.concatenate([existing, candidate[1:]], axis=0)),
        (np.linalg.norm(existing[-1] - candidate[-1]), np.concatenate([existing, candidate[-2::-1]], axis=0)),
        (np.linalg.norm(existing[0] - candidate[-1]), np.concatenate([candidate[:-1], existing], axis=0)),
        (np.linalg.norm(existing[0] - candidate[0]), np.concatenate([candidate[::-1][:-1], existing], axis=0)),
    ]
    options.sort(key=lambda x: float(x[0]))
    best_dist, best_points = options[0]
    if float(best_dist) > float(tolerance_px):
        return None
    return best_points.astype(np.float32)


def _merge_feature_into_memory(memory: List[Dict], feature: Dict, task_schema: TaskSchema, tolerance_px: float = 3.0) -> None:
    if task_schema.geometry_type != "linestring":
        record = {
            "properties": dict(feature.get("properties", {})),
            "points": np.asarray(feature.get("points", []), dtype=np.float32),
        }
        if feature.get("rings"):
            record["rings"] = [
                np.asarray(ring, dtype=np.float32)
                for ring in feature.get("rings", [])
                if np.asarray(ring, dtype=np.float32).ndim == 2
            ]
        memory.append(record)
        return
    candidate_points = np.asarray(feature.get("points", []), dtype=np.float32)
    for idx, existing in enumerate(memory):
        if not _properties_equal(existing.get("properties", {}), feature.get("properties", {})):
            continue
        merged = _merge_line_points(
            existing=np.asarray(existing.get("points", []), dtype=np.float32),
            candidate=candidate_points,
            tolerance_px=float(tolerance_px),
        )
        if merged is not None:
            memory[idx] = {"properties": dict(existing.get("properties", {})), "points": merged.astype(np.float32)}
            return
    memory.append({"properties": dict(feature.get("properties", {})), "points": candidate_points.astype(np.float32)})


def _infer_model_context_limit(model) -> int:
    llm_cfg = getattr(model, "llm", None)
    llm_cfg = getattr(llm_cfg, "config", None)
    if llm_cfg is None:
        return 4096
    for key in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
        "max_seq_len",
        "model_max_length",
    ):
        value = getattr(llm_cfg, key, None)
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < value < 10_000_000:
            return value
    return 4096


def _resolve_generation_limits(
    model,
    prompt_input_ids: torch.Tensor,
    state_input_ids: torch.Tensor,
    decode_cfg: Dict,
) -> Tuple[int, int]:
    raw_max_new_tokens = int(decode_cfg.get("max_new_tokens", 512))
    raw_min_new_tokens = max(0, int(decode_cfg.get("min_new_tokens", 0)))
    if raw_max_new_tokens > 0:
        return raw_max_new_tokens, min(raw_min_new_tokens, raw_max_new_tokens)
    sat_hw = getattr(getattr(model, "sat_encoder", None), "out_hw", None)
    sat_token_count = 0
    if sat_hw is not None and len(sat_hw) == 2:
        sat_token_count = max(0, int(sat_hw[0])) * max(0, int(sat_hw[1]))
    prefix_length = sat_token_count + int(prompt_input_ids.shape[1]) + int(state_input_ids.shape[1])
    context_limit = _infer_model_context_limit(model=model)
    remaining = max(1, int(context_limit) - int(prefix_length) - 1)
    return remaining, min(raw_min_new_tokens, remaining)


def run_tiled_sample_prediction(
    cfg: Dict,
    image_path: str,
    task_schemas: Dict[str, TaskSchema],
    text_tokenizer,
    model,
    device,
    decode_cfg: Dict,
    stage: str = "predict",
    progress_label: str = "",
    log_progress: bool = False,
) -> Dict:
    total_t0 = time.time()
    raster_meta = read_raster_meta(image_path)
    tile_windows, tile_neighbors, review_mask, tile_audit = _build_tile_windows_for_inference(
        cfg=cfg,
        image_path=image_path,
        raster_meta=raster_meta,
        stage=stage,
    )
    if log_progress:
        prefix = f"[Infer] {progress_label} " if progress_label else "[Infer] "
        print(
            f"{prefix}tiling_ready tile_count={len(tile_windows)} audit_count={len(tile_audit)}",
            flush=True,
        )
    state_cfg = cfg.get("state_update", {})
    border_margin_px = int(state_cfg.get("border_margin_px", 128))
    state_max_features = int(state_cfg.get("max_features", 32))
    state_anchor_max_points = int(state_cfg.get("anchor_max_points", 6))
    sample_interval_meter = cfg.get("serialization", {}).get("sample_interval_meter")
    image_size = int(cfg["data"]["image_size"])
    task_predictions: Dict[str, List[Dict]] = {}
    parse_stats: Dict[str, Dict[str, float]] = {}
    raw_outputs: Dict[str, List[Dict]] = {}

    for task_name in _ordered_task_names(task_schemas):
        task_schema = task_schemas[task_name]
        task_t0 = time.time()
        task_state_memory: List[Dict] = []
        kept_predictions: List[Dict] = []
        raw_task_outputs: List[Dict] = []
        task_parse_stats = {
            "tile_count": 0,
            "decoded_ok_count": 0,
            "non_empty_feature_tiles": 0,
            "pred_feature_count_sum": 0,
            "kept_feature_count_sum": 0,
            "all_empty_geojson_tiles": 0,
            "dropped_by_keep_bbox_tiles": 0,
        }

        for tile_index, tile_window in enumerate(tile_windows):
            tile_t0 = time.time()
            crop_bbox = None
            keep_bbox = None
            if tile_window is not None:
                crop_bbox = (
                    int(tile_window["x0"]),
                    int(tile_window["y0"]),
                    int(tile_window["x1"]),
                    int(tile_window["y1"]),
                )
                keep_bbox = (
                    int(tile_window["keep_x0"]),
                    int(tile_window["keep_y0"]),
                    int(tile_window["keep_x1"]),
                    int(tile_window["keep_y1"]),
                )

            io_t0 = time.time()
            image_hwc, _ = read_rgb_geotiff(
                path=image_path,
                band_indices=[int(x) for x in cfg["data"].get("band_indices", [1, 2, 3])],
                crop_bbox=crop_bbox,
            )
            resize_ctx = build_resize_context(
                width=int(raster_meta.width),
                height=int(raster_meta.height),
                target_size=image_size,
                crop_bbox=crop_bbox,
            )
            image_chw = _resize_image_crop(crop_hwc=image_hwc, resize_ctx=resize_ctx)
            image_tensor = torch.from_numpy(image_chw).float().unsqueeze(0).to(device)
            io_sec = time.time() - io_t0

            neighbors = tile_neighbors[tile_index] if tile_index < len(tile_neighbors) else {"left": False, "top": False}
            state_t0 = time.time()
            state_regions = _state_region_bboxes(
                crop_bbox=crop_bbox,
                border_margin_px=border_margin_px,
                use_left=bool(neighbors.get("left", False)),
                use_top=bool(neighbors.get("top", False)),
            )
            state_features_abs = _collect_state_features(
                state_memory=task_state_memory,
                task_schema=task_schema,
                crop_bbox=crop_bbox,
                raster_meta=raster_meta,
                state_region_bboxes=state_regions,
                sample_interval_meter=float(sample_interval_meter) if sample_interval_meter is not None else None,
                max_features=state_max_features,
            )
            state_features_uv = _feature_records_to_uv(feature_records=state_features_abs, resize_ctx=resize_ctx)
            state_items = uv_feature_records_to_state_items(
                feature_records=state_features_uv,
                task_schema=task_schema,
                image_size=image_size,
                anchor_max_points=state_anchor_max_points,
            )
            has_state = bool(state_items)
            companion_task_name = _paired_task_name(task_name, task_schemas)
            companion_geojson_text = ""
            companion_feature_count = 0
            if companion_task_name and companion_task_name in task_predictions:
                companion_task_schema = task_schemas[companion_task_name]
                companion_bbox = (
                    crop_bbox
                    if crop_bbox is not None
                    else (0, 0, int(raster_meta.width), int(raster_meta.height))
                )
                companion_features_abs = _collect_state_features(
                    state_memory=task_predictions.get(companion_task_name, []),
                    task_schema=companion_task_schema,
                    crop_bbox=crop_bbox,
                    raster_meta=raster_meta,
                    state_region_bboxes=[tuple(int(v) for v in companion_bbox)],
                    sample_interval_meter=float(sample_interval_meter) if sample_interval_meter is not None else None,
                    max_features=int(companion_task_schema.max_features),
                )
                companion_feature_count = int(len(companion_features_abs))
                if companion_features_abs:
                    companion_geojson_uv = pixel_features_to_uv_geojson(
                        task_schema=companion_task_schema,
                        feature_records=companion_features_abs,
                        resize_ctx=resize_ctx,
                    )
                    companion_geojson_train = strip_non_training_fields_from_feature_collection(
                        companion_geojson_uv,
                        task_schema=companion_task_schema,
                    )
                    companion_geojson_text = geojson_dumps_compact(companion_geojson_train)
            prompt_text = prompt_text_for_task(
                cfg=cfg,
                task_schema=task_schema,
                task_name=task_name,
                has_state=has_state,
                raster_meta=raster_meta,
                crop_bbox=crop_bbox,
                companion_task_name=str(getattr(task_schemas.get(companion_task_name, None), "collection_name", "")),
                companion_geojson_text=companion_geojson_text,
            )
            prompt_ids = text_tokenizer.encode_prompt(
                prompt_text
            )
            prompt_input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
            state_text = build_state_text(
                task_schema=task_schema,
                state_items=state_items,
                geojson_text=geojson_dumps_compact(
                    pixel_features_to_uv_geojson(
                        task_schema=task_schema,
                        feature_records=state_features_abs,
                        resize_ctx=resize_ctx,
                    )
                ),
            )
            state_ids = text_tokenizer.encode_text(
                text=state_text,
                max_length=cfg.get("text", {}).get("state_max_tokens"),
                append_eos=True,
            )
            state_input_ids = torch.tensor(state_ids, dtype=torch.long, device=device).unsqueeze(0)
            state_attention_mask = torch.ones_like(state_input_ids, dtype=torch.long)
            max_new_tokens, min_new_tokens = _resolve_generation_limits(
                model=model,
                prompt_input_ids=prompt_input_ids,
                state_input_ids=state_input_ids,
                decode_cfg=decode_cfg,
            )
            state_sec = time.time() - state_t0

            if log_progress:
                prefix = f"[Infer] {progress_label} " if progress_label else "[Infer] "
                print(
                    f"{prefix}task={task_name} tile={tile_index + 1}/{len(tile_windows)} "
                    f"crop_bbox={crop_bbox} keep_bbox={keep_bbox} "
                    f"state_anchors={len(state_items)} prompt_tokens={prompt_input_ids.shape[1]} "
                    f"state_tokens={state_input_ids.shape[1]} companion_features={companion_feature_count} "
                    f"max_new_tokens={max_new_tokens} "
                    f"io_sec={io_sec:.2f} state_sec={state_sec:.2f}",
                    flush=True,
                )

            gen_t0 = time.time()
            with torch.no_grad():
                pred_qwen_ids = model.generate(
                    image=image_tensor,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    pv_images=None,
                    state_input_ids=state_input_ids,
                    state_attention_mask=state_attention_mask,
                    max_new_tokens=int(max_new_tokens),
                    min_new_tokens=int(min_new_tokens),
                    temperature=float(decode_cfg.get("temperature", 1.0)),
                    top_k=int(decode_cfg.get("top_k", 1)),
                    repetition_penalty=float(decode_cfg.get("repetition_penalty", 1.0)),
                    grammar_helper=None,
                    use_kv_cache=_as_bool(decode_cfg.get("use_kv_cache", True), default=True),
                    return_token_meta=False,
                )
            gen_sec = time.time() - gen_t0
            parse_t0 = time.time()
            pred_qwen_ids = pred_qwen_ids[0].detach().cpu().tolist()
            pred_text = text_tokenizer.decode_text(pred_qwen_ids)
            pred_geojson = coerce_feature_collection(
                task_schema=task_schema,
                obj=extract_first_json_object(pred_text),
            )
            pred_features_abs = (
                uv_geojson_to_pixel_features(
                    geojson_dict=pred_geojson,
                    task_schema=task_schema,
                    resize_ctx=resize_ctx,
                )
                if pred_geojson is not None
                else []
            )
            parse_sec = time.time() - parse_t0

            keep_t0 = time.time()
            kept_current = _retain_predictions_for_keep_bbox(
                feature_records=pred_features_abs,
                keep_bbox=keep_bbox,
            )
            for feature in kept_current:
                _merge_feature_into_memory(task_state_memory, feature=feature, task_schema=task_schema, tolerance_px=3.0)
                _merge_feature_into_memory(kept_predictions, feature=feature, task_schema=task_schema, tolerance_px=3.0)
            keep_sec = time.time() - keep_t0

            stripped_ids = text_tokenizer.strip_padding(pred_qwen_ids)
            if text_tokenizer.eos_token_id in stripped_ids:
                stripped_ids = stripped_ids[: stripped_ids.index(text_tokenizer.eos_token_id)]
            empty_sequence_ok = stripped_ids == []
            decoded_ok = bool(pred_geojson is not None or empty_sequence_ok)
            raw_task_outputs.append(
                {
                    "tile_index": int(tile_index),
                    "crop_bbox": None if crop_bbox is None else [int(v) for v in crop_bbox],
                    "keep_bbox": None if keep_bbox is None else [int(v) for v in keep_bbox],
                    "mask_ratio": float(tile_window.get("mask_ratio", 0.0)) if isinstance(tile_window, dict) else 0.0,
                    "mask_pixels": int(tile_window.get("mask_pixels", 0)) if isinstance(tile_window, dict) else 0,
                    "decoded_ok": bool(decoded_ok),
                    "item_count": int(len(pred_features_abs)),
                    "pred_feature_count": int(len(pred_features_abs)),
                    "kept_feature_count": int(len(kept_current)),
                    "state_anchor_count": int(len(state_items)),
                    "prompt_token_count": int(prompt_input_ids.shape[1]),
                    "state_token_count": int(state_input_ids.shape[1]),
                    "companion_task_name": str(companion_task_name),
                    "companion_feature_count": int(companion_feature_count),
                    "max_new_tokens": int(max_new_tokens),
                    "min_new_tokens": int(min_new_tokens),
                    "prompt_text": str(prompt_text),
                    "state_text": str(state_text),
                    "generate_sec": float(gen_sec),
                    "parse_sec": float(parse_sec),
                    "keep_sec": float(keep_sec),
                    "token_ids": [int(x) for x in pred_qwen_ids],
                    "pred_text": pred_text,
                    "pred_uv_geojson": pred_geojson,
                    "pred_geojson": pixel_features_to_geojson(
                        task_schema=task_schema,
                        feature_records=pred_features_abs,
                        raster_meta=raster_meta,
                    ),
                    "kept_uv_geojson": pixel_features_to_uv_geojson(
                        task_schema=task_schema,
                        feature_records=kept_current,
                        resize_ctx=resize_ctx,
                    ),
                    "kept_geojson": pixel_features_to_geojson(
                        task_schema=task_schema,
                        feature_records=kept_current,
                        raster_meta=raster_meta,
                    ),
                    "items": [],
                }
            )
            task_parse_stats["tile_count"] += 1
            if decoded_ok:
                task_parse_stats["decoded_ok_count"] += 1
            task_parse_stats["pred_feature_count_sum"] += int(len(pred_features_abs))
            task_parse_stats["kept_feature_count_sum"] += int(len(kept_current))
            if pred_geojson is not None and int(len(pred_features_abs)) == 0:
                task_parse_stats["all_empty_geojson_tiles"] += 1
            if int(len(pred_features_abs)) > 0 and int(len(kept_current)) == 0:
                task_parse_stats["dropped_by_keep_bbox_tiles"] += 1
            if pred_features_abs:
                task_parse_stats["non_empty_feature_tiles"] += 1
            if log_progress:
                prefix = f"[Infer] {progress_label} " if progress_label else "[Infer] "
                print(
                    f"{prefix}task={task_name} tile={tile_index + 1}/{len(tile_windows)} "
                    f"crop_bbox={crop_bbox} keep_bbox={keep_bbox} "
                    f"generate_sec={gen_sec:.2f} parse_sec={parse_sec:.2f} keep_sec={keep_sec:.2f} "
                    f"pred_features={len(pred_features_abs)} kept_features={len(kept_current)} "
                    f"decoded_ok={decoded_ok} tile_total_sec={time.time() - tile_t0:.2f}",
                    flush=True,
                )

        task_predictions[task_name] = deduplicate_feature_records(
            task_schema=task_schema,
            feature_records=kept_predictions,
            raster_meta=raster_meta,
            line_distance_threshold_m=float(cfg.get("postprocess", {}).get("line_dedup_distance_m", 1.0)),
            polygon_iou_threshold=float(cfg.get("postprocess", {}).get("polygon_dedup_iou", 0.5)),
        )
        parse_stats[task_name] = task_parse_stats
        raw_outputs[task_name] = raw_task_outputs
        if log_progress:
            prefix = f"[Infer] {progress_label} " if progress_label else "[Infer] "
            print(
                f"{prefix}task={task_name} finished task_sec={time.time() - task_t0:.2f} "
                f"tile_count={task_parse_stats['tile_count']} decoded_ok={task_parse_stats['decoded_ok_count']} "
                f"pred_sum={task_parse_stats['pred_feature_count_sum']} kept_sum={task_parse_stats['kept_feature_count_sum']} "
                f"dropped_by_keep_bbox={task_parse_stats['dropped_by_keep_bbox_tiles']}",
                flush=True,
            )

    result = {
        "raster_meta": raster_meta,
        "review_mask": review_mask,
        "tile_windows": tile_windows,
        "tile_audit": tile_audit,
        "task_predictions": task_predictions,
        "task_prediction_geojsons": assign_incremental_feature_ids(
            {
                task_name: pixel_features_to_geojson(
                    task_schema=task_schemas[task_name],
                    feature_records=task_predictions.get(task_name, []),
                    raster_meta=raster_meta,
                )
                for task_name in task_schemas.keys()
            }
        ),
        "parse_stats": parse_stats,
        "raw_outputs": raw_outputs,
        "alignment_stats": alignment_stats,
    }
    if log_progress:
        prefix = f"[Infer] {progress_label} " if progress_label else "[Infer] "
        print(
            f"{prefix}sample_finished total_sec={time.time() - total_t0:.2f}",
            flush=True,
        )
    return result
