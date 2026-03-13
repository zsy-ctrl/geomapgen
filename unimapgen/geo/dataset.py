from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from unimapgen.utils import ensure_dir

from .coord_sequence import (
    feature_record_sort_key,
    points_abs_to_uv,
    rings_abs_to_uv,
    uv_feature_records_to_state_items,
    uv_feature_records_to_target_items,
)
from .errors import raise_geo_error
from .geometry import (
    ResizeContext,
    audit_tile_window_selection,
    apply_square_augment,
    annotate_tile_windows_with_mask,
    build_resize_context,
    clip_feature_to_bbox,
    clip_polygon_rings_to_bbox,
    compute_mask_bbox,
    expand_bbox,
    feature_intersects_bbox,
    generate_tile_windows,
    resample_feature_points,
    select_tile_windows,
)
from .io import RasterMeta, geojson_to_pixel_features, load_geojson, read_binary_mask, read_raster_meta, read_rgb_geotiff
from .prompting import build_state_text, build_target_text, build_task_prompt_text
from .io import geojson_dumps_compact, pixel_features_to_geojson
from .schema import TaskSchema


@dataclass
class GeoVectorDatasetConfig:
    stage: str
    dataset_root: str
    split: str
    image_relpath: str
    review_mask_relpath: Optional[str]
    task_to_label_relpath: Dict[str, str]
    image_size: int
    band_indices: List[int]
    mask_threshold: int
    crop_to_review_mask: bool
    review_crop_pad_px: int
    sample_interval_meter: Optional[float]
    max_samples: Optional[int]
    train_augment: bool
    aug_rot90_prob: float
    aug_hflip_prob: float
    aug_vflip_prob: float
    tiling_enabled: bool
    tile_size_px: int
    tile_overlap_px: int
    tile_keep_margin_px: int
    tile_min_mask_ratio: float
    tile_min_mask_pixels: int
    tile_fallback_to_all_if_empty: bool
    tile_search_within_review_bbox: bool
    tile_allow_empty: bool
    tile_max_per_sample: Optional[int]
    feature_filter_by_review_mask: bool
    feature_mask_min_inside_ratio: float
    state_enabled: bool
    state_border_margin_px: int
    state_max_features: int
    state_anchor_max_points: int
    prompt_with_state: str
    prompt_without_state: str
    prompt_include_geospatial_context: bool
    prompt_geospatial_precision: int
    cache_enabled: bool
    cache_write_enabled: bool
    cache_dir: Optional[str]
    cache_namespace: str


class GeoVectorDataset(Dataset):
    def __init__(
        self,
        cfg: GeoVectorDatasetConfig,
        task_schemas: Dict[str, TaskSchema],
    ) -> None:
        self.cfg = cfg
        self.task_schemas = dict(task_schemas)
        self.features_by_key: Dict[str, List[Dict]] = {}
        self.tile_audit_records: List[Dict] = []
        self.cache_root = str(cfg.cache_dir).strip() if cfg.cache_dir else ""
        self.cache_runtime_hits = 0
        self.cache_runtime_misses = 0

        split_root = os.path.join(cfg.dataset_root, cfg.split)
        if not os.path.isdir(split_root):
            raise_geo_error("GEO-1501", f"dataset split directory not found: {split_root}")

        sample_dirs = [
            os.path.join(split_root, name)
            for name in sorted(os.listdir(split_root))
            if os.path.isdir(os.path.join(split_root, name))
        ]
        if cfg.max_samples is not None and int(cfg.max_samples) > 0:
            sample_dirs = sample_dirs[: int(cfg.max_samples)]

        self.items: List[Dict] = []
        for sample_dir in sample_dirs:
            sample_id = os.path.basename(sample_dir)
            image_path = os.path.join(sample_dir, cfg.image_relpath)
            review_mask_path = (
                os.path.join(sample_dir, cfg.review_mask_relpath) if cfg.review_mask_relpath else None
            )
            if not os.path.isfile(image_path):
                continue
            raster_meta = read_raster_meta(path=image_path)
            review_mask = None
            review_bbox = None
            if review_mask_path and os.path.isfile(review_mask_path):
                review_mask = read_binary_mask(path=review_mask_path, threshold=cfg.mask_threshold)
                review_bbox = compute_mask_bbox(review_mask)

            crop_bbox = None
            if bool(cfg.crop_to_review_mask) and review_bbox is not None:
                crop_bbox = expand_bbox(
                    bbox=review_bbox,
                    pad_px=int(cfg.review_crop_pad_px),
                    width=int(raster_meta.width),
                    height=int(raster_meta.height),
                )

            tile_windows, tile_audits = self._build_tile_windows(
                raster_width=int(raster_meta.width),
                raster_height=int(raster_meta.height),
        review_mask=review_mask,
        review_bbox=review_bbox,
        crop_bbox=crop_bbox,
            )
            for audit in tile_audits:
                self.tile_audit_records.append(
                    {
                        "stage": str(self.cfg.stage),
                        "split": str(self.cfg.split),
                        "sample_id": sample_id,
                        "sample_dir": sample_dir,
                        "image_path": image_path,
                        "crop_bbox": None if crop_bbox is None else [int(v) for v in crop_bbox],
                        **audit,
                    }
                )
            tile_neighbors = self._build_tile_scan_neighbors(tile_windows)

            for task_name, task_schema in self.task_schemas.items():
                label_relpath = cfg.task_to_label_relpath.get(task_name)
                if not label_relpath:
                    continue
                label_path = os.path.join(sample_dir, label_relpath)
                if not os.path.isfile(label_path):
                    continue
                feature_key = f"{sample_id}::{task_name}"
                self.features_by_key[feature_key] = geojson_to_pixel_features(
                    geojson_dict=load_geojson(label_path),
                    task_schema=task_schema,
                    raster_meta=raster_meta,
                )
                self._append_task_items(
                    sample_id=sample_id,
                    sample_dir=sample_dir,
                    image_path=image_path,
                    review_mask_path=review_mask_path,
                    label_path=label_path,
                    task_name=task_name,
                    task_schema=task_schema,
                    feature_key=feature_key,
                    tile_windows=tile_windows,
                    tile_neighbors=tile_neighbors,
                    crop_bbox=crop_bbox,
                )
        self.cache_stats = self._summarize_cache_stats()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        base = self._load_or_build_cached_base(item=item)
        raster_meta = base["raster_meta"]
        if isinstance(raster_meta, dict):
            raster_meta = RasterMeta.from_dict(raster_meta)
        resize_ctx = base["resize_ctx"]
        if isinstance(resize_ctx, dict):
            resize_ctx = ResizeContext.from_dict(resize_ctx)
        image_chw = np.asarray(base["image_chw"], dtype=np.float32).copy()
        target_features_uv = self._clone_feature_records(base["target_features_uv"])
        state_features_uv = self._clone_feature_records(base["state_features_uv"])

        if bool(self.cfg.train_augment):
            rot_k, hflip, vflip = self._sample_augment_params()
            image_chw, target_features_uv = self._apply_feature_augment(
                image_chw=image_chw,
                feature_records=target_features_uv,
                rot_k=rot_k,
                hflip=hflip,
                vflip=vflip,
            )
            _, state_features_uv = self._apply_feature_augment(
                image_chw=image_chw.copy(),
                feature_records=state_features_uv,
                rot_k=rot_k,
                hflip=hflip,
                vflip=vflip,
            )

        target_items = uv_feature_records_to_target_items(
            feature_records=target_features_uv,
            task_schema=item["task_schema"],
            image_size=int(self.cfg.image_size),
        )
        state_items = uv_feature_records_to_state_items(
            feature_records=state_features_uv,
            task_schema=item["task_schema"],
            image_size=int(self.cfg.image_size),
            anchor_max_points=int(self.cfg.state_anchor_max_points),
        )
        prompt_text = build_task_prompt_text(
            task_name=item["task_name"],
            base_prompt=item["task_schema"].prompt_template,
            has_state=bool(self.cfg.state_enabled and state_items),
            with_state_suffix=self.cfg.prompt_with_state,
            without_state_suffix=self.cfg.prompt_without_state,
            raster_meta=raster_meta,
            crop_bbox=item["crop_bbox"],
            include_geospatial_context=bool(self.cfg.prompt_include_geospatial_context),
            geospatial_precision=int(self.cfg.prompt_geospatial_precision),
        )
        state_geojson = pixel_features_to_geojson(
            task_schema=item["task_schema"],
            feature_records=self._feature_records_from_uv(state_features_uv, resize_ctx=resize_ctx),
            raster_meta=raster_meta,
        )
        target_geojson = pixel_features_to_geojson(
            task_schema=item["task_schema"],
            feature_records=self._feature_records_from_uv(target_features_uv, resize_ctx=resize_ctx),
            raster_meta=raster_meta,
        )
        state_text = build_state_text(
            task_schema=item["task_schema"],
            state_items=state_items,
            geojson_text=geojson_dumps_compact(state_geojson),
        )
        target_text = build_target_text(
            task_schema=item["task_schema"],
            target_items=target_items,
            geojson_text=geojson_dumps_compact(target_geojson),
        )

        return {
            "image": torch.from_numpy(image_chw).float(),
            "sample_id": item["sample_id"],
            "sample_dir": item["sample_dir"],
            "task_name": item["task_name"],
            "prompt_text": prompt_text,
            "state_text": state_text,
            "target_text": target_text,
            "state_items": state_items,
            "target_items": target_items,
            "state_feature_records": self._clone_feature_records(self._feature_records_from_uv(state_features_uv, resize_ctx=resize_ctx)),
            "target_feature_records": self._clone_feature_records(self._feature_records_from_uv(target_features_uv, resize_ctx=resize_ctx)),
            "raster_meta": raster_meta.to_dict(),
            "resize_ctx": resize_ctx.to_dict(),
            "review_mask_path": item["review_mask_path"],
            "label_path": item["label_path"],
            "image_path": item["image_path"],
            "crop_bbox": item["crop_bbox"],
            "tile_window": item["tile_window"],
            "tile_index": int(item["tile_index"]),
            "tile_count": int(item["tile_count"]),
            "state_has_left": bool(item["state_has_left"]),
            "state_has_top": bool(item["state_has_top"]),
            "cache_path": item.get("cache_path", ""),
        }

    def _resize_cropped_image(self, crop_hwc: np.ndarray, resize_ctx) -> np.ndarray:
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

    def _load_or_build_cached_base(self, item: Dict) -> Dict:
        cache_path = str(item.get("cache_path", "")).strip()
        if bool(self.cfg.cache_enabled) and cache_path and os.path.isfile(cache_path):
            self.cache_runtime_hits += 1
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            return {
                "image_chw": np.asarray(cached["image_chw"], dtype=np.float32),
                "raster_meta": dict(cached["raster_meta"]),
                "resize_ctx": dict(cached["resize_ctx"]),
                "target_features_uv": self._deserialize_feature_records(cached.get("target_features_uv", [])),
                "state_features_uv": self._deserialize_feature_records(cached.get("state_features_uv", [])),
            }
        if bool(self.cfg.cache_enabled):
            self.cache_runtime_misses += 1

        image_hwc, raster_meta = read_rgb_geotiff(
            path=item["image_path"],
            band_indices=self.cfg.band_indices,
            crop_bbox=item["crop_bbox"],
        )
        review_mask = None
        if item["review_mask_path"] and os.path.isfile(item["review_mask_path"]):
            review_mask = read_binary_mask(path=item["review_mask_path"], threshold=self.cfg.mask_threshold)
        image_hwc = np.clip(image_hwc, 0.0, 255.0)
        if review_mask is not None:
            image_hwc = self._apply_review_mask_to_image(
                image_hwc=image_hwc,
                review_mask=review_mask,
                crop_bbox=item["crop_bbox"],
            )
        resize_ctx = build_resize_context(
            width=int(raster_meta.width),
            height=int(raster_meta.height),
            target_size=int(self.cfg.image_size),
            crop_bbox=item["crop_bbox"],
        )
        image_chw = self._resize_cropped_image(crop_hwc=image_hwc, resize_ctx=resize_ctx)

        raw_features = self.features_by_key[item["feature_key"]]
        target_features_abs = self._prepare_features(
            raw_features=raw_features,
            task_schema=item["task_schema"],
            crop_bbox=item["crop_bbox"],
            raster_meta=raster_meta,
            review_mask=review_mask,
            state_bboxes=None,
            max_features=int(item["task_schema"].max_features),
        )
        state_bboxes = self._build_state_region_bboxes(
            crop_bbox=item["crop_bbox"],
            use_left=bool(item["state_has_left"]),
            use_top=bool(item["state_has_top"]),
        )
        state_features_abs = self._prepare_features(
            raw_features=raw_features,
            task_schema=item["task_schema"],
            crop_bbox=item["crop_bbox"],
            raster_meta=raster_meta,
            review_mask=review_mask,
            state_bboxes=state_bboxes,
            max_features=int(self.cfg.state_max_features),
        )
        target_features_uv = self._feature_records_to_uv(feature_records=target_features_abs, resize_ctx=resize_ctx)
        state_features_uv = self._feature_records_to_uv(feature_records=state_features_abs, resize_ctx=resize_ctx)

        if bool(self.cfg.cache_write_enabled) and cache_path:
            ensure_dir(os.path.dirname(cache_path))
            torch.save(
                {
                    "image_chw": np.asarray(image_chw, dtype=np.float32),
                    "raster_meta": raster_meta.to_dict(),
                    "resize_ctx": resize_ctx.to_dict(),
                    "target_features_uv": self._serialize_feature_records(target_features_uv),
                    "state_features_uv": self._serialize_feature_records(state_features_uv),
                },
                cache_path,
            )

        return {
            "image_chw": np.asarray(image_chw, dtype=np.float32),
            "raster_meta": raster_meta.to_dict(),
            "resize_ctx": resize_ctx.to_dict(),
            "target_features_uv": self._clone_feature_records(target_features_uv),
            "state_features_uv": self._clone_feature_records(state_features_uv),
        }

    def _summarize_cache_stats(self) -> Dict[str, object]:
        total_records = int(len(self.items))
        if not bool(self.cfg.cache_enabled) or not self.cache_root:
            return {
                "enabled": bool(self.cfg.cache_enabled),
                "cache_root": self.cache_root,
                "total_records": total_records,
                "existing_records": 0,
                "missing_records": total_records,
                "write_enabled": bool(self.cfg.cache_write_enabled),
            }
        existing_records = 0
        for item in self.items:
            cache_path = str(item.get("cache_path", "")).strip()
            if cache_path and os.path.isfile(cache_path):
                existing_records += 1
        return {
            "enabled": True,
            "cache_root": self.cache_root,
            "total_records": total_records,
            "existing_records": int(existing_records),
            "missing_records": int(max(0, total_records - existing_records)),
            "write_enabled": bool(self.cfg.cache_write_enabled),
        }

    def _serialize_feature_records(self, feature_records: Sequence[Dict]) -> List[Dict]:
        return self._clone_feature_records(feature_records)

    def _deserialize_feature_records(self, feature_records: Sequence[Dict]) -> List[Dict]:
        return self._clone_feature_records(feature_records)

    def _clone_feature_records(self, feature_records: Sequence[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for feature in feature_records:
            record = {
                "properties": dict(feature.get("properties", {})),
                "points": np.asarray(feature.get("points", []), dtype=np.float32).copy(),
            }
            if feature.get("rings"):
                record["rings"] = [np.asarray(ring, dtype=np.float32).copy() for ring in feature.get("rings", [])]
            if "cut_start" in feature:
                record["cut_start"] = bool(feature.get("cut_start", False))
            if "cut_end" in feature:
                record["cut_end"] = bool(feature.get("cut_end", False))
            if "clipped" in feature:
                record["clipped"] = bool(feature.get("clipped", False))
            out.append(record)
        return out

    def _feature_records_to_uv(self, feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
        out = []
        for feature in feature_records:
            points_abs = np.asarray(feature.get("points", []), dtype=np.float32)
            points_uv = points_abs_to_uv(points_xy=points_abs, resize_ctx=resize_ctx)
            record = {"properties": dict(feature.get("properties", {})), "points": points_uv.astype(np.float32)}
            if feature.get("rings"):
                record["rings"] = rings_abs_to_uv(feature.get("rings", []), resize_ctx=resize_ctx)
            if "cut_start" in feature:
                record["cut_start"] = bool(feature.get("cut_start", False))
            if "cut_end" in feature:
                record["cut_end"] = bool(feature.get("cut_end", False))
            if "clipped" in feature:
                record["clipped"] = bool(feature.get("clipped", False))
            out.append(record)
        return out

    def _feature_records_from_uv(self, feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
        from .coord_sequence import points_uv_to_abs, rings_uv_to_abs

        out: List[Dict] = []
        for feature in feature_records:
            record = {
                "properties": dict(feature.get("properties", {})),
                "points": points_uv_to_abs(
                    points_uv=np.asarray(feature.get("points", []), dtype=np.float32),
                    resize_ctx=resize_ctx,
                ),
            }
            if feature.get("rings"):
                record["rings"] = rings_uv_to_abs(feature.get("rings", []), resize_ctx=resize_ctx)
            if "cut_start" in feature:
                record["cut_start"] = bool(feature.get("cut_start", False))
            if "cut_end" in feature:
                record["cut_end"] = bool(feature.get("cut_end", False))
            if "clipped" in feature:
                record["clipped"] = bool(feature.get("clipped", False))
            out.append(record)
        return out

    def _crop_mask_to_bbox(
        self,
        review_mask: np.ndarray,
        crop_bbox: Optional[Sequence[int]],
    ) -> np.ndarray:
        mask = np.asarray(review_mask, dtype=np.uint8)
        if crop_bbox is None:
            return mask
        x0, y0, x1, y1 = [int(v) for v in crop_bbox]
        return mask[y0:y1, x0:x1]

    def _apply_review_mask_to_image(
        self,
        image_hwc: np.ndarray,
        review_mask: np.ndarray,
        crop_bbox: Optional[Sequence[int]],
    ) -> np.ndarray:
        mask_crop = self._crop_mask_to_bbox(review_mask=review_mask, crop_bbox=crop_bbox)
        if mask_crop.ndim != 2:
            return np.asarray(image_hwc, dtype=np.float32)
        masked = np.asarray(image_hwc, dtype=np.float32).copy()
        if mask_crop.shape[0] != masked.shape[0] or mask_crop.shape[1] != masked.shape[1]:
            h = min(mask_crop.shape[0], masked.shape[0])
            w = min(mask_crop.shape[1], masked.shape[1])
            masked[:h, :w][mask_crop[:h, :w] <= 0] = 0.0
            if h < masked.shape[0]:
                masked[h:, :] = 0.0
            if w < masked.shape[1]:
                masked[:, w:] = 0.0
            return masked
        masked[mask_crop <= 0] = 0.0
        return masked

    def _sample_augment_params(self) -> Tuple[int, bool, bool]:
        rot_k = 0
        if np.random.rand() < float(self.cfg.aug_rot90_prob):
            rot_k = int(np.random.randint(1, 4))
        hflip = bool(np.random.rand() < float(self.cfg.aug_hflip_prob))
        vflip = bool(np.random.rand() < float(self.cfg.aug_vflip_prob))
        return rot_k, hflip, vflip

    def _apply_feature_augment(
        self,
        image_chw: np.ndarray,
        feature_records: Sequence[Dict],
        rot_k: int,
        hflip: bool,
        vflip: bool,
    ) -> Tuple[np.ndarray, List[Dict]]:
        point_arrays = [np.asarray(feature.get("points", []), dtype=np.float32).copy() for feature in feature_records]
        ring_arrays: List[np.ndarray] = []
        ring_counts: List[int] = []
        for feature in feature_records:
            rings = [np.asarray(ring, dtype=np.float32).copy() for ring in feature.get("rings", [])]
            ring_counts.append(len(rings))
            ring_arrays.extend(rings)

        combined_arrays = point_arrays + ring_arrays
        aug_image, aug_arrays = apply_square_augment(
            image_chw=image_chw,
            feature_points=combined_arrays,
            rot90_k=rot_k,
            hflip=hflip,
            vflip=vflip,
        )
        aug_points = aug_arrays[: len(point_arrays)]
        aug_rings_flat = aug_arrays[len(point_arrays) :]
        out = []
        ring_offset = 0
        for feature, pts, ring_count in zip(feature_records, aug_points, ring_counts):
            record = {"properties": dict(feature.get("properties", {})), "points": pts.astype(np.float32)}
            if ring_count > 0:
                record["rings"] = [
                    np.asarray(aug_rings_flat[ring_offset + idx], dtype=np.float32).astype(np.float32)
                    for idx in range(int(ring_count))
                ]
                ring_offset += int(ring_count)
            if "cut_start" in feature:
                record["cut_start"] = bool(feature.get("cut_start", False))
            if "cut_end" in feature:
                record["cut_end"] = bool(feature.get("cut_end", False))
            if "clipped" in feature:
                record["clipped"] = bool(feature.get("clipped", False))
            out.append(record)
        return aug_image, out

    def _build_tile_windows(
        self,
        raster_width: int,
        raster_height: int,
        review_mask: Optional[np.ndarray],
        review_bbox: Optional[Sequence[int]],
        crop_bbox: Optional[Sequence[int]],
    ) -> Tuple[List[Optional[Dict]], List[Dict]]:
        if not bool(self.cfg.tiling_enabled):
            return [None], [
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
        region_bbox = None
        if bool(self.cfg.tile_search_within_review_bbox) and review_bbox is not None:
            region_bbox = expand_bbox(
                bbox=tuple(int(v) for v in review_bbox),
                pad_px=int(self.cfg.review_crop_pad_px),
                width=int(raster_width),
                height=int(raster_height),
            )
        elif crop_bbox is not None:
            region_bbox = tuple(int(v) for v in crop_bbox)
        tile_windows = generate_tile_windows(
            width=int(raster_width),
            height=int(raster_height),
            tile_size_px=int(self.cfg.tile_size_px),
            overlap_px=int(self.cfg.tile_overlap_px),
            region_bbox=None if region_bbox is None else tuple(int(v) for v in region_bbox),
            keep_margin_px=int(self.cfg.tile_keep_margin_px),
        )
        tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
        tile_windows, tile_audits = audit_tile_window_selection(
            tile_windows=tile_windows,
            min_mask_ratio=float(self.cfg.tile_min_mask_ratio),
            min_mask_pixels=int(self.cfg.tile_min_mask_pixels),
            max_tiles=self.cfg.tile_max_per_sample,
            fallback_to_all_if_empty=bool(self.cfg.tile_fallback_to_all_if_empty),
        )
        tile_windows = sorted(tile_windows, key=lambda window: (int(window.y0), int(window.x0)))
        return [window.to_dict() for window in tile_windows], tile_audits

    def _build_tile_scan_neighbors(self, tile_windows: Sequence[Optional[Dict]]) -> List[Dict[str, bool]]:
        if not tile_windows:
            return []
        if tile_windows == [None]:
            return [{"left": False, "top": False}]
        seen_rows = set()
        seen_cols = set()
        out: List[Dict[str, bool]] = []
        for tile_window in tile_windows:
            if tile_window is None:
                out.append({"left": False, "top": False})
                continue
            row_key = int(tile_window["y0"])
            col_key = int(tile_window["x0"])
            out.append({"left": row_key in seen_rows, "top": col_key in seen_cols})
            seen_rows.add(row_key)
            seen_cols.add(col_key)
        return out

    def _build_state_region_bboxes(
        self,
        crop_bbox: Optional[Sequence[int]],
        use_left: bool,
        use_top: bool,
    ) -> List[Tuple[int, int, int, int]]:
        if not self.cfg.state_enabled or crop_bbox is None:
            return []
        x0, y0, x1, y1 = [int(v) for v in crop_bbox]
        margin = max(1, int(self.cfg.state_border_margin_px))
        out: List[Tuple[int, int, int, int]] = []
        if bool(use_left):
            out.append((int(x0), int(y0), int(min(x1, x0 + margin)), int(y1)))
        if bool(use_top):
            out.append((int(x0), int(y0), int(x1), int(min(y1, y0 + margin))))
        return out

    def _sorted_raw_features(
        self,
        raw_features: Sequence[Dict],
        task_schema: TaskSchema,
    ) -> List[Dict]:
        cloned = self._clone_feature_records(raw_features)
        cloned.sort(key=lambda feature: feature_record_sort_key(feature=feature, geometry_type=task_schema.geometry_type))
        return cloned

    def _line_mask_segments(
        self,
        points_xy: np.ndarray,
        task_schema: TaskSchema,
        review_mask: Optional[np.ndarray],
    ) -> List[Dict]:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < int(task_schema.min_points_per_feature):
            return []
        if review_mask is None or not bool(self.cfg.feature_filter_by_review_mask):
            return [{"points": pts.astype(np.float32), "cut_start": False, "cut_end": False}]
        height, width = review_mask.shape[:2]
        cols = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, width - 1)
        rows = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, height - 1)
        inside = (review_mask[rows, cols] > 0).tolist()
        out: List[Dict] = []
        start = None
        for idx, flag in enumerate(inside):
            if flag and start is None:
                start = idx
            if (not flag) and start is not None:
                piece = pts[start:idx]
                if piece.shape[0] >= int(task_schema.min_points_per_feature):
                    out.append(
                        {
                            "points": piece.astype(np.float32),
                            "cut_start": bool(start > 0),
                            "cut_end": bool(idx < pts.shape[0]),
                        }
                    )
                start = None
        if start is not None:
            piece = pts[start:]
            if piece.shape[0] >= int(task_schema.min_points_per_feature):
                out.append(
                    {
                        "points": piece.astype(np.float32),
                        "cut_start": bool(start > 0),
                        "cut_end": False,
                    }
                )
        return out

    def _line_piece_cut_flags_after_clip(
        self,
        source_points: np.ndarray,
        clipped_points: np.ndarray,
        cut_start: bool,
        cut_end: bool,
    ) -> Tuple[bool, bool]:
        src = np.asarray(source_points, dtype=np.float32)
        dst = np.asarray(clipped_points, dtype=np.float32)
        if src.ndim != 2 or dst.ndim != 2 or src.shape[0] == 0 or dst.shape[0] == 0:
            return bool(cut_start), bool(cut_end)
        tol = 1e-3
        new_cut_start = bool(cut_start) or not np.allclose(dst[0], src[0], atol=tol)
        new_cut_end = bool(cut_end) or not np.allclose(dst[-1], src[-1], atol=tol)
        return new_cut_start, new_cut_end

    def _polygon_passes_review_mask(
        self,
        rings_xy: Sequence[np.ndarray],
        task_schema: TaskSchema,
        review_mask: Optional[np.ndarray],
    ) -> Tuple[bool, bool]:
        if not rings_xy:
            return False, False
        outer = np.asarray(rings_xy[0], dtype=np.float32)
        if outer.ndim != 2 or outer.shape[0] < int(task_schema.min_points_per_feature):
            return False, False
        if review_mask is None or not bool(self.cfg.feature_filter_by_review_mask):
            return True, False
        height, width = review_mask.shape[:2]
        cols = np.clip(np.round(outer[:, 0]).astype(np.int64), 0, width - 1)
        rows = np.clip(np.round(outer[:, 1]).astype(np.int64), 0, height - 1)
        inside = review_mask[rows, cols] > 0
        inside_ratio = float(inside.mean()) if inside.size > 0 else 0.0
        return bool(inside_ratio >= float(self.cfg.feature_mask_min_inside_ratio)), bool(inside_ratio < 0.999)

    def _feature_record_output_sort_key(
        self,
        feature: Dict,
        task_schema: TaskSchema,
    ) -> Tuple:
        return feature_record_sort_key(feature=feature, geometry_type=task_schema.geometry_type)

    def _rings_equal(
        self,
        src_rings: Sequence[np.ndarray],
        dst_rings: Sequence[np.ndarray],
        tol: float = 1e-3,
    ) -> bool:
        if len(src_rings) != len(dst_rings):
            return False
        for src_ring, dst_ring in zip(src_rings, dst_rings):
            src = np.asarray(src_ring, dtype=np.float32)
            dst = np.asarray(dst_ring, dtype=np.float32)
            if src.shape != dst.shape or not np.allclose(src, dst, atol=float(tol)):
                return False
        return True

    def _prepare_features(
        self,
        raw_features: Sequence[Dict],
        task_schema: TaskSchema,
        crop_bbox: Optional[Sequence[int]],
        raster_meta,
        review_mask: Optional[np.ndarray],
        state_bboxes: Optional[Sequence[Tuple[int, int, int, int]]],
        max_features: int,
    ) -> List[Dict]:
        out: List[Dict] = []
        max_features_limit = int(max_features)
        pixel_size_meter = max(abs(float(raster_meta.pixel_size_x)), 1e-6)
        interval_px = None
        if self.cfg.sample_interval_meter is not None and float(self.cfg.sample_interval_meter) > 0:
            interval_px = float(self.cfg.sample_interval_meter) / pixel_size_meter
        closed = bool(task_schema.geometry_type == "polygon")
        crop_bbox_tuple = None if crop_bbox is None else tuple(int(v) for v in crop_bbox)
        for feature in self._sorted_raw_features(raw_features=raw_features, task_schema=task_schema):
            if task_schema.geometry_type == "polygon" and feature.get("rings"):
                rings_original = [np.asarray(ring, dtype=np.float32) for ring in feature.get("rings", [])]
                ok_by_mask, partial_by_mask = self._polygon_passes_review_mask(
                    rings_xy=rings_original,
                    task_schema=task_schema,
                    review_mask=review_mask,
                )
                if not ok_by_mask:
                    continue
                candidate_groups: List[Dict] = [{"rings": rings_original, "clipped": bool(partial_by_mask)}]
                if crop_bbox_tuple is not None:
                    cropped_groups: List[Dict] = []
                    for candidate_group in candidate_groups:
                        outer = np.asarray(candidate_group["rings"][0], dtype=np.float32)
                        if outer.ndim != 2 or outer.shape[0] < int(task_schema.min_points_per_feature):
                            continue
                        if not feature_intersects_bbox(outer, crop_bbox_tuple):
                            continue
                        clipped_groups = clip_polygon_rings_to_bbox(
                            rings_xy=candidate_group["rings"],
                            bbox=crop_bbox_tuple,
                        )
                        for clipped_group in clipped_groups:
                            if not clipped_group or clipped_group[0].shape[0] < int(task_schema.min_points_per_feature):
                                continue
                            cropped_groups.append(
                                {
                                    "rings": clipped_group,
                                    "clipped": bool(candidate_group["clipped"]) or not self._rings_equal(candidate_group["rings"], clipped_group),
                                }
                            )
                    candidate_groups = cropped_groups
                if state_bboxes:
                    state_groups: List[Dict] = []
                    for candidate_group in candidate_groups:
                        outer = np.asarray(candidate_group["rings"][0], dtype=np.float32)
                        if outer.ndim != 2 or outer.shape[0] < int(task_schema.min_points_per_feature):
                            continue
                        for state_bbox in state_bboxes:
                            if not feature_intersects_bbox(outer, state_bbox):
                                continue
                            clipped_groups = clip_polygon_rings_to_bbox(
                                rings_xy=candidate_group["rings"],
                                bbox=state_bbox,
                            )
                            for clipped_group in clipped_groups:
                                if not clipped_group or clipped_group[0].shape[0] < int(task_schema.min_points_per_feature):
                                    continue
                                state_groups.append(
                                    {
                                        "rings": clipped_group,
                                        "clipped": True,
                                    }
                                )
                    candidate_groups = state_groups
                for candidate_group in candidate_groups:
                    rings_group = [np.asarray(ring, dtype=np.float32) for ring in candidate_group["rings"]]
                    if interval_px is not None:
                        resampled_group = []
                        for ring in rings_group:
                            resampled_ring = resample_feature_points(
                                points_xy=ring,
                                interval_px=float(interval_px),
                                max_points=max(4, int(ring.shape[0] * 2)),
                                closed=True,
                            )
                            if resampled_ring.shape[0] >= int(task_schema.min_points_per_feature):
                                resampled_group.append(resampled_ring)
                        rings_group = resampled_group
                    if not rings_group or rings_group[0].shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    out.append(
                        {
                            "properties": dict(feature.get("properties", {})),
                            "points": rings_group[0].astype(np.float32),
                            "rings": [ring.astype(np.float32) for ring in rings_group],
                            "clipped": bool(candidate_group.get("clipped", False)),
                        }
                    )
                    if max_features_limit > 0 and len(out) >= max_features_limit:
                        out.sort(key=lambda record: self._feature_record_output_sort_key(record, task_schema))
                        return out
                continue

            pts_original = np.asarray(feature.get("points", []), dtype=np.float32)
            if pts_original.ndim != 2 or pts_original.shape[0] < int(task_schema.min_points_per_feature):
                continue
            trusted_segments = self._line_mask_segments(
                points_xy=pts_original,
                task_schema=task_schema,
                review_mask=review_mask,
            )
            for trusted_segment in trusted_segments:
                source_points = np.asarray(trusted_segment["points"], dtype=np.float32)
                if source_points.ndim != 2 or source_points.shape[0] < int(task_schema.min_points_per_feature):
                    continue
                candidate_segments: List[Dict] = [trusted_segment]
                if crop_bbox_tuple is not None:
                    cropped_segments: List[Dict] = []
                    if feature_intersects_bbox(source_points, crop_bbox_tuple):
                        clipped_pieces = clip_feature_to_bbox(
                            points_xy=source_points,
                            bbox=crop_bbox_tuple,
                            geometry_type=task_schema.geometry_type,
                        )
                        for clipped_piece in clipped_pieces:
                            if clipped_piece.shape[0] < int(task_schema.min_points_per_feature):
                                continue
                            new_cut_start, new_cut_end = self._line_piece_cut_flags_after_clip(
                                source_points=source_points,
                                clipped_points=clipped_piece,
                                cut_start=bool(trusted_segment.get("cut_start", False)),
                                cut_end=bool(trusted_segment.get("cut_end", False)),
                            )
                            cropped_segments.append(
                                {
                                    "points": clipped_piece.astype(np.float32),
                                    "cut_start": bool(new_cut_start),
                                    "cut_end": bool(new_cut_end),
                                }
                            )
                    candidate_segments = cropped_segments
                if state_bboxes:
                    state_segments: List[Dict] = []
                    for candidate_segment in candidate_segments:
                        candidate_points = np.asarray(candidate_segment["points"], dtype=np.float32)
                        for state_bbox in state_bboxes:
                            if not feature_intersects_bbox(candidate_points, state_bbox):
                                continue
                            clipped_pieces = clip_feature_to_bbox(
                                points_xy=candidate_points,
                                bbox=state_bbox,
                                geometry_type=task_schema.geometry_type,
                            )
                            for clipped_piece in clipped_pieces:
                                if clipped_piece.shape[0] < int(task_schema.min_points_per_feature):
                                    continue
                                new_cut_start, new_cut_end = self._line_piece_cut_flags_after_clip(
                                    source_points=candidate_points,
                                    clipped_points=clipped_piece,
                                    cut_start=bool(candidate_segment.get("cut_start", False)),
                                    cut_end=bool(candidate_segment.get("cut_end", False)),
                                )
                                state_segments.append(
                                    {
                                        "points": clipped_piece.astype(np.float32),
                                        "cut_start": bool(new_cut_start),
                                        "cut_end": bool(new_cut_end),
                                    }
                                )
                    candidate_segments = state_segments
                for candidate_segment in candidate_segments:
                    candidate = np.asarray(candidate_segment["points"], dtype=np.float32)
                    if candidate.shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    if interval_px is not None:
                        candidate = resample_feature_points(
                            points_xy=candidate,
                            interval_px=float(interval_px),
                            max_points=max(4, int(candidate.shape[0] * 2)),
                            closed=closed,
                        )
                    if candidate.shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    out.append(
                        {
                            "properties": dict(feature.get("properties", {})),
                            "points": candidate.astype(np.float32),
                            "cut_start": bool(candidate_segment.get("cut_start", False)),
                            "cut_end": bool(candidate_segment.get("cut_end", False)),
                            "clipped": bool(candidate_segment.get("cut_start", False) or candidate_segment.get("cut_end", False)),
                        }
                    )
                    if max_features_limit > 0 and len(out) >= max_features_limit:
                        out.sort(key=lambda record: self._feature_record_output_sort_key(record, task_schema))
                        return out
        out.sort(key=lambda record: self._feature_record_output_sort_key(record, task_schema))
        return out

    def _filter_candidate_with_review_mask(
        self,
        candidate: np.ndarray,
        task_schema: TaskSchema,
        review_mask: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        pts = np.asarray(candidate, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < int(task_schema.min_points_per_feature):
            return []
        if review_mask is None or not bool(self.cfg.feature_filter_by_review_mask):
            return [pts.astype(np.float32)]
        height, width = review_mask.shape[:2]
        cols = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, width - 1)
        rows = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, height - 1)
        inside = review_mask[rows, cols] > 0
        if task_schema.geometry_type == "linestring":
            out: List[np.ndarray] = []
            start = None
            for idx, flag in enumerate(inside.tolist()):
                if flag and start is None:
                    start = idx
                if (not flag) and start is not None:
                    piece = pts[start:idx]
                    if piece.shape[0] >= int(task_schema.min_points_per_feature):
                        out.append(piece.astype(np.float32))
                    start = None
            if start is not None:
                piece = pts[start:]
                if piece.shape[0] >= int(task_schema.min_points_per_feature):
                    out.append(piece.astype(np.float32))
            return out
        inside_ratio = float(inside.mean()) if inside.size > 0 else 0.0
        if inside_ratio >= float(self.cfg.feature_mask_min_inside_ratio):
            return [pts.astype(np.float32)]
        return []

    def _append_task_items(
        self,
        sample_id: str,
        sample_dir: str,
        image_path: str,
        review_mask_path: Optional[str],
        label_path: str,
        task_name: str,
        task_schema: TaskSchema,
        feature_key: str,
        tile_windows: Sequence[Optional[Dict]],
        tile_neighbors: Sequence[Dict[str, bool]],
        crop_bbox: Optional[Sequence[int]],
    ) -> None:
        raw_features = self.features_by_key[feature_key]
        tile_count = len(tile_windows)
        for tile_index, tile_window in enumerate(tile_windows):
            if tile_window is None:
                effective_crop_bbox = None if crop_bbox is None else tuple(int(v) for v in crop_bbox)
            else:
                effective_crop_bbox = (
                    int(tile_window["x0"]),
                    int(tile_window["y0"]),
                    int(tile_window["x1"]),
                    int(tile_window["y1"]),
                )
            has_feature = any(
                feature_intersects_bbox(np.asarray(feature["points"], dtype=np.float32), effective_crop_bbox)
                for feature in raw_features
            ) if effective_crop_bbox is not None else len(raw_features) > 0
            if not bool(self.cfg.tile_allow_empty) and not has_feature:
                continue
            neighbors = tile_neighbors[tile_index] if tile_index < len(tile_neighbors) else {"left": False, "top": False}
            self.items.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "image_path": image_path,
                    "review_mask_path": review_mask_path,
                    "label_path": label_path,
                    "task_name": task_name,
                    "task_schema": task_schema,
                    "feature_key": feature_key,
                    "crop_bbox": effective_crop_bbox,
                    "tile_window": tile_window,
                    "tile_index": int(tile_index),
                    "tile_count": int(tile_count),
                    "state_has_left": bool(neighbors.get("left", False)),
                    "state_has_top": bool(neighbors.get("top", False)),
                    "cache_path": self._build_cache_path(
                        sample_id=sample_id,
                        task_name=task_name,
                        tile_index=int(tile_index),
                        crop_bbox=effective_crop_bbox,
                        state_has_left=bool(neighbors.get("left", False)),
                        state_has_top=bool(neighbors.get("top", False)),
                    ),
                }
            )

    def _build_cache_path(
        self,
        sample_id: str,
        task_name: str,
        tile_index: int,
        crop_bbox: Optional[Sequence[int]],
        state_has_left: bool,
        state_has_top: bool,
    ) -> str:
        if not self.cache_root:
            return ""
        crop_text = "full" if crop_bbox is None else "_".join(str(int(v)) for v in crop_bbox)
        key = "|".join(
            [
                str(self.cfg.cache_namespace),
                str(self.cfg.stage),
                str(self.cfg.split),
                str(sample_id),
                str(task_name),
                str(int(tile_index)),
                crop_text,
                str(int(self.cfg.image_size)),
                str(int(self.cfg.tile_size_px)),
                str(int(self.cfg.tile_overlap_px)),
                str(int(self.cfg.tile_keep_margin_px)),
                str(self.cfg.sample_interval_meter),
                str(int(self.cfg.mask_threshold)),
                str(int(bool(self.cfg.feature_filter_by_review_mask))),
                str(int(bool(state_has_left))),
                str(int(bool(state_has_top))),
                str(int(self.cfg.state_border_margin_px)),
                str(int(self.cfg.state_max_features)),
            ]
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        return os.path.join(
            self.cache_root,
            str(self.cfg.split),
            str(sample_id),
            str(task_name),
            f"tile_{int(tile_index):04d}_{digest}.pt",
        )


class GeoVectorCollator:
    def __init__(
        self,
        map_tokenizer,
        image_size: int,
        prompt_max_tokens: Optional[int] = None,
        state_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
    ) -> None:
        self.map_tokenizer = map_tokenizer
        self.image_size = int(image_size)
        self.prompt_max_tokens = prompt_max_tokens
        self.state_max_tokens = state_max_tokens
        self.target_max_tokens = target_max_tokens
        self.pad_id = int(map_tokenizer.pad_token_id)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        prompt_ids = [
            self.map_tokenizer.encode_prompt(b["prompt_text"], max_length=self.prompt_max_tokens)
            for b in batch
        ]
        state_ids = [
            self.map_tokenizer.encode_text(
                b.get("state_text", ""),
                max_length=self.state_max_tokens,
                append_eos=True,
            )
            for b in batch
        ]
        target_ids = [
            self.map_tokenizer.encode_text(
                b.get("target_text", ""),
                max_length=self.target_max_tokens,
                append_eos=True,
            )
            for b in batch
        ]
        prompt_max = max((len(x) for x in prompt_ids), default=0)
        state_max = max((len(x) for x in state_ids), default=0)
        target_max = max((len(x) for x in target_ids), default=0)

        prompt_tensor = torch.full((len(batch), prompt_max), self.pad_id, dtype=torch.long)
        prompt_mask = torch.zeros((len(batch), prompt_max), dtype=torch.long)
        state_tensor = torch.full((len(batch), state_max), self.pad_id, dtype=torch.long)
        state_mask = torch.zeros((len(batch), state_max), dtype=torch.long)
        target_tensor = torch.full((len(batch), target_max), self.pad_id, dtype=torch.long)
        target_mask = torch.zeros((len(batch), target_max), dtype=torch.long)

        for i, (p_ids, s_ids, t_ids) in enumerate(zip(prompt_ids, state_ids, target_ids)):
            if p_ids:
                prompt_tensor[i, : len(p_ids)] = torch.tensor(p_ids, dtype=torch.long)
                prompt_mask[i, : len(p_ids)] = 1
            if s_ids:
                state_tensor[i, : len(s_ids)] = torch.tensor(s_ids, dtype=torch.long)
                state_mask[i, : len(s_ids)] = 1
            if t_ids:
                target_tensor[i, : len(t_ids)] = torch.tensor(t_ids, dtype=torch.long)
                target_mask[i, : len(t_ids)] = 1

        return {
            "image": images,
            "prompt_input_ids": prompt_tensor,
            "prompt_attention_mask": prompt_mask,
            "state_input_ids": state_tensor,
            "state_attention_mask": state_mask,
            "map_input_ids": target_tensor,
            "map_attention_mask": target_mask,
            "sample_ids": [b["sample_id"] for b in batch],
            "sample_dirs": [b["sample_dir"] for b in batch],
            "task_names": [b["task_name"] for b in batch],
            "prompt_texts": [b["prompt_text"] for b in batch],
            "state_texts": [b.get("state_text", "") for b in batch],
            "target_texts": [b.get("target_text", "") for b in batch],
            "state_items_list": [b["state_items"] for b in batch],
            "target_items_list": [b["target_items"] for b in batch],
            "state_feature_records_list": [b.get("state_feature_records", []) for b in batch],
            "target_feature_records_list": [b.get("target_feature_records", []) for b in batch],
            "raster_metas": [b["raster_meta"] for b in batch],
            "resize_ctxs": [b["resize_ctx"] for b in batch],
            "review_mask_paths": [b["review_mask_path"] for b in batch],
            "label_paths": [b["label_path"] for b in batch],
            "image_paths": [b["image_path"] for b in batch],
            "crop_bboxes": [b["crop_bbox"] for b in batch],
            "tile_windows": [b["tile_window"] for b in batch],
            "tile_indices": [int(b["tile_index"]) for b in batch],
            "tile_counts": [int(b["tile_count"]) for b in batch],
            "state_has_left": [bool(b["state_has_left"]) for b in batch],
            "state_has_top": [bool(b["state_has_top"]) for b in batch],
        }
