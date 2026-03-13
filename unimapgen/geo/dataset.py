from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .coord_sequence import (
    points_abs_to_uv,
    rings_abs_to_uv,
    uv_feature_records_to_state_items,
    uv_feature_records_to_target_items,
)
from .errors import raise_geo_error
from .geometry import (
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
from .io import geojson_to_pixel_features, load_geojson, read_binary_mask, read_raster_meta, read_rgb_geotiff
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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        image_hwc, raster_meta = read_rgb_geotiff(
            path=item["image_path"],
            band_indices=self.cfg.band_indices,
            crop_bbox=item["crop_bbox"],
        )
        review_mask = None
        if bool(self.cfg.feature_filter_by_review_mask) and item["review_mask_path"] and os.path.isfile(item["review_mask_path"]):
            review_mask = read_binary_mask(path=item["review_mask_path"], threshold=self.cfg.mask_threshold)
        image_hwc = np.clip(image_hwc, 0.0, 255.0)
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
        prompt_text = item["task_schema"].prompt_template
        if self.cfg.state_enabled:
            if state_items:
                prompt_text = f"{prompt_text} {self.cfg.prompt_with_state}".strip()
            else:
                prompt_text = f"{prompt_text} {self.cfg.prompt_without_state}".strip()

        return {
            "image": torch.from_numpy(image_chw).float(),
            "sample_id": item["sample_id"],
            "sample_dir": item["sample_dir"],
            "task_name": item["task_name"],
            "prompt_text": prompt_text,
            "state_items": state_items,
            "target_items": target_items,
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

    def _feature_records_to_uv(self, feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
        out = []
        for feature in feature_records:
            points_abs = np.asarray(feature.get("points", []), dtype=np.float32)
            points_uv = points_abs_to_uv(points_xy=points_abs, resize_ctx=resize_ctx)
            record = {"properties": dict(feature.get("properties", {})), "points": points_uv.astype(np.float32)}
            if feature.get("rings"):
                record["rings"] = rings_abs_to_uv(feature.get("rings", []), resize_ctx=resize_ctx)
            out.append(record)
        return out

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
        points = [np.asarray(feature.get("points", []), dtype=np.float32).copy() for feature in feature_records]
        aug_image, aug_points = apply_square_augment(
            image_chw=image_chw,
            feature_points=points,
            rot90_k=rot_k,
            hflip=hflip,
            vflip=vflip,
        )
        out = []
        for feature, pts in zip(feature_records, aug_points):
            out.append({"properties": dict(feature.get("properties", {})), "points": pts.astype(np.float32)})
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

        for feature in raw_features:
            if task_schema.geometry_type == "polygon" and feature.get("rings"):
                rings_original = [np.asarray(ring, dtype=np.float32) for ring in feature.get("rings", [])]
                pts_original = np.asarray(rings_original[0], dtype=np.float32) if rings_original else np.zeros((0, 2), dtype=np.float32)
                if pts_original.ndim != 2 or pts_original.shape[0] < int(task_schema.min_points_per_feature):
                    continue
                if crop_bbox_tuple is not None and not feature_intersects_bbox(pts_original, crop_bbox_tuple):
                    continue
                ring_groups = [rings_original] if crop_bbox_tuple is None else clip_polygon_rings_to_bbox(
                    rings_xy=rings_original,
                    bbox=crop_bbox_tuple,
                )
                for ring_group in ring_groups:
                    if not ring_group or ring_group[0].shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    candidate_groups = [ring_group]
                    if state_bboxes:
                        candidate_groups = []
                        for state_bbox in state_bboxes:
                            if not feature_intersects_bbox(ring_group[0], state_bbox):
                                continue
                            candidate_groups.extend(
                                clip_polygon_rings_to_bbox(
                                    rings_xy=ring_group,
                                    bbox=state_bbox,
                                )
                            )
                    for candidate_group in candidate_groups:
                        if not candidate_group or candidate_group[0].shape[0] < int(task_schema.min_points_per_feature):
                            continue
                        if interval_px is not None:
                            resampled_group = []
                            for ring in candidate_group:
                                resampled_ring = resample_feature_points(
                                    points_xy=ring,
                                    interval_px=float(interval_px),
                                    max_points=max(4, int(ring.shape[0] * 2)),
                                    closed=True,
                                )
                                if resampled_ring.shape[0] >= int(task_schema.min_points_per_feature):
                                    resampled_group.append(resampled_ring)
                            candidate_group = resampled_group
                        if not candidate_group or candidate_group[0].shape[0] < int(task_schema.min_points_per_feature):
                            continue
                        mask_points = np.concatenate(candidate_group, axis=0)
                        trusted_pieces = self._filter_candidate_with_review_mask(
                            candidate=mask_points,
                            task_schema=task_schema,
                            review_mask=review_mask,
                        )
                        if not trusted_pieces:
                            continue
                        out.append(
                            {
                                "properties": dict(feature.get("properties", {})),
                                "points": candidate_group[0].astype(np.float32),
                                "rings": [ring.astype(np.float32) for ring in candidate_group],
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
                if piece.shape[0] < int(task_schema.min_points_per_feature):
                    continue
                candidate_pieces = [piece]
                if state_bboxes:
                    candidate_pieces = []
                    for state_bbox in state_bboxes:
                        if not feature_intersects_bbox(piece, state_bbox):
                            continue
                        candidate_pieces.extend(
                            clip_feature_to_bbox(
                                points_xy=piece,
                                bbox=state_bbox,
                                geometry_type=task_schema.geometry_type,
                            )
                        )
                for candidate in candidate_pieces:
                    if candidate.shape[0] < int(task_schema.min_points_per_feature):
                        continue
                    if interval_px is not None:
                        candidate = resample_feature_points(
                            points_xy=candidate,
                            interval_px=float(interval_px),
                            max_points=max(4, int(candidate.shape[0] * 2)),
                            closed=closed,
                        )
                    trusted_pieces = self._filter_candidate_with_review_mask(
                        candidate=candidate,
                        task_schema=task_schema,
                        review_mask=review_mask,
                    )
                    for trusted_piece in trusted_pieces:
                        out.append({"properties": dict(feature.get("properties", {})), "points": trusted_piece.astype(np.float32)})
                        if max_features_limit > 0 and len(out) >= max_features_limit:
                            return out
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
                }
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
            self.map_tokenizer.encode_state_items(
                b["state_items"],
                image_size=self.image_size,
                max_length=self.state_max_tokens,
                append_eos=True,
            )
            for b in batch
        ]
        target_ids = [
            self.map_tokenizer.encode_map_items(
                b["target_items"],
                image_size=self.image_size,
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
            "state_items_list": [b["state_items"] for b in batch],
            "target_items_list": [b["target_items"] for b in batch],
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
