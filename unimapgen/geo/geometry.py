from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon, box


@dataclass
class ResizeContext:
    target_size: int
    original_width: int
    original_height: int
    crop_x0: int
    crop_y0: int
    crop_width: int
    crop_height: int
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int

    @property
    def scale_x(self) -> float:
        return float(self.resized_width) / float(max(1, self.crop_width))

    @property
    def scale_y(self) -> float:
        return float(self.resized_height) / float(max(1, self.crop_height))

    def to_dict(self) -> dict:
        return {
            "target_size": int(self.target_size),
            "original_width": int(self.original_width),
            "original_height": int(self.original_height),
            "crop_x0": int(self.crop_x0),
            "crop_y0": int(self.crop_y0),
            "crop_width": int(self.crop_width),
            "crop_height": int(self.crop_height),
            "resized_width": int(self.resized_width),
            "resized_height": int(self.resized_height),
            "pad_x": int(self.pad_x),
            "pad_y": int(self.pad_y),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ResizeContext":
        return cls(
            target_size=int(raw["target_size"]),
            original_width=int(raw["original_width"]),
            original_height=int(raw["original_height"]),
            crop_x0=int(raw["crop_x0"]),
            crop_y0=int(raw["crop_y0"]),
            crop_width=int(raw["crop_width"]),
            crop_height=int(raw["crop_height"]),
            resized_width=int(raw["resized_width"]),
            resized_height=int(raw["resized_height"]),
            pad_x=int(raw["pad_x"]),
            pad_y=int(raw["pad_y"]),
        )


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int
    keep_x0: int
    keep_y0: int
    keep_x1: int
    keep_y1: int
    mask_ratio: float = 0.0
    mask_pixels: int = 0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)

    @property
    def keep_bbox(self) -> Tuple[int, int, int, int]:
        return int(self.keep_x0), int(self.keep_y0), int(self.keep_x1), int(self.keep_y1)

    def to_dict(self) -> dict:
        return {
            "x0": int(self.x0),
            "y0": int(self.y0),
            "x1": int(self.x1),
            "y1": int(self.y1),
            "keep_x0": int(self.keep_x0),
            "keep_y0": int(self.keep_y0),
            "keep_x1": int(self.keep_x1),
            "keep_y1": int(self.keep_y1),
            "mask_ratio": float(self.mask_ratio),
            "mask_pixels": int(self.mask_pixels),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "TileWindow":
        return cls(
            x0=int(raw["x0"]),
            y0=int(raw["y0"]),
            x1=int(raw["x1"]),
            y1=int(raw["y1"]),
            keep_x0=int(raw["keep_x0"]),
            keep_y0=int(raw["keep_y0"]),
            keep_x1=int(raw["keep_x1"]),
            keep_y1=int(raw["keep_y1"]),
            mask_ratio=float(raw.get("mask_ratio", 0.0)),
            mask_pixels=int(raw.get("mask_pixels", 0)),
        )


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def expand_bbox(
    bbox: Optional[Tuple[int, int, int, int]],
    pad_px: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    if bbox is None:
        return 0, 0, int(width), int(height)
    x0, y0, x1, y1 = bbox
    pad = max(0, int(pad_px))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(int(width), x1 + pad),
        min(int(height), y1 + pad),
    )


def generate_tile_windows(
    width: int,
    height: int,
    tile_size_px: int,
    overlap_px: int,
    region_bbox: Optional[Tuple[int, int, int, int]] = None,
    keep_margin_px: int = 0,
) -> List[TileWindow]:
    width = int(width)
    height = int(height)
    tile_size_px = max(1, int(tile_size_px))
    overlap_px = max(0, min(int(overlap_px), tile_size_px - 1))
    stride = max(1, tile_size_px - overlap_px)
    if region_bbox is None:
        region_bbox = (0, 0, width, height)
    rx0, ry0, rx1, ry1 = [int(v) for v in region_bbox]
    x_positions = _sliding_positions(start=rx0, end=rx1, tile_size=tile_size_px, limit=width, stride=stride)
    y_positions = _sliding_positions(start=ry0, end=ry1, tile_size=tile_size_px, limit=height, stride=stride)
    windows: List[TileWindow] = []
    seen = set()
    for y0 in y_positions:
        for x0 in x_positions:
            x1 = min(width, x0 + tile_size_px)
            y1 = min(height, y0 + tile_size_px)
            key = (int(x0), int(y0), int(x1), int(y1))
            if key in seen:
                continue
            seen.add(key)
            keep_bbox = _compute_keep_bbox(
                bbox=key,
                width=width,
                height=height,
                keep_margin_px=int(keep_margin_px),
            )
            windows.append(
                TileWindow(
                    x0=int(x0),
                    y0=int(y0),
                    x1=int(x1),
                    y1=int(y1),
                    keep_x0=int(keep_bbox[0]),
                    keep_y0=int(keep_bbox[1]),
                    keep_x1=int(keep_bbox[2]),
                    keep_y1=int(keep_bbox[3]),
                )
            )
    return windows


def annotate_tile_windows_with_mask(
    tile_windows: Sequence[TileWindow],
    mask: Optional[np.ndarray],
) -> List[TileWindow]:
    if mask is None:
        return [TileWindow.from_dict(window.to_dict()) for window in tile_windows]
    out: List[TileWindow] = []
    for window in tile_windows:
        x0, y0, x1, y1 = window.bbox
        crop = mask[y0:y1, x0:x1]
        mask_pixels = int(crop.sum()) if crop.size > 0 else 0
        mask_ratio = float(crop.mean()) if crop.size > 0 else 0.0
        out.append(
            TileWindow(
                x0=window.x0,
                y0=window.y0,
                x1=window.x1,
                y1=window.y1,
                keep_x0=window.keep_x0,
                keep_y0=window.keep_y0,
                keep_x1=window.keep_x1,
                keep_y1=window.keep_y1,
                mask_ratio=mask_ratio,
                mask_pixels=mask_pixels,
            )
        )
    return out


def select_tile_windows(
    tile_windows: Sequence[TileWindow],
    min_mask_ratio: float,
    min_mask_pixels: int,
    max_tiles: Optional[int] = None,
    fallback_to_all_if_empty: bool = True,
) -> List[TileWindow]:
    selected = [
        window
        for window in tile_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    if not selected and bool(fallback_to_all_if_empty):
        selected = list(tile_windows)
    selected.sort(key=lambda window: (float(window.mask_ratio), int(window.mask_pixels)), reverse=True)
    if max_tiles is not None and int(max_tiles) > 0:
        selected = selected[: int(max_tiles)]
    return selected


def audit_tile_window_selection(
    tile_windows: Sequence[TileWindow],
    min_mask_ratio: float,
    min_mask_pixels: int,
    max_tiles: Optional[int] = None,
    fallback_to_all_if_empty: bool = True,
) -> Tuple[List[TileWindow], List[dict]]:
    all_windows = list(tile_windows)
    filtered = [
        window
        for window in all_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    used_fallback = len(filtered) == 0 and bool(fallback_to_all_if_empty)
    candidates = filtered if filtered else (list(all_windows) if bool(fallback_to_all_if_empty) else [])
    candidates = sorted(candidates, key=lambda window: (float(window.mask_ratio), int(window.mask_pixels)), reverse=True)
    selected = list(candidates)
    if max_tiles is not None and int(max_tiles) > 0:
        selected = selected[: int(max_tiles)]

    selected_keys = {window.bbox for window in selected}
    candidate_keys = {window.bbox for window in candidates}
    audits: List[dict] = []
    for index, window in enumerate(all_windows):
        key = window.bbox
        is_selected = key in selected_keys
        if is_selected:
            reason = "selected"
        elif (not used_fallback) and key not in candidate_keys:
            reason = "below_mask_threshold"
        elif max_tiles is not None and int(max_tiles) > 0 and key in candidate_keys:
            reason = "truncated_by_max_tiles"
        else:
            reason = "discarded"
        audits.append(
            {
                "candidate_index": int(index),
                "selected": bool(is_selected),
                "reason": str(reason),
                "bbox": [int(v) for v in window.bbox],
                "keep_bbox": [int(v) for v in window.keep_bbox],
                "mask_ratio": float(window.mask_ratio),
                "mask_pixels": int(window.mask_pixels),
            }
        )
    return selected, audits


def build_resize_context(
    width: int,
    height: int,
    target_size: int,
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> ResizeContext:
    if crop_bbox is None:
        crop_bbox = (0, 0, int(width), int(height))
    x0, y0, x1, y1 = crop_bbox
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    scale = min(float(target_size) / float(crop_w), float(target_size) / float(crop_h))
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    pad_x = max(0, int((target_size - resized_w) // 2))
    pad_y = max(0, int((target_size - resized_h) // 2))
    return ResizeContext(
        target_size=int(target_size),
        original_width=int(width),
        original_height=int(height),
        crop_x0=int(x0),
        crop_y0=int(y0),
        crop_width=int(crop_w),
        crop_height=int(crop_h),
        resized_width=int(resized_w),
        resized_height=int(resized_h),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
    )


def transform_points_to_model(points_xy: np.ndarray, ctx: ResizeContext) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] - float(ctx.crop_x0)) * float(ctx.scale_x) + float(ctx.pad_x)
    pts[:, 1] = (pts[:, 1] - float(ctx.crop_y0)) * float(ctx.scale_y) + float(ctx.pad_y)
    return pts


def transform_points_to_original(points_xy: np.ndarray, ctx: ResizeContext) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] - float(ctx.pad_x)) / max(float(ctx.scale_x), 1e-6) + float(ctx.crop_x0)
    pts[:, 1] = (pts[:, 1] - float(ctx.pad_y)) / max(float(ctx.scale_y), 1e-6) + float(ctx.crop_y0)
    return pts


def clip_points_to_image(points_xy: np.ndarray, size: int) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(size - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(size - 1))
    return pts


def _resample_path(points_xy: np.ndarray, interval_px: float, max_points: int, closed: bool) -> np.ndarray:
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if closed and arr.shape[0] >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if arr.shape[0] <= 1:
        return arr[:1].astype(np.float32)

    pts = arr
    if closed:
        pts = np.concatenate([arr, arr[:1]], axis=0)

    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return arr[:1].astype(np.float32)

    step = max(float(interval_px), 1e-3)
    n = int(np.floor(total / step)) + 1
    min_points = 3 if closed else 2
    n = max(min_points, min(n, int(max_points)))
    targets = np.linspace(0.0, total, n, dtype=np.float32)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    sampled = []
    for target in targets:
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        t0 = float(cum[j])
        t1 = float(cum[j + 1])
        ratio = 0.0 if t1 <= t0 else (float(target) - t0) / (t1 - t0)
        point = pts[j] * (1.0 - ratio) + pts[j + 1] * ratio
        sampled.append(point)
    out = np.asarray(sampled, dtype=np.float32)
    if closed and out.shape[0] >= 2 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return out


def resample_feature_points(
    points_xy: np.ndarray,
    interval_px: float,
    max_points: int,
    closed: bool,
) -> np.ndarray:
    out = _resample_path(
        points_xy=points_xy,
        interval_px=interval_px,
        max_points=max_points,
        closed=closed,
    )
    if out.shape[0] > int(max_points):
        out = out[: int(max_points)]
    return out.astype(np.float32)


def apply_square_augment(
    image_chw: np.ndarray,
    feature_points: Sequence[np.ndarray],
    rot90_k: int = 0,
    hflip: bool = False,
    vflip: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    image = np.asarray(image_chw, dtype=np.float32)
    points_out = [np.asarray(points, dtype=np.float32).copy() for points in feature_points]
    size = int(image.shape[-1])

    if int(rot90_k) % 4:
        k = int(rot90_k) % 4
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        for pts in points_out:
            for _ in range(k):
                x = pts[:, 0].copy()
                y = pts[:, 1].copy()
                pts[:, 0] = y
                pts[:, 1] = (size - 1) - x

    if bool(hflip):
        image = image[:, :, ::-1].copy()
        for pts in points_out:
            pts[:, 0] = (size - 1) - pts[:, 0]

    if bool(vflip):
        image = image[:, ::-1, :].copy()
        for pts in points_out:
            pts[:, 1] = (size - 1) - pts[:, 1]

    return image, points_out


def filter_points_inside_bbox(points_xy: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    if points_xy.size == 0:
        return False
    x0, y0, x1, y1 = bbox
    pts = np.asarray(points_xy, dtype=np.float32)
    inside_x = (pts[:, 0] >= float(x0)) & (pts[:, 0] < float(x1))
    inside_y = (pts[:, 1] >= float(y0)) & (pts[:, 1] < float(y1))
    return bool(np.any(inside_x & inside_y))


def feature_intersects_bbox(points_xy: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return False
    px0 = float(np.min(pts[:, 0]))
    py0 = float(np.min(pts[:, 1]))
    px1 = float(np.max(pts[:, 0]))
    py1 = float(np.max(pts[:, 1]))
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return not (px1 < x0 or px0 >= x1 or py1 < y0 or py0 >= y1)


def clip_feature_to_bbox(
    points_xy: np.ndarray,
    bbox: Tuple[int, int, int, int],
    geometry_type: str,
) -> List[np.ndarray]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return []
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bbox_geom = box(x0, y0, x1, y1)
    geometry_type = str(geometry_type).strip().lower()
    try:
        if geometry_type == "polygon":
            coords = pts.tolist()
            if coords[0] != coords[-1]:
                coords.append(list(coords[0]))
            geom = Polygon(coords)
        else:
            geom = LineString(pts.tolist())
        clipped = geom.intersection(bbox_geom)
    except Exception:
        return []
    return _shapely_to_point_arrays(clipped=clipped, geometry_type=geometry_type)


def clip_polygon_rings_to_bbox(
    rings_xy: Sequence[np.ndarray],
    bbox: Tuple[int, int, int, int],
) -> List[List[np.ndarray]]:
    rings = [np.asarray(ring, dtype=np.float32) for ring in rings_xy if np.asarray(ring, dtype=np.float32).ndim == 2]
    if not rings:
        return []
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bbox_geom = box(x0, y0, x1, y1)
    try:
        exterior = rings[0].tolist()
        holes = [ring.tolist() for ring in rings[1:] if ring.shape[0] >= 3]
        if exterior and exterior[0] != exterior[-1]:
            exterior.append(list(exterior[0]))
        normalized_holes = []
        for hole in holes:
            if hole and hole[0] != hole[-1]:
                hole = hole + [list(hole[0])]
            normalized_holes.append(hole)
        geom = Polygon(exterior, holes=normalized_holes)
        clipped = geom.intersection(bbox_geom)
    except Exception:
        return []
    return _shapely_to_polygon_ring_lists(clipped)


def feature_center_inside_bbox(points_xy: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return False
    center = np.mean(pts, axis=0)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return bool(x0 <= float(center[0]) < x1 and y0 <= float(center[1]) < y1)


def _sliding_positions(start: int, end: int, tile_size: int, limit: int, stride: int) -> List[int]:
    start = max(0, int(start))
    end = min(int(limit), int(end))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if end <= start:
        return [0]
    if end - start <= tile_size:
        pos = min(max(0, start), max(0, limit - tile_size))
        return [int(pos)]
    positions = []
    pos = int(start)
    last_start = min(max(0, end - tile_size), max(0, limit - tile_size))
    while pos < last_start:
        positions.append(int(pos))
        pos += stride
    positions.append(int(last_start))
    return sorted(set(positions))


def _compute_keep_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    keep_margin_px: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    margin = max(0, int(keep_margin_px))
    keep_x0 = x0 if x0 <= 0 else min(x1, x0 + margin)
    keep_y0 = y0 if y0 <= 0 else min(y1, y0 + margin)
    keep_x1 = x1 if x1 >= int(width) else max(x0, x1 - margin)
    keep_y1 = y1 if y1 >= int(height) else max(y0, y1 - margin)
    return keep_x0, keep_y0, keep_x1, keep_y1


def _shapely_to_point_arrays(clipped, geometry_type: str) -> List[np.ndarray]:
    if clipped.is_empty:
        return []
    geometry_type = str(geometry_type).strip().lower()
    arrays: List[np.ndarray] = []
    if geometry_type == "polygon":
        geoms = getattr(clipped, "geoms", [clipped])
        for geom in geoms:
            if geom.geom_type != "Polygon":
                continue
            coords = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
            if coords.ndim == 2 and coords.shape[0] >= 3:
                arrays.append(coords)
        return arrays
    geoms = getattr(clipped, "geoms", [clipped])
    for geom in geoms:
        if geom.geom_type != "LineString":
            continue
        coords = np.asarray(geom.coords, dtype=np.float32)
        if coords.ndim == 2 and coords.shape[0] >= 2:
            arrays.append(coords)
    return arrays


def _shapely_to_polygon_ring_lists(clipped) -> List[List[np.ndarray]]:
    if clipped.is_empty:
        return []
    geoms = getattr(clipped, "geoms", [clipped])
    ring_lists: List[List[np.ndarray]] = []
    for geom in geoms:
        if geom.geom_type != "Polygon":
            continue
        rings: List[np.ndarray] = []
        exterior = np.asarray(geom.exterior.coords[:-1], dtype=np.float32)
        if exterior.ndim == 2 and exterior.shape[0] >= 3:
            rings.append(exterior)
        for interior in list(geom.interiors):
            hole = np.asarray(interior.coords[:-1], dtype=np.float32)
            if hole.ndim == 2 and hole.shape[0] >= 3:
                rings.append(hole)
        if rings:
            ring_lists.append(rings)
    return ring_lists
