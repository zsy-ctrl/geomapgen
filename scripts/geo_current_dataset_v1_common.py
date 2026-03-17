import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_llamafactory_patch_only_from_raw_family_manifest import (
    canonicalize_line_direction,
    clip_polyline_to_rect,
    dedup_points,
    point_boundary_side,
    resample_polyline,
    simplify_for_json,
    sort_lines,
)
from export_llamafactory_state_sft_from_raw_family_manifest import build_state_lines_by_mode


DEFAULT_IMAGE_RELPATH = "patch_tif/0.tif"
DEFAULT_MASK_RELPATH = "patch_tif/0_edit_poly.tif"
DEFAULT_LANE_RELPATH = "label_check_crop/Lane.geojson"
DEFAULT_INTERSECTION_RELPATH = "label_check_crop/Intersection.geojson"
DEFAULT_STAGEA_PROMPT_TEMPLATE = """<image>
Please construct the complete road-structure line map in the current satellite patch."""
DEFAULT_STAGEA_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Predict the complete patch-local line map from the current image.\n"
    "The output JSON schema is {\"lines\": [...]}.\n"
    "Each line must stay in patch-local pixel coordinates.\n"
    "Use category lane_line for roads and intersection_boundary for intersection borders.\n"
    "Return only valid JSON and no extra text."
)
DEFAULT_STAGEB_PROMPT_TEMPLATE = """<image>
Please construct the road-structure line map in the current patch.
The previous state contains cut traces passed from already processed neighboring patches.
Continue those traces when appropriate and also predict all owned line segments for the current patch.
Previous state:
{state_json}"""
DEFAULT_STAGEB_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Use the image and the previous line-map state to predict the current patch.\n"
    "The previous state contains cut traces from already processed neighboring patches.\n"
    "Preserve cross-patch continuity whenever those traces enter the current patch.\n"
    "The output JSON schema is {\"lines\": [...]}.\n"
    "Each line must stay in patch-local pixel coordinates.\n"
    "Use category lane_line for roads and intersection_boundary for intersection borders.\n"
    "Return only valid JSON and no markdown fences."
)


@dataclass
class RasterMeta:
    path: str
    width: int
    height: int
    crs: str
    transform: List[float]

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_rgb_geotiff(path: Path, band_indices: Sequence[int]) -> Tuple[np.ndarray, RasterMeta]:
    with rasterio_open(path) as ds:
        arr = ds.read(indexes=[int(x) for x in band_indices])
        image = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        meta = RasterMeta(
            path=str(path),
            width=int(ds.width),
            height=int(ds.height),
            crs=str(ds.crs) if ds.crs is not None else "",
            transform=[float(x) for x in tuple(ds.transform)[:6]],
        )
    return image, meta


def read_binary_mask(path: Path, threshold: int) -> np.ndarray:
    with rasterio_open(path) as ds:
        arr = ds.read(1)
    return (arr > int(threshold)).astype(np.uint8)


def detect_geojson_crs(geojson_dict: Dict) -> str:
    crs = geojson_dict.get("crs", {})
    props = crs.get("properties", {}) if isinstance(crs, dict) else {}
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return str(name).strip()
    return "urn:ogc:def:crs:OGC:1.3:CRS84"


def build_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def project_coords(coordinates, transformer: Transformer) -> np.ndarray:
    points = []
    for value in coordinates:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        x, y = transformer.transform(float(value[0]), float(value[1]))
        points.append([float(x), float(y)])
    return np.asarray(points, dtype=np.float32)


def world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inv = ~affine
    cols = []
    rows = []
    for x, y in points_world:
        col, row = inv * (float(x), float(y))
        cols.append(float(col))
        rows.append(float(row))
    return np.stack([cols, rows], axis=-1).astype(np.float32)


def compute_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


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


def _sliding_positions(start: int, end: int, tile_size: int, limit: int, stride: int) -> List[int]:
    start = max(0, int(start))
    end = min(int(limit), int(end))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if end - start <= tile_size:
        return [int(start)]
    positions = list(range(int(start), max(int(start), int(end - tile_size)) + 1, stride))
    last = max(int(start), int(end - tile_size))
    if not positions or positions[-1] != last:
        positions.append(last)
    return [int(p) for p in positions]


def _compute_keep_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    keep_margin_px: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    margin = max(0, int(keep_margin_px))
    return (
        max(0, x0 + margin),
        max(0, y0 + margin),
        min(int(width), x1 - margin),
        min(int(height), y1 - margin),
    )


def generate_tile_windows(
    width: int,
    height: int,
    tile_size_px: int,
    overlap_px: int,
    region_bbox: Optional[Tuple[int, int, int, int]],
    keep_margin_px: int,
) -> List[TileWindow]:
    stride = max(1, int(tile_size_px) - int(overlap_px))
    rx0, ry0, rx1, ry1 = (0, 0, int(width), int(height)) if region_bbox is None else tuple(int(v) for v in region_bbox)
    xs = _sliding_positions(start=rx0, end=rx1, tile_size=int(tile_size_px), limit=int(width), stride=int(stride))
    ys = _sliding_positions(start=ry0, end=ry1, tile_size=int(tile_size_px), limit=int(height), stride=int(stride))
    out: List[TileWindow] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(int(width), int(x0 + tile_size_px))
            y1 = min(int(height), int(y0 + tile_size_px))
            keep_bbox = _compute_keep_bbox(
                bbox=(x0, y0, x1, y1),
                width=int(width),
                height=int(height),
                keep_margin_px=int(keep_margin_px),
            )
            out.append(
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
    return out


def annotate_tile_windows_with_mask(tile_windows: Sequence[TileWindow], mask: Optional[np.ndarray]) -> List[TileWindow]:
    if mask is None:
        return list(tile_windows)
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


def audit_tile_window_selection(
    tile_windows: Sequence[TileWindow],
    min_mask_ratio: float,
    min_mask_pixels: int,
    max_tiles: Optional[int],
    fallback_to_all_if_empty: bool,
) -> Tuple[List[TileWindow], List[Dict]]:
    all_windows = list(tile_windows)
    filtered = [
        window
        for window in all_windows
        if float(window.mask_ratio) >= float(min_mask_ratio) or int(window.mask_pixels) >= int(min_mask_pixels)
    ]
    used_fallback = len(filtered) == 0 and bool(fallback_to_all_if_empty)
    candidates = filtered if filtered else (list(all_windows) if bool(fallback_to_all_if_empty) else [])
    candidates = sorted(candidates, key=lambda item: (float(item.mask_ratio), int(item.mask_pixels)), reverse=True)
    selected = list(candidates)
    if max_tiles is not None and int(max_tiles) > 0:
        selected = selected[: int(max_tiles)]
    selected_keys = {window.bbox for window in selected}
    candidate_keys = {window.bbox for window in candidates}
    audits: List[Dict] = []
    for index, window in enumerate(all_windows):
        key = window.bbox
        if key in selected_keys:
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
                "selected": bool(key in selected_keys),
                "reason": str(reason),
                "bbox": [int(v) for v in window.bbox],
                "keep_bbox": [int(v) for v in window.keep_bbox],
                "mask_ratio": float(window.mask_ratio),
                "mask_pixels": int(window.mask_pixels),
            }
        )
    return selected, audits


def _densify_line_for_mask(points_xy: np.ndarray, step_px: float = 1.0) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return pts.astype(np.float32)
    out: List[np.ndarray] = [pts[0].astype(np.float32)]
    step = max(0.25, float(step_px))
    for start_pt, end_pt in zip(pts[:-1], pts[1:]):
        seg = np.asarray(end_pt - start_pt, dtype=np.float32)
        seg_len = float(np.linalg.norm(seg))
        steps = max(1, int(np.ceil(seg_len / step)))
        for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[1:]:
            out.append((start_pt + seg * float(t)).astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def _mask_contains_point(review_mask: np.ndarray, point_xy: np.ndarray) -> bool:
    x = int(np.clip(round(float(point_xy[0])), 0, int(review_mask.shape[1]) - 1))
    y = int(np.clip(round(float(point_xy[1])), 0, int(review_mask.shape[0]) - 1))
    return bool(review_mask[y, x] > 0)


def _refine_mask_transition_point(
    review_mask: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    inside_a: bool,
    inside_b: bool,
    iterations: int = 8,
) -> Optional[np.ndarray]:
    if bool(inside_a) == bool(inside_b):
        return None
    low = np.asarray(point_a, dtype=np.float32).copy()
    high = np.asarray(point_b, dtype=np.float32).copy()
    low_inside = bool(inside_a)
    for _ in range(max(1, int(iterations))):
        mid = ((low + high) * 0.5).astype(np.float32)
        mid_inside = _mask_contains_point(review_mask=review_mask, point_xy=mid)
        if mid_inside == low_inside:
            low = mid
        else:
            high = mid
    return ((low + high) * 0.5).astype(np.float32)


def mask_clip_line(points_xy: np.ndarray, review_mask: Optional[np.ndarray], min_points: int = 2) -> List[Dict]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < int(min_points):
        return []
    if review_mask is None:
        return [{"points": pts.astype(np.float32), "cut_start": False, "cut_end": False}]
    dense_pts = _densify_line_for_mask(points_xy=pts, step_px=1.0)
    height, width = review_mask.shape[:2]
    cols = np.clip(np.round(dense_pts[:, 0]).astype(np.int64), 0, width - 1)
    rows = np.clip(np.round(dense_pts[:, 1]).astype(np.int64), 0, height - 1)
    inside = (review_mask[rows, cols] > 0).tolist()
    out: List[Dict] = []
    start = None
    piece_points: List[np.ndarray] = []
    for idx, flag in enumerate(inside):
        if flag and start is None:
            piece_points = []
            if idx > 0:
                transition = _refine_mask_transition_point(
                    review_mask=review_mask,
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=bool(inside[idx - 1]),
                    inside_b=bool(flag),
                )
                if transition is not None:
                    piece_points.append(transition.astype(np.float32))
            start = idx
        if flag and start is not None:
            if not piece_points or not np.allclose(piece_points[-1], dense_pts[idx], atol=1e-3):
                piece_points.append(dense_pts[idx].astype(np.float32))
        if (not flag) and start is not None:
            if idx > 0:
                transition = _refine_mask_transition_point(
                    review_mask=review_mask,
                    point_a=dense_pts[idx - 1],
                    point_b=dense_pts[idx],
                    inside_a=bool(inside[idx - 1]),
                    inside_b=bool(flag),
                )
                if transition is not None and (
                    not piece_points or not np.allclose(piece_points[-1], transition, atol=1e-3)
                ):
                    piece_points.append(transition.astype(np.float32))
            piece = np.asarray(piece_points, dtype=np.float32)
            if piece.ndim == 2 and piece.shape[0] >= int(min_points):
                out.append(
                    {
                        "points": piece.astype(np.float32),
                        "cut_start": bool(start > 0) or not np.allclose(piece[0], pts[0], atol=1e-3),
                        "cut_end": True,
                    }
                )
            start = None
            piece_points = []
    if start is not None:
        piece = np.asarray(piece_points, dtype=np.float32)
        if piece.ndim == 2 and piece.shape[0] >= int(min_points):
            out.append(
                {
                    "points": piece.astype(np.float32),
                    "cut_start": bool(start > 0) or not np.allclose(piece[0], pts[0], atol=1e-3),
                    "cut_end": not np.allclose(piece[-1], pts[-1], atol=1e-3),
                }
            )
    return out


def _line_piece_cut_flags_after_clip(
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
    return bool(cut_start) or not np.allclose(dst[0], src[0], atol=tol), bool(cut_end) or not np.allclose(
        dst[-1], src[-1], atol=tol
    )


def geojson_lines_to_pixel_lines(geojson_dict: Dict, raster_meta: RasterMeta, category: str) -> List[Dict]:
    src_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
    out: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "linestring":
            continue
        world = project_coords(geometry.get("coordinates", []), transformer=transformer)
        pixel = dedup_points(world_to_pixel(world, affine=raster_meta.affine))
        if pixel.ndim != 2 or pixel.shape[0] < 2:
            continue
        out.append({"category": str(category), "points_global": pixel.astype(np.float32)})
    return out


def geojson_polygon_boundaries_to_pixel_lines(geojson_dict: Dict, raster_meta: RasterMeta, category: str) -> List[Dict]:
    src_crs = detect_geojson_crs(geojson_dict)
    transformer = build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
    out: List[Dict] = []
    for feature in geojson_dict.get("features", []):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry", {})
        if str(geometry.get("type", "")).strip().lower() != "polygon":
            continue
        for ring_coords in geometry.get("coordinates", []):
            world = project_coords(ring_coords, transformer=transformer)
            pixel = world_to_pixel(world, affine=raster_meta.affine)
            if pixel.ndim != 2 or pixel.shape[0] < 2:
                continue
            if np.allclose(pixel[0], pixel[-1], atol=1e-3):
                pixel = pixel[:-1]
            pixel = dedup_points(pixel)
            if pixel.ndim != 2 or pixel.shape[0] < 2:
                continue
            out.append({"category": str(category), "points_global": pixel.astype(np.float32)})
    return out


def load_sample_global_lines(
    sample_dir: Path,
    raster_meta: RasterMeta,
    lane_relpath: str,
    intersection_relpath: str,
    include_lane: bool = True,
    include_intersection: bool = True,
) -> List[Dict]:
    out: List[Dict] = []
    if include_lane:
        lane_path = sample_dir / lane_relpath
        if lane_path.is_file():
            out.extend(geojson_lines_to_pixel_lines(load_json(lane_path), raster_meta=raster_meta, category="lane_line"))
    if include_intersection:
        intersection_path = sample_dir / intersection_relpath
        if intersection_path.is_file():
            out.extend(
                geojson_polygon_boundaries_to_pixel_lines(
                    load_json(intersection_path),
                    raster_meta=raster_meta,
                    category="intersection_boundary",
                )
            )
    return out


def build_patch_segments_global(
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    rect_global: Tuple[float, float, float, float],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    out: List[Dict] = []
    for line in global_lines:
        for masked_piece in mask_clip_line(line["points_global"], review_mask=review_mask, min_points=2):
            source_points = np.asarray(masked_piece["points"], dtype=np.float32)
            for clipped_piece in clip_polyline_to_rect(source_points, rect_global):
                piece = np.asarray(clipped_piece, dtype=np.float32)
                if piece.ndim != 2 or piece.shape[0] < 2:
                    continue
                cut_start, cut_end = _line_piece_cut_flags_after_clip(
                    source_points=source_points,
                    clipped_points=piece,
                    cut_start=bool(masked_piece.get("cut_start", False)),
                    cut_end=bool(masked_piece.get("cut_end", False)),
                )
                piece = resample_polyline(piece, step_px=resample_step_px)
                if piece.ndim != 2 or piece.shape[0] < 2:
                    continue
                start_side = point_boundary_side(piece[0], rect_global, boundary_tol_px)
                end_side = point_boundary_side(piece[-1], rect_global, boundary_tol_px)
                start_type = "cut" if bool(cut_start) or start_side is not None else "start"
                end_type = "cut" if bool(cut_end) or end_side is not None else "end"
                piece, start_type, end_type = canonicalize_line_direction(piece, start_type=start_type, end_type=end_type)
                out.append(
                    {
                        "category": str(line["category"]),
                        "points_global": piece.astype(np.float32),
                        "start_type": str(start_type),
                        "end_type": str(end_type),
                    }
                )
    return sort_lines(out)


def build_patch_target_lines(patch_segments_global: Sequence[Dict], patch: Dict) -> List[Dict]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    out: List[Dict] = []
    for segment in patch_segments_global:
        local = np.asarray(segment["points_global"], dtype=np.float32) - offset
        points_json = simplify_for_json(local, patch_size=patch_size)
        if len(points_json) < 2:
            continue
        out.append(
            {
                "category": str(segment["category"]),
                "start_type": str(segment["start_type"]),
                "end_type": str(segment["end_type"]),
                "points": points_json,
            }
        )
    return sort_lines(out)


def build_manifest_for_dataset(
    dataset_root: Path,
    splits: Sequence[str],
    image_relpath: str,
    mask_relpath: str,
    lane_relpath: str,
    intersection_relpath: str,
    mask_threshold: int,
    tile_size_px: int,
    overlap_px: int,
    keep_margin_px: int,
    review_crop_pad_px: int,
    tile_min_mask_ratio: float,
    tile_min_mask_pixels: int,
    tile_max_per_sample: int,
    search_within_review_bbox: bool,
    fallback_to_all_if_empty: bool,
    max_samples_per_split: int,
) -> List[Dict]:
    families: List[Dict] = []
    for split in splits:
        split_root = dataset_root / str(split)
        if not split_root.is_dir():
            continue
        sample_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        if int(max_samples_per_split) > 0:
            sample_dirs = sample_dirs[: int(max_samples_per_split)]
        for sample_dir in sample_dirs:
            sample_id = str(sample_dir.name)
            image_path = sample_dir / image_relpath
            mask_path = sample_dir / mask_relpath
            lane_path = sample_dir / lane_relpath
            intersection_path = sample_dir / intersection_relpath
            if not image_path.is_file():
                continue
            _, raster_meta = read_rgb_geotiff(image_path, band_indices=[1, 2, 3])
            review_mask = read_binary_mask(mask_path, threshold=mask_threshold) if mask_path.is_file() else None
            review_bbox = compute_mask_bbox(review_mask) if review_mask is not None else None
            region_bbox = None
            if bool(search_within_review_bbox) and review_bbox is not None:
                region_bbox = expand_bbox(review_bbox, pad_px=int(review_crop_pad_px), width=int(raster_meta.width), height=int(raster_meta.height))
            tile_windows = generate_tile_windows(
                width=int(raster_meta.width),
                height=int(raster_meta.height),
                tile_size_px=int(tile_size_px),
                overlap_px=int(overlap_px),
                region_bbox=region_bbox,
                keep_margin_px=int(keep_margin_px),
            )
            tile_windows = annotate_tile_windows_with_mask(tile_windows=tile_windows, mask=review_mask)
            selected_windows, tile_audits = audit_tile_window_selection(
                tile_windows=tile_windows,
                min_mask_ratio=float(tile_min_mask_ratio),
                min_mask_pixels=int(tile_min_mask_pixels),
                max_tiles=None if int(tile_max_per_sample) <= 0 else int(tile_max_per_sample),
                fallback_to_all_if_empty=bool(fallback_to_all_if_empty),
            )
            selected_windows = sorted(selected_windows, key=lambda item: (int(item.y0), int(item.x0)))
            y_keys = sorted({int(item.y0) for item in selected_windows})
            x_keys = sorted({int(item.x0) for item in selected_windows})
            row_by_y = {int(y): idx for idx, y in enumerate(y_keys)}
            col_by_x = {int(x): idx for idx, x in enumerate(x_keys)}
            patches: List[Dict] = []
            for patch_id, window in enumerate(selected_windows):
                bbox = window.bbox
                keep_bbox = window.keep_bbox
                patches.append(
                    {
                        "patch_id": int(patch_id),
                        "row": int(row_by_y[int(window.y0)]),
                        "col": int(col_by_x[int(window.x0)]),
                        "center_x": int((bbox[0] + bbox[2]) // 2),
                        "center_y": int((bbox[1] + bbox[3]) // 2),
                        "crop_box": {
                            "x_min": int(bbox[0]),
                            "y_min": int(bbox[1]),
                            "x_max": int(bbox[2]),
                            "y_max": int(bbox[3]),
                            "center_x": int((bbox[0] + bbox[2]) // 2),
                            "center_y": int((bbox[1] + bbox[3]) // 2),
                        },
                        "keep_box": {
                            "x_min": int(keep_bbox[0]),
                            "y_min": int(keep_bbox[1]),
                            "x_max": int(keep_bbox[2]),
                            "y_max": int(keep_bbox[3]),
                        },
                        "mask_ratio": float(window.mask_ratio),
                        "mask_pixels": int(window.mask_pixels),
                    }
                )
            families.append(
                {
                    "family_id": sample_id,
                    "split": str(split),
                    "source_sample_id": sample_id,
                    "source_image": image_path.name,
                    "source_image_path": str(image_path),
                    "source_mask_path": str(mask_path) if mask_path.is_file() else "",
                    "source_lane_path": str(lane_path) if lane_path.is_file() else "",
                    "source_intersection_path": str(intersection_path) if intersection_path.is_file() else "",
                    "image_size": [int(raster_meta.width), int(raster_meta.height)],
                    "tiling": {
                        "tile_size_px": int(tile_size_px),
                        "overlap_px": int(overlap_px),
                        "keep_margin_px": int(keep_margin_px),
                        "review_crop_pad_px": int(review_crop_pad_px),
                        "search_within_review_bbox": bool(search_within_review_bbox),
                    },
                    "crop_bbox": None if region_bbox is None else [int(v) for v in region_bbox],
                    "patches": patches,
                    "tile_audits": tile_audits,
                }
            )
    return families


def load_family_raster_and_mask(family: Dict, band_indices: Sequence[int], mask_threshold: int) -> Tuple[np.ndarray, RasterMeta, Optional[np.ndarray]]:
    image_hwc, raster_meta = read_rgb_geotiff(Path(family["source_image_path"]).resolve(), band_indices=band_indices)
    mask_path = str(family.get("source_mask_path", "")).strip()
    review_mask = read_binary_mask(Path(mask_path), threshold=mask_threshold) if mask_path else None
    return image_hwc, raster_meta, review_mask


def family_global_lines(
    family: Dict,
    raster_meta: RasterMeta,
    include_lane: bool = True,
    include_intersection: bool = True,
) -> List[Dict]:
    image_path = Path(family["source_image_path"]).resolve()
    sample_dir = image_path.parents[1]
    lane_path = Path(str(family.get("source_lane_path", "")).strip()) if str(family.get("source_lane_path", "")).strip() else sample_dir / DEFAULT_LANE_RELPATH
    intersection_path = Path(str(family.get("source_intersection_path", "")).strip()) if str(family.get("source_intersection_path", "")).strip() else sample_dir / DEFAULT_INTERSECTION_RELPATH
    lane_rel = str(lane_path.resolve().relative_to(sample_dir))
    intersection_rel = str(intersection_path.resolve().relative_to(sample_dir))
    return load_sample_global_lines(
        sample_dir=sample_dir,
        raster_meta=raster_meta,
        lane_relpath=lane_rel,
        intersection_relpath=intersection_rel,
        include_lane=bool(include_lane),
        include_intersection=bool(include_intersection),
    )


def build_patch_image(raw_image_hwc: np.ndarray, patch: Dict) -> Image.Image:
    crop_box = patch["crop_box"]
    crop = raw_image_hwc[
        int(crop_box["y_min"]) : int(crop_box["y_max"]),
        int(crop_box["x_min"]) : int(crop_box["x_max"]),
    ]
    return Image.fromarray(np.asarray(np.clip(crop, 0.0, 255.0), dtype=np.uint8))


def build_owned_segments_by_patch(
    family: Dict,
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    resample_step_px: float,
    boundary_tol_px: float,
) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
        keep_box = patch["keep_box"]
        rect_global = (
            float(keep_box["x_min"]),
            float(keep_box["y_min"]),
            float(keep_box["x_max"]),
            float(keep_box["y_max"]),
        )
        out[int(patch["patch_id"])] = build_patch_segments_global(
            global_lines=global_lines,
            review_mask=review_mask,
            rect_global=rect_global,
            resample_step_px=float(resample_step_px),
            boundary_tol_px=float(boundary_tol_px),
        )
    return out


def build_full_segments_for_patch(
    patch: Dict,
    global_lines: Sequence[Dict],
    review_mask: Optional[np.ndarray],
    resample_step_px: float,
    boundary_tol_px: float,
) -> List[Dict]:
    crop_box = patch["crop_box"]
    rect_global = (
        float(crop_box["x_min"]),
        float(crop_box["y_min"]),
        float(crop_box["x_max"]),
        float(crop_box["y_max"]),
    )
    return build_patch_segments_global(
        global_lines=global_lines,
        review_mask=review_mask,
        rect_global=rect_global,
        resample_step_px=float(resample_step_px),
        boundary_tol_px=float(boundary_tol_px),
    )


def extract_state_lines(
    patch: Dict,
    family: Dict,
    owned_segments_by_patch: Dict[int, List[Dict]],
    trace_points: int,
    boundary_tol_px: float,
) -> List[Dict]:
    patches = sorted(list(family["patches"]), key=lambda item: int(item["patch_id"]))
    patch_map = {(int(item["row"]), int(item["col"])): item for item in patches}
    row = int(patch["row"])
    col = int(patch["col"])
    crop_box = patch["crop_box"]
    crop_rect_global = (
        float(crop_box["x_min"]),
        float(crop_box["y_min"]),
        float(crop_box["x_max"]),
        float(crop_box["y_max"]),
    )
    patch_size = int(crop_box["x_max"] - crop_box["x_min"])
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    neighbors = []
    if (row, col - 1) in patch_map:
        neighbors.append((patch_map[(row, col - 1)], "left"))
    if (row - 1, col) in patch_map:
        neighbors.append((patch_map[(row - 1, col)], "top"))
    out: List[Dict] = []
    patch_rect_local = (0.0, 0.0, float(patch_size), float(patch_size))
    for neighbor_patch, handoff_side in neighbors:
        for segment in owned_segments_by_patch.get(int(neighbor_patch["patch_id"]), []):
            for piece in clip_polyline_to_rect(np.asarray(segment["points_global"], dtype=np.float32), crop_rect_global):
                local = np.asarray(piece, dtype=np.float32) - offset
                if local.ndim != 2 or local.shape[0] < 2:
                    continue
                boundary_idx = None
                if point_boundary_side(local[0], patch_rect_local, boundary_tol_px) == handoff_side:
                    boundary_idx = 0
                elif point_boundary_side(local[-1], patch_rect_local, boundary_tol_px) == handoff_side:
                    boundary_idx = -1
                if boundary_idx is None:
                    continue
                if boundary_idx == -1:
                    local = local[::-1].copy()
                trace = local[: max(2, int(trace_points))]
                trace_json = simplify_for_json(trace, patch_size=patch_size)
                if len(trace_json) < 2:
                    continue
                out.append(
                    {
                        "source_patch": int(neighbor_patch["patch_id"]),
                        "category": str(segment["category"]),
                        "start_type": "cut",
                        "end_type": "cut",
                        "points": trace_json,
                    }
                )
    seen = set()
    deduped: List[Dict] = []
    for line in sort_lines(out):
        key = (int(line["source_patch"]), str(line["category"]), tuple((int(p[0]), int(p[1])) for p in line["points"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return deduped


def build_patch_only_record(
    image_rel_path: str,
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
    prompt_template: str,
) -> Dict:
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}


def build_state_record(
    image_rel_path: str,
    state_lines: Sequence[Dict],
    target_lines: Sequence[Dict],
    sample_id: str,
    system_prompt: str,
    prompt_template: str,
) -> Dict:
    state_json = json.dumps({"lines": list(state_lines)}, ensure_ascii=False, separators=(",", ":"))
    target_json = json.dumps({"lines": list(target_lines)}, ensure_ascii=False, separators=(",", ":"))
    messages: List[Dict] = []
    if str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append({"role": "user", "content": str(prompt_template).format(state_json=state_json)})
    messages.append({"role": "assistant", "content": target_json})
    return {"id": str(sample_id), "messages": messages, "images": [str(image_rel_path).replace("\\", "/")]}


def parse_generated_json(text: str) -> Tuple[Optional[Dict], str]:
    raw = str(text or "").strip()
    if not raw:
        return None, ""
    start = raw.find("{")
    if start < 0:
        return None, raw
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        return None, raw
    cleaned = raw[start:end]
    try:
        return json.loads(cleaned), cleaned
    except Exception:
        return None, cleaned


def sanitize_pred_lines(pred_lines: Sequence[Dict], patch_size: int) -> List[Dict]:
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        points = simplify_for_json(arr, patch_size=patch_size)
        if len(points) < 2:
            continue
        start_type = str(line.get("start_type", "start"))
        end_type = str(line.get("end_type", "end"))
        if start_type not in {"start", "cut"}:
            start_type = "start"
        if end_type not in {"end", "cut"}:
            end_type = "end"
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": start_type,
                "end_type": end_type,
                "points": points,
            }
        )
    return sort_lines(out)


def local_lines_to_global(pred_lines: Sequence[Dict], patch: Dict) -> List[Dict]:
    crop_box = patch["crop_box"]
    offset = np.asarray([crop_box["x_min"], crop_box["y_min"]], dtype=np.float32)[None, :]
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points_global": dedup_points(arr + offset),
            }
        )
    return out


def apply_state_mode(
    raw_state_lines: Sequence[Dict],
    state_mode: str,
    patch_size: int,
    weak_trace_points: int,
    state_line_dropout: float,
    state_point_jitter_px: float,
    state_truncate_prob: float,
    rng: np.random.Generator,
) -> List[Dict]:
    return build_state_lines_by_mode(
        raw_state_lines=raw_state_lines,
        state_mode=state_mode,
        patch_size=int(patch_size),
        weak_trace_points=int(weak_trace_points),
        state_line_dropout=float(state_line_dropout),
        state_point_jitter_px=float(state_point_jitter_px),
        state_truncate_prob=float(state_truncate_prob),
        rng=rng,
    )
