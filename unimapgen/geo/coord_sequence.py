from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import ResizeContext
from .schema import TaskSchema


_SIDE_ORDER = {"left": 0, "top": 1, "none": 2, "right": 3, "bottom": 4}


def points_abs_to_uv(points_xy: np.ndarray, resize_ctx: ResizeContext) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    u = (pts[:, 0] - float(resize_ctx.crop_x0)) * float(resize_ctx.scale_x) + float(resize_ctx.pad_x)
    v = (pts[:, 1] - float(resize_ctx.crop_y0)) * float(resize_ctx.scale_y) + float(resize_ctx.pad_y)
    return np.stack([u, v], axis=-1).astype(np.float32)


def points_uv_to_abs(points_uv: np.ndarray, resize_ctx: ResizeContext) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    scale_x = max(float(resize_ctx.scale_x), 1e-6)
    scale_y = max(float(resize_ctx.scale_y), 1e-6)
    x = (pts[:, 0] - float(resize_ctx.pad_x)) / scale_x + float(resize_ctx.crop_x0)
    y = (pts[:, 1] - float(resize_ctx.pad_y)) / scale_y + float(resize_ctx.crop_y0)
    return np.stack([x, y], axis=-1).astype(np.float32)


def compact_props_json(props: Dict) -> str:
    return json.dumps(dict(props or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def feature_record_sort_key(feature: Dict, geometry_type: str) -> Tuple:
    geom = str(geometry_type or "").strip().lower()
    if geom == "polygon" and feature.get("rings"):
        pts = canonicalize_polygon_rings(feature.get("rings", []))
        points = pts[0] if pts else np.zeros((0, 2), dtype=np.float32)
    else:
        points = _canonicalize_line_lex(np.asarray(feature.get("points", []), dtype=np.float32))
    if points.ndim != 2 or points.shape[0] == 0:
        return (10**9, 10**9, 10**9, 10**9, 0, compact_props_json(feature.get("properties", {})))
    first = points[0]
    last = points[-1]
    props_text = compact_props_json(feature.get("properties", {}))
    return (
        float(first[1]),
        float(first[0]),
        float(last[1]),
        float(last[0]),
        int(points.shape[0]),
        props_text,
    )


def parse_props_json(text: str) -> Dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    candidates = [raw]
    normalized = raw.replace("，", ",").replace("：", ":")
    normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
    if normalized not in candidates:
        candidates.append(normalized)
    try:
        for candidate in candidates:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    py_like = normalized
    py_like = re.sub(r"\btrue\b", "True", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
    py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
    try:
        obj = ast.literal_eval(py_like)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"_raw_text": raw}


def relaxed_parse_props_json(text: str) -> Dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    normalized = raw
    for src, dst in (
        ("，", ","),
        ("：", ":"),
        ("；", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
    candidates = [normalized]
    body = normalized.strip().strip(",")
    if body and ":" in body and not body.startswith("{"):
        candidates.append("{" + body.strip("{}") + "}")
    for candidate in candidates:
        parsed = parse_props_json(candidate)
        if "_raw_text" not in parsed:
            return parsed
    return {"_raw_text": raw}


def boundary_side_for_point_uv(point_xy: Sequence[float], image_size: int, tol_px: float = 1.5) -> str:
    x = float(point_xy[0])
    y = float(point_xy[1])
    size = max(1, int(image_size))
    distances = {
        "left": abs(x - 0.0),
        "top": abs(y - 0.0),
        "right": abs(x - float(size - 1)),
        "bottom": abs(y - float(size - 1)),
    }
    side = min(distances.items(), key=lambda kv: kv[1])[0]
    return side if distances[side] <= float(tol_px) else "none"


def collect_boundary_cut_points_uv(
    points_uv: np.ndarray,
    image_size: int,
    tol_px: float = 1.5,
) -> Tuple[np.ndarray, List[str]]:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), []
    cut_points: List[np.ndarray] = []
    cut_sides: List[str] = []
    for point in pts:
        side = boundary_side_for_point_uv(point_xy=point, image_size=image_size, tol_px=tol_px)
        if side == "none":
            continue
        if any(np.allclose(point, existing, atol=1e-3) for existing in cut_points):
            if side not in cut_sides:
                cut_sides.append(side)
            continue
        cut_points.append(np.asarray(point, dtype=np.float32))
        if side not in cut_sides:
            cut_sides.append(side)
    if not cut_points:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.stack(cut_points, axis=0).astype(np.float32), list(cut_sides)


def line_cut_metadata_from_uv(
    points_uv: np.ndarray,
    image_size: int,
    cut_start: bool = False,
    cut_end: bool = False,
    tol_px: float = 1.5,
) -> Dict:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {
            "cut_in": "none",
            "cut_out": "none",
            "cut_points_uv": np.zeros((0, 2), dtype=np.float32),
            "cut_sides": [],
        }
    cut_points: List[np.ndarray] = []
    cut_sides: List[str] = []
    cut_in = "none"
    cut_out = "none"
    start_side = boundary_side_for_point_uv(points_uv[0], image_size=image_size, tol_px=tol_px)
    end_side = boundary_side_for_point_uv(points_uv[-1], image_size=image_size, tol_px=tol_px)
    if bool(cut_start):
        cut_in = start_side
        if cut_in == "none":
            cut_in = "internal"
        cut_points.append(np.asarray(points_uv[0], dtype=np.float32))
        if cut_in not in {"none", "internal"} and cut_in not in cut_sides:
            cut_sides.append(cut_in)
    if bool(cut_end):
        cut_out = end_side
        if cut_out == "none":
            cut_out = "internal"
        if not any(np.allclose(points_uv[-1], existing, atol=1e-3) for existing in cut_points):
            cut_points.append(np.asarray(points_uv[-1], dtype=np.float32))
        if cut_out not in {"none", "internal"} and cut_out not in cut_sides:
            cut_sides.append(cut_out)
    if cut_in == "none" and start_side != "none":
        cut_in = start_side
        if not any(np.allclose(points_uv[0], existing, atol=1e-3) for existing in cut_points):
            cut_points.append(np.asarray(points_uv[0], dtype=np.float32))
        if start_side not in cut_sides:
            cut_sides.append(start_side)
    if cut_out == "none" and end_side != "none":
        cut_out = end_side
        if not any(np.allclose(points_uv[-1], existing, atol=1e-3) for existing in cut_points):
            cut_points.append(np.asarray(points_uv[-1], dtype=np.float32))
        if end_side not in cut_sides:
            cut_sides.append(end_side)
    if not cut_points:
        boundary_points, boundary_sides = collect_boundary_cut_points_uv(
            points_uv=pts,
            image_size=image_size,
            tol_px=tol_px,
        )
        cut_points = [point.astype(np.float32) for point in boundary_points]
        cut_sides = list(boundary_sides)
    cut_points_uv = np.stack(cut_points, axis=0).astype(np.float32) if cut_points else np.zeros((0, 2), dtype=np.float32)
    return {
        "cut_in": str(cut_in),
        "cut_out": str(cut_out),
        "cut_points_uv": cut_points_uv,
        "cut_sides": list(cut_sides),
    }


def polygon_cut_metadata_from_uv(
    rings_uv: Sequence[np.ndarray],
    image_size: int,
    clipped: bool = False,
    tol_px: float = 1.5,
) -> Dict:
    cut_points: List[np.ndarray] = []
    cut_sides: List[str] = []
    for ring in rings_uv:
        ring_points, ring_sides = collect_boundary_cut_points_uv(
            points_uv=np.asarray(ring, dtype=np.float32),
            image_size=image_size,
            tol_px=tol_px,
        )
        for point in ring_points:
            if not any(np.allclose(point, existing, atol=1e-3) for existing in cut_points):
                cut_points.append(np.asarray(point, dtype=np.float32))
        for side in ring_sides:
            if side not in cut_sides:
                cut_sides.append(side)
    if not cut_points and bool(clipped):
        return {
            "cut_in": "internal",
            "cut_out": "internal",
            "cut_points_uv": np.zeros((0, 2), dtype=np.float32),
            "cut_sides": [],
        }
    cut_points_uv = np.stack(cut_points, axis=0).astype(np.float32) if cut_points else np.zeros((0, 2), dtype=np.float32)
    return {
        "cut_in": "internal" if (bool(clipped) and cut_points_uv.shape[0] > 0) else "none",
        "cut_out": "internal" if (bool(clipped) and cut_points_uv.shape[0] > 0) else "none",
        "cut_points_uv": cut_points_uv,
        "cut_sides": list(cut_sides),
    }


def detect_feature_boundary_sides(points_uv: np.ndarray, image_size: int, tol_px: float = 1.5) -> List[str]:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return []
    sides = []
    for point in pts:
        side = boundary_side_for_point_uv(point_xy=point, image_size=image_size, tol_px=tol_px)
        if side != "none" and side not in sides:
            sides.append(side)
    return sides


def canonicalize_feature_points(
    points_uv: np.ndarray,
    geometry_type: str,
    image_size: int,
    tol_px: float = 1.5,
) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    geom = str(geometry_type).strip().lower()
    if geom == "polygon":
        return _rotate_polygon_to_min(pts)
    return _canonicalize_line(pts, image_size=image_size, tol_px=tol_px)


def canonicalize_polygon_rings(
    rings_uv: Sequence[Sequence[Sequence[float]]],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for ring in rings_uv:
        pts = np.asarray(ring, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        out.append(_rotate_polygon_to_min(pts))
    return out


def _canonicalize_line(points_uv: np.ndarray, image_size: int, tol_px: float) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts
    start_side = boundary_side_for_point_uv(points_uv[0], image_size=image_size, tol_px=tol_px)
    end_side = boundary_side_for_point_uv(points_uv[-1], image_size=image_size, tol_px=tol_px)
    start_key = (_SIDE_ORDER.get(start_side, 99), float(points_uv[0, 0]), float(points_uv[0, 1]))
    end_key = (_SIDE_ORDER.get(end_side, 99), float(points_uv[-1, 0]), float(points_uv[-1, 1]))
    if end_key < start_key:
        return points_uv[::-1].copy()
    return points_uv.copy()


def _canonicalize_line_lex(points_uv: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts.copy()
    start_key = (float(pts[0, 1]), float(pts[0, 0]), float(pts[-1, 1]), float(pts[-1, 0]))
    end_key = (float(pts[-1, 1]), float(pts[-1, 0]), float(pts[0, 1]), float(pts[0, 0]))
    if end_key < start_key:
        return pts[::-1].copy()
    return pts.copy()


def _rotate_polygon_to_min(points_uv: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.shape[0] <= 1:
        return pts.copy()
    keys = [(float(pt[0]), float(pt[1]), idx) for idx, pt in enumerate(pts)]
    min_idx = min(keys)[2]
    return np.concatenate([pts[min_idx:], pts[:min_idx]], axis=0).astype(np.float32)


def sample_anchor_points(points_uv: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float32)
    max_points = max(2, int(max_points))
    if pts.ndim != 2 or pts.shape[0] <= max_points:
        return pts.astype(np.float32)
    indices = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
    return pts[indices].astype(np.float32)


def rings_abs_to_uv(rings_xy: Sequence[np.ndarray], resize_ctx: ResizeContext) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for ring in rings_xy:
        uv = points_abs_to_uv(points_xy=np.asarray(ring, dtype=np.float32), resize_ctx=resize_ctx)
        if uv.ndim == 2 and uv.shape[0] > 0:
            out.append(uv.astype(np.float32))
    return out


def rings_uv_to_abs(rings_uv: Sequence[np.ndarray], resize_ctx: ResizeContext) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for ring in rings_uv:
        pts = points_uv_to_abs(points_uv=np.asarray(ring, dtype=np.float32), resize_ctx=resize_ctx)
        if pts.ndim == 2 and pts.shape[0] > 0:
            out.append(pts.astype(np.float32))
    return out


def uv_feature_records_to_target_items(
    feature_records: Sequence[Dict],
    task_schema: TaskSchema,
    image_size: int,
    boundary_tol_px: float = 1.5,
) -> List[Dict]:
    out: List[Dict] = []
    for feature in feature_records:
        rings_uv = None
        if task_schema.geometry_type == "polygon" and feature.get("rings"):
            rings_uv = canonicalize_polygon_rings(feature.get("rings", []))
            points_uv = rings_uv[0] if rings_uv else np.zeros((0, 2), dtype=np.float32)
        else:
            points_uv = canonicalize_feature_points(
                points_uv=np.asarray(feature.get("points", []), dtype=np.float32),
                geometry_type=task_schema.geometry_type,
                image_size=image_size,
                tol_px=boundary_tol_px,
            )
        if points_uv.shape[0] < int(task_schema.min_points_per_feature):
            continue
        if task_schema.geometry_type == "linestring":
            cut_meta = line_cut_metadata_from_uv(
                points_uv=points_uv,
                image_size=image_size,
                cut_start=bool(feature.get("cut_start", False)),
                cut_end=bool(feature.get("cut_end", False)),
                tol_px=boundary_tol_px,
            )
        else:
            cut_meta = polygon_cut_metadata_from_uv(
                rings_uv=rings_uv or [points_uv],
                image_size=image_size,
                clipped=bool(feature.get("clipped", False)),
                tol_px=boundary_tol_px,
            )
        cut_in = str(cut_meta.get("cut_in", "none"))
        cut_out = str(cut_meta.get("cut_out", "none"))
        out.append(
            {
                "geometry_type": task_schema.geometry_type,
                "props_json": compact_props_json(feature.get("properties", {})),
                "points_uv": points_uv.astype(np.float32),
                "rings_uv": [ring.astype(np.float32) for ring in rings_uv] if rings_uv else None,
                "cut_in": str(cut_in),
                "cut_out": str(cut_out),
                "cut_points_uv": np.asarray(cut_meta.get("cut_points_uv", []), dtype=np.float32),
                "cut_sides": [str(side) for side in cut_meta.get("cut_sides", [])],
                "source": "state" if cut_in in {"left", "top"} or any(side in {"left", "top"} for side in cut_meta.get("cut_sides", [])) else "local",
            }
        )
    out.sort(key=lambda item: _item_sort_key(item=item, geometry_type=task_schema.geometry_type))
    return out


def uv_feature_records_to_state_items(
    feature_records: Sequence[Dict],
    task_schema: TaskSchema,
    image_size: int,
    anchor_max_points: int,
    boundary_tol_px: float = 1.5,
) -> List[Dict]:
    out: List[Dict] = []
    for feature in feature_records:
        if task_schema.geometry_type == "polygon" and feature.get("rings"):
            rings_uv = canonicalize_polygon_rings(feature.get("rings", []))
            points_uv = rings_uv[0] if rings_uv else np.zeros((0, 2), dtype=np.float32)
        else:
            points_uv = canonicalize_feature_points(
                points_uv=np.asarray(feature.get("points", []), dtype=np.float32),
                geometry_type=task_schema.geometry_type,
                image_size=image_size,
                tol_px=boundary_tol_px,
            )
        if points_uv.shape[0] < int(task_schema.min_points_per_feature):
            continue
        if task_schema.geometry_type == "polygon" and rings_uv:
            cut_meta = polygon_cut_metadata_from_uv(
                rings_uv=rings_uv,
                image_size=image_size,
                clipped=bool(feature.get("clipped", False)),
                tol_px=boundary_tol_px,
            )
            cut_points_uv = np.asarray(cut_meta.get("cut_points_uv", []), dtype=np.float32)
            cut_sides = [str(side) for side in cut_meta.get("cut_sides", [])]
            if cut_points_uv.ndim == 2 and cut_points_uv.shape[0] > 0:
                for side in cut_sides:
                    if side not in {"left", "top"}:
                        continue
                    side_points = [
                        point.astype(np.float32)
                        for point in cut_points_uv
                        if boundary_side_for_point_uv(point_xy=point, image_size=image_size, tol_px=boundary_tol_px) == side
                    ]
                    if not side_points:
                        continue
                    anchors = np.stack(side_points, axis=0).astype(np.float32)
                    out.append(
                        {
                            "geometry_type": task_schema.geometry_type,
                            "side": side,
                            "points_uv": sample_anchor_points(points_uv=anchors, max_points=anchor_max_points),
                        }
                    )
                if cut_points_uv.shape[0] > 0:
                    continue
        sides = detect_feature_boundary_sides(points_uv=points_uv, image_size=image_size, tol_px=boundary_tol_px)
        for side in sides:
            if side not in {"left", "top"}:
                continue
            out.append(
                {
                    "geometry_type": task_schema.geometry_type,
                    "side": side,
                    "points_uv": sample_anchor_points(points_uv=points_uv, max_points=anchor_max_points),
                }
            )
    out.sort(key=lambda item: _state_item_sort_key(item=item))
    return out


def uv_items_to_abs_feature_records(
    items: Sequence[Dict],
    task_schema: TaskSchema,
    resize_ctx: ResizeContext,
) -> List[Dict]:
    out: List[Dict] = []
    for item in items:
        props = relaxed_parse_props_json(item.get("props_json", ""))
        rings_uv = item.get("rings_uv")
        if task_schema.geometry_type == "polygon" and rings_uv:
            rings_abs = rings_uv_to_abs(rings_uv=rings_uv, resize_ctx=resize_ctx)
            if not rings_abs or rings_abs[0].shape[0] < int(task_schema.min_points_per_feature):
                continue
            points_abs = rings_abs[0]
        else:
            points_uv = np.asarray(item.get("points_uv", []), dtype=np.float32)
            if points_uv.ndim != 2 or points_uv.shape[0] < int(task_schema.min_points_per_feature):
                continue
            points_abs = points_uv_to_abs(points_uv=points_uv, resize_ctx=resize_ctx)
            rings_abs = None
        out.append(
            {
                "properties": props,
                "points": points_abs.astype(np.float32),
                "rings": [ring.astype(np.float32) for ring in rings_abs] if rings_abs else None,
                "cut_in": str(item.get("cut_in", "none")),
                "cut_out": str(item.get("cut_out", "none")),
                "cut_sides": [str(side) for side in item.get("cut_sides", [])],
                "cut_points": points_uv_to_abs(
                    points_uv=np.asarray(item.get("cut_points_uv", []), dtype=np.float32),
                    resize_ctx=resize_ctx,
                ).astype(np.float32)
                if np.asarray(item.get("cut_points_uv", []), dtype=np.float32).ndim == 2
                else np.zeros((0, 2), dtype=np.float32),
                "source": str(item.get("source", "local")),
            }
        )
    return out


def _item_sort_key(item: Dict, geometry_type: str) -> Tuple:
    geom = str(geometry_type or "").strip().lower()
    if geom == "polygon" and item.get("rings_uv"):
        rings = canonicalize_polygon_rings(item.get("rings_uv", []))
        points = rings[0] if rings else np.zeros((0, 2), dtype=np.float32)
    else:
        points = _canonicalize_line_lex(np.asarray(item.get("points_uv", []), dtype=np.float32))
    if points.ndim != 2 or points.shape[0] == 0:
        return (10**9, 10**9, 10**9, 10**9, 0, str(item.get("props_json", "")))
    first = points[0]
    last = points[-1]
    return (
        float(first[1]),
        float(first[0]),
        float(last[1]),
        float(last[0]),
        int(points.shape[0]),
        str(item.get("props_json", "")),
    )


def _state_item_sort_key(item: Dict) -> Tuple:
    points = _canonicalize_line_lex(np.asarray(item.get("points_uv", []), dtype=np.float32))
    if points.ndim != 2 or points.shape[0] == 0:
        return (_SIDE_ORDER.get(str(item.get("side", "none")), 99), 10**9, 10**9, 0)
    first = points[0]
    return (
        _SIDE_ORDER.get(str(item.get("side", "none")), 99),
        float(first[1]),
        float(first[0]),
        int(points.shape[0]),
    )
