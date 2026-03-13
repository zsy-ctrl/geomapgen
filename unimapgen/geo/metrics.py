from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from .io import RasterMeta, pixel_to_world
from .schema import TaskSchema


def filter_features_by_review_mask(
    feature_records: Sequence[Dict],
    mask: np.ndarray,
    min_inside_ratio: float = 0.5,
) -> List[Dict]:
    if mask is None:
        return [dict(x) for x in feature_records]
    out = []
    height, width = mask.shape[:2]
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
        cols = np.clip(np.round(points[:, 0]).astype(np.int64), 0, width - 1)
        rows = np.clip(np.round(points[:, 1]).astype(np.int64), 0, height - 1)
        ratio = float(mask[rows, cols].mean())
        if ratio >= float(min_inside_ratio):
            record = {
                "properties": dict(feature.get("properties", {})),
                "points": np.asarray(feature.get("points", []), dtype=np.float32),
            }
            if rings:
                record["rings"] = [ring.astype(np.float32) for ring in rings]
            out.append(record)
    return out


def feature_records_to_world_geometries(
    task_schema: TaskSchema,
    feature_records: Sequence[Dict],
    raster_meta: RasterMeta,
) -> List[Dict]:
    out = []
    for source_index, feature in enumerate(feature_records):
        try:
            if task_schema.geometry_type == "polygon":
                rings_px = [
                    np.asarray(ring, dtype=np.float32)
                    for ring in feature.get("rings", [])
                    if np.asarray(ring, dtype=np.float32).ndim == 2
                ] if feature.get("rings") else []
                if rings_px:
                    rings_world = [pixel_to_world(ring, raster_meta=raster_meta) for ring in rings_px]
                    rings_world = [ring for ring in rings_world if ring.shape[0] >= task_schema.min_points_per_feature]
                    if not rings_world:
                        continue
                    exterior = rings_world[0].tolist()
                    if exterior[0] != exterior[-1]:
                        exterior.append(list(exterior[0]))
                    holes = []
                    for hole_ring in rings_world[1:]:
                        hole = hole_ring.tolist()
                        if hole[0] != hole[-1]:
                            hole.append(list(hole[0]))
                        holes.append(hole)
                    geom = Polygon(exterior, holes=holes)
                    points_world = rings_world[0]
                else:
                    points_px = np.asarray(feature.get("points", []), dtype=np.float32)
                    if points_px.ndim != 2 or points_px.shape[0] < task_schema.min_points_per_feature:
                        continue
                    points_world = pixel_to_world(points_px, raster_meta=raster_meta)
                    coords = points_world.tolist()
                    if coords[0] != coords[-1]:
                        coords.append(list(coords[0]))
                    geom = Polygon(coords)
            else:
                points_px = np.asarray(feature.get("points", []), dtype=np.float32)
                if points_px.ndim != 2 or points_px.shape[0] < task_schema.min_points_per_feature:
                    continue
                points_world = pixel_to_world(points_px, raster_meta=raster_meta)
                geom = LineString(points_world.tolist())
        except Exception:
            continue
        if geom.is_empty:
            continue
        out.append(
            {
                "geometry": geom,
                "properties": dict(feature.get("properties", {})),
                "points_world": points_world,
                "source_index": int(source_index),
            }
        )
    return out


def evaluate_lane_predictions(
    gt_features: Sequence[Dict],
    pred_features: Sequence[Dict],
    raster_meta: RasterMeta,
    task_schema: TaskSchema,
    distance_threshold_m: float = 2.0,
) -> Dict[str, float]:
    gt_geoms = feature_records_to_world_geometries(task_schema=task_schema, feature_records=gt_features, raster_meta=raster_meta)
    pred_geoms = feature_records_to_world_geometries(task_schema=task_schema, feature_records=pred_features, raster_meta=raster_meta)
    matches = _greedy_match_lines(pred_geoms=pred_geoms, gt_geoms=gt_geoms, threshold=float(distance_threshold_m))
    matched_pred = {m["pred_idx"] for m in matches}
    matched_gt = {m["gt_idx"] for m in matches}
    precision = float(len(matches)) / float(max(1, len(pred_geoms)))
    recall = float(len(matches)) / float(max(1, len(gt_geoms)))
    f1 = 0.0 if precision + recall <= 1e-9 else 2.0 * precision * recall / (precision + recall)
    mean_hausdorff = float(np.mean([m["distance"] for m in matches])) if matches else float("inf")
    mean_endpoint_error = float(np.mean([m["endpoint_error"] for m in matches])) if matches else float("inf")
    property_acc = _property_accuracy(matches=matches, pred_geoms=pred_geoms, gt_geoms=gt_geoms)
    return {
        "lane_precision_2m": precision,
        "lane_recall_2m": recall,
        "lane_f1_2m": f1,
        "lane_mean_hausdorff_m": mean_hausdorff,
        "lane_mean_endpoint_error_m": mean_endpoint_error,
        "lane_property_exact_acc": property_acc,
        "lane_pred_count": float(len(pred_geoms)),
        "lane_gt_count": float(len(gt_geoms)),
        "lane_unmatched_pred": float(len(pred_geoms) - len(matched_pred)),
        "lane_unmatched_gt": float(len(gt_geoms) - len(matched_gt)),
    }


def evaluate_intersection_predictions(
    gt_features: Sequence[Dict],
    pred_features: Sequence[Dict],
    raster_meta: RasterMeta,
    task_schema: TaskSchema,
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    gt_geoms = feature_records_to_world_geometries(task_schema=task_schema, feature_records=gt_features, raster_meta=raster_meta)
    pred_geoms = feature_records_to_world_geometries(task_schema=task_schema, feature_records=pred_features, raster_meta=raster_meta)
    matches = _greedy_match_polygons(pred_geoms=pred_geoms, gt_geoms=gt_geoms, threshold=float(iou_threshold))
    precision = float(len(matches)) / float(max(1, len(pred_geoms)))
    recall = float(len(matches)) / float(max(1, len(gt_geoms)))
    f1 = 0.0 if precision + recall <= 1e-9 else 2.0 * precision * recall / (precision + recall)
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0
    property_acc = _property_accuracy(matches=matches, pred_geoms=pred_geoms, gt_geoms=gt_geoms)
    return {
        "intersection_precision_iou30": precision,
        "intersection_recall_iou30": recall,
        "intersection_f1_iou30": f1,
        "intersection_mean_iou": mean_iou,
        "intersection_property_exact_acc": property_acc,
        "intersection_pred_count": float(len(pred_geoms)),
        "intersection_gt_count": float(len(gt_geoms)),
    }


def _greedy_match_lines(pred_geoms: Sequence[Dict], gt_geoms: Sequence[Dict], threshold: float) -> List[Dict]:
    candidates = []
    for pred_idx, pred in enumerate(pred_geoms):
        for gt_idx, gt in enumerate(gt_geoms):
            distance = float(pred["geometry"].hausdorff_distance(gt["geometry"]))
            endpoint_error = _endpoint_error(pred["points_world"], gt["points_world"])
            candidates.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": gt_idx,
                    "distance": distance,
                    "endpoint_error": endpoint_error,
                }
            )
    candidates.sort(key=lambda x: (x["distance"], x["endpoint_error"]))
    matches = []
    used_pred = set()
    used_gt = set()
    for cand in candidates:
        if cand["distance"] > float(threshold):
            continue
        if cand["pred_idx"] in used_pred or cand["gt_idx"] in used_gt:
            continue
        used_pred.add(cand["pred_idx"])
        used_gt.add(cand["gt_idx"])
        matches.append(cand)
    return matches


def _greedy_match_polygons(pred_geoms: Sequence[Dict], gt_geoms: Sequence[Dict], threshold: float) -> List[Dict]:
    candidates = []
    for pred_idx, pred in enumerate(pred_geoms):
        for gt_idx, gt in enumerate(gt_geoms):
            inter = float(pred["geometry"].intersection(gt["geometry"]).area)
            union = float(pred["geometry"].union(gt["geometry"]).area)
            iou = 0.0 if union <= 1e-9 else inter / union
            candidates.append({"pred_idx": pred_idx, "gt_idx": gt_idx, "iou": iou})
    candidates.sort(key=lambda x: x["iou"], reverse=True)
    matches = []
    used_pred = set()
    used_gt = set()
    for cand in candidates:
        if cand["iou"] < float(threshold):
            continue
        if cand["pred_idx"] in used_pred or cand["gt_idx"] in used_gt:
            continue
        used_pred.add(cand["pred_idx"])
        used_gt.add(cand["gt_idx"])
        matches.append(cand)
    return matches


def _endpoint_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("inf")
    forward = float(np.linalg.norm(pred_pts[0] - gt_pts[0]) + np.linalg.norm(pred_pts[-1] - gt_pts[-1])) / 2.0
    reverse = float(np.linalg.norm(pred_pts[0] - gt_pts[-1]) + np.linalg.norm(pred_pts[-1] - gt_pts[0])) / 2.0
    return min(forward, reverse)


def _property_accuracy(matches: Sequence[Dict], pred_geoms: Sequence[Dict], gt_geoms: Sequence[Dict]) -> float:
    if not matches:
        return 0.0
    correct = 0
    for match in matches:
        pred_props = pred_geoms[int(match["pred_idx"])]["properties"]
        gt_props = gt_geoms[int(match["gt_idx"])]["properties"]
        if _properties_equal(pred_props, gt_props):
            correct += 1
    return float(correct) / float(max(1, len(matches)))


def _properties_equal(pred_props: Dict, gt_props: Dict) -> bool:
    if pred_props.keys() != gt_props.keys():
        return False
    for key in pred_props.keys():
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


def deduplicate_feature_records(
    task_schema: TaskSchema,
    feature_records: Sequence[Dict],
    raster_meta: RasterMeta,
    line_distance_threshold_m: float = 1.0,
    polygon_iou_threshold: float = 0.5,
) -> List[Dict]:
    geometries = feature_records_to_world_geometries(
        task_schema=task_schema,
        feature_records=feature_records,
        raster_meta=raster_meta,
    )
    if len(geometries) <= 1:
        return [dict(x) for x in feature_records]

    def _priority(record: Dict) -> Tuple[float, int]:
        geom = record["geometry"]
        size = float(geom.area) if task_schema.geometry_type == "polygon" else float(geom.length)
        return size, int(record["source_index"])

    ordered = sorted(geometries, key=_priority, reverse=True)
    kept = []
    kept_indices = []
    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if not _properties_equal(candidate["properties"], existing["properties"]):
                continue
            if task_schema.geometry_type == "polygon":
                inter = float(candidate["geometry"].intersection(existing["geometry"]).area)
                union = float(candidate["geometry"].union(existing["geometry"]).area)
                iou = 0.0 if union <= 1e-9 else inter / union
                if iou >= float(polygon_iou_threshold):
                    duplicate = True
                    break
            else:
                distance = float(candidate["geometry"].hausdorff_distance(existing["geometry"]))
                if distance <= float(line_distance_threshold_m):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(candidate)
            kept_indices.append(int(candidate["source_index"]))
    kept_indices = sorted(set(kept_indices))
    return [dict(feature_records[idx]) for idx in kept_indices if 0 <= idx < len(feature_records)]
