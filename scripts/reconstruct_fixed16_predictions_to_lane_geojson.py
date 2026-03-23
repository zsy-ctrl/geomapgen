import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from rasterio import open as rasterio_open

from export_llamafactory_patch_only_from_raw_family_manifest import clip_polyline_to_rect, dedup_points, point_boundary_side
from geo_current_dataset_v1_common import load_jsonl, parse_generated_json, sanitize_pred_lines_uv


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pixel_to_world(points_xy: Iterable[Iterable[float]], transform) -> List[List[float]]:
    out: List[List[float]] = []
    for point in points_xy:
        x = float(point[0])
        y = float(point[1])
        world_x, world_y = transform * (x, y)
        out.append([float(world_x), float(world_y)])
    return out


def build_feature_collection(features: List[Dict], crs_name: str, name: str) -> Dict:
    return {
        "type": "FeatureCollection",
        "name": str(name),
        "crs": {
            "type": "name",
            "properties": {"name": str(crs_name)},
        },
        "features": features,
    }


def build_pixel_feature_collection(features: List[Dict], name: str) -> Dict:
    return {
        "type": "FeatureCollection",
        "name": str(name),
        "properties": {"coord_system": "pixel_global"},
        "features": features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch fixed16 prediction outputs back into full Lane.geojson files."
    )
    parser.add_argument("--fixed16-root", type=str, required=True, help="fixed16_stage_a or fixed16_stage_b root.")
    parser.add_argument("--predictions-path", type=str, required=True, help="Prediction json/jsonl file aligned with the fixed16 split.")
    parser.add_argument("--output-root", type=str, required=True, help="Where to write stitched GeoJSON outputs.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "auto"])
    parser.add_argument("--family-manifest", type=str, default="", help="Optional family_manifest.jsonl path. Defaults to fixed16_root/../family_manifest.jsonl")
    parser.add_argument("--source-root", type=str, default="", help="Optional original split root. If provided, source image path is inferred as source_root/source_sample_id/source_image_relpath.")
    parser.add_argument("--source-image-relpath", type=str, default="patch_tif/0.tif", help="Relative path to the original raster under each source sample directory.")
    parser.add_argument("--merge-endpoint-tol-px", type=float, default=2.0, help="Tolerance for merging cut-to-cut box fragments.")
    parser.add_argument("--source-sample-id", type=str, default="", help="Optional source sample filter.")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("predictions", "rows", "data", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [obj]
    return []


def discover_available_splits(fixed16_root: Path) -> List[str]:
    out: List[str] = []
    for path in sorted(fixed16_root.glob("meta_*.jsonl")):
        name = path.name
        if not name.startswith("meta_") or not name.endswith(".jsonl"):
            continue
        split = name[len("meta_") : -len(".jsonl")].strip()
        if split and split not in out:
            out.append(split)
    return out


def load_meta_rows_for_split(fixed16_root: Path, split: str) -> Optional[List[Dict]]:
    meta_path = fixed16_root / f"meta_{split}.jsonl"
    if not meta_path.is_file():
        return None
    return load_jsonl(meta_path)


def ordered_split_candidates(fixed16_root: Path, requested_split: str) -> List[str]:
    available = discover_available_splits(fixed16_root)
    if not available:
        return []
    if str(requested_split).strip().lower() == "auto":
        return available
    ordered: List[str] = []
    wanted = str(requested_split).strip()
    if wanted in available:
        ordered.append(wanted)
    for split in available:
        if split not in ordered:
            ordered.append(split)
    return ordered


def _normalize_match_text(value: str) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def _normalize_image_key(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text


def _split_agnostic_image_key(value: str) -> str:
    text = _normalize_image_key(value)
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 3 and parts[0].lower() == "images":
        return "/".join(parts[2:])
    return text


def _extract_first_image_path(row: Dict) -> str:
    images = row.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, dict):
                text = _normalize_image_key(value.get("path", "") or value.get("image", "") or value.get("url", ""))
                if text:
                    return text
            else:
                text = _normalize_image_key(value)
                if text:
                    return text
    image = row.get("image")
    if isinstance(image, dict):
        text = _normalize_image_key(image.get("path", "") or image.get("image", "") or image.get("url", ""))
        if text:
            return text
    if isinstance(image, str):
        return _normalize_image_key(image)
    return ""


def _extract_user_prompt_text(row: Dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", msg.get("from", ""))).strip().lower()
            if role not in {"user", "human"}:
                continue
            content = msg.get("content", msg.get("value", ""))
            if isinstance(content, str) and content.strip():
                return _normalize_match_text(content)
    for key in ("prompt", "query", "instruction", "input", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_match_text(value)
    return ""


def build_meta_signature(meta: Dict) -> str:
    image_key = _normalize_image_key(str(meta.get("image", "")))
    prompt_key = _normalize_match_text(str(meta.get("prompt_text", "")))
    if image_key and prompt_key:
        return f"{image_key}|||{prompt_key}"
    return ""


def build_meta_signature_variants(meta: Dict) -> List[str]:
    prompt_key = _normalize_match_text(str(meta.get("prompt_text", "")))
    if not prompt_key:
        return []
    image_value = str(meta.get("image", ""))
    variants: List[str] = []
    exact = _normalize_image_key(image_value)
    splitless = _split_agnostic_image_key(image_value)
    for image_key in (exact, splitless):
        if image_key:
            signature = f"{image_key}|||{prompt_key}"
            if signature not in variants:
                variants.append(signature)
    return variants


def build_prediction_signature(row: Dict) -> str:
    image_key = _extract_first_image_path(row)
    prompt_key = _extract_user_prompt_text(row)
    if image_key and prompt_key:
        return f"{image_key}|||{prompt_key}"
    return ""


def build_prediction_signature_variants(row: Dict) -> List[str]:
    image_value = _extract_first_image_path(row)
    prompt_key = _extract_user_prompt_text(row)
    if not prompt_key:
        return []
    variants: List[str] = []
    exact = _normalize_image_key(image_value)
    splitless = _split_agnostic_image_key(image_value)
    for image_key in (exact, splitless):
        if image_key:
            signature = f"{image_key}|||{prompt_key}"
            if signature not in variants:
                variants.append(signature)
    return variants


def _parse_prediction_text(text: str) -> List[Dict]:
    pred_obj, _ = parse_generated_json(str(text or ""))
    if isinstance(pred_obj, dict):
        return sanitize_pred_lines_uv(list(pred_obj.get("lines", [])))
    return []


def extract_prediction_lines(row: Dict) -> List[Dict]:
    direct_keys = ("pred_lines", "prediction_lines", "lines")
    for key in direct_keys:
        value = row.get(key)
        if isinstance(value, list):
            return sanitize_pred_lines_uv(value)
        if isinstance(value, dict) and isinstance(value.get("lines"), list):
            return sanitize_pred_lines_uv(value.get("lines", []))

    text_keys = (
        "predict",
        "prediction",
        "pred_text",
        "generated_text",
        "response",
        "output",
        "assistant",
        "text",
    )
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, dict):
            if isinstance(value.get("lines"), list):
                return sanitize_pred_lines_uv(value.get("lines", []))
            if isinstance(value.get("text"), str):
                parsed = _parse_prediction_text(value["text"])
                if parsed:
                    return parsed
        if isinstance(value, str) and value.strip():
            parsed = _parse_prediction_text(value)
            if parsed:
                return parsed

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).strip().lower() != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parsed = _parse_prediction_text(content)
                if parsed:
                    return parsed
    return []


def build_prediction_index(
    prediction_rows: Sequence[Dict],
    meta_rows: Sequence[Dict],
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], Dict[str, int], int]:
    by_id: Dict[str, List[Dict]] = {}
    by_signature: Dict[str, List[Dict]] = {}
    parse_count = 0
    for row in prediction_rows:
        pred_lines = extract_prediction_lines(row)
        parse_count += 1 if pred_lines else 0
        sample_id = str(
            row.get("id")
            or row.get("sample_id")
            or row.get("source_id")
            or row.get("instance_id")
            or row.get("custom_id")
            or ""
        ).strip()
        if sample_id:
            by_id[sample_id] = pred_lines
        for signature in build_prediction_signature_variants(row):
            if signature and signature not in by_signature:
                by_signature[signature] = pred_lines

    fallback_index: Dict[str, int] = {}
    if len(by_id) < len(meta_rows) and len(prediction_rows) == len(meta_rows):
        for idx, meta in enumerate(meta_rows):
            fallback_index[str(meta.get("id"))] = idx
    return by_id, by_signature, fallback_index, parse_count


def resolve_family_manifest_path(fixed16_root: Path, family_manifest_arg: str) -> Optional[Path]:
    if str(family_manifest_arg).strip():
        path = Path(str(family_manifest_arg).strip()).resolve()
        return path if path.is_file() else None
    candidate = (fixed16_root.parent / "family_manifest.jsonl").resolve()
    return candidate if candidate.is_file() else None


def meta_source_sample_id(meta: Dict, family: Optional[Dict]) -> str:
    direct = str(meta.get("source_sample_id", "")).strip()
    if direct:
        return direct
    if family is not None:
        family_value = str(family.get("source_sample_id", family.get("source_image", meta.get("family_id", "")))).strip()
        if family_value:
            return family_value
    fallback = str(meta.get("source_id") or meta.get("family_id") or meta.get("id") or "").strip()
    return fallback


def infer_source_image_path_from_root(meta: Dict, family: Optional[Dict], source_root: Optional[Path], source_image_relpath: str) -> Optional[Path]:
    if source_root is None:
        return None
    sample_id = meta_source_sample_id(meta=meta, family=family)
    if not sample_id:
        return None
    relpath = str(source_image_relpath or "").strip().replace("\\", "/")
    if not relpath:
        return None
    path = (source_root / sample_id / Path(relpath)).resolve()
    if path.is_file():
        return path
    return None


def meta_source_image_path(meta: Dict, family: Optional[Dict], source_root: Optional[Path], source_image_relpath: str) -> Optional[Path]:
    direct = str(meta.get("source_image_path", "")).strip()
    if direct:
        path = Path(direct).resolve()
        if path.is_file():
            return path
    if family is not None:
        family_path = str(family.get("source_image_path", "")).strip()
        if family_path:
            path = Path(family_path).resolve()
            if path.is_file():
                return path
    inferred = infer_source_image_path_from_root(
        meta=meta,
        family=family,
        source_root=source_root,
        source_image_relpath=source_image_relpath,
    )
    if inferred is not None:
        return inferred
    return None


def collect_prediction_targets(
    meta_rows: Sequence[Dict],
    prediction_rows: Sequence[Dict],
    pred_by_id: Dict[str, List[Dict]],
    pred_by_signature: Dict[str, List[Dict]],
    fallback_index: Dict[str, int],
) -> Tuple[List[Tuple[Dict, List[Dict]]], int, int, List[str], str]:
    meta_by_id = {str(meta.get("id")): meta for meta in meta_rows}
    matched: List[Tuple[Dict, List[Dict]]] = []
    unmatched_prediction_ids: List[str] = []

    if pred_by_id:
        for meta in meta_rows:
            meta_id = str(meta.get("id"))
            if meta_id in pred_by_id:
                matched.append((meta, pred_by_id[meta_id]))
        for pred_id in sorted(pred_by_id.keys()):
            if pred_id not in meta_by_id:
                unmatched_prediction_ids.append(pred_id)
        return matched, len(matched), 0, unmatched_prediction_ids, "by_id"

    if pred_by_signature:
        matched_signatures: List[str] = []
        for meta in meta_rows:
            chosen_signature = None
            for signature in build_meta_signature_variants(meta):
                if signature in pred_by_signature:
                    chosen_signature = signature
                    break
            if chosen_signature is None:
                continue
            matched.append((meta, pred_by_signature[chosen_signature]))
            matched_signatures.append(chosen_signature)
        unmatched_prediction_signatures = sorted(sig for sig in pred_by_signature.keys() if sig not in set(matched_signatures))
        return matched, len(matched), 0, unmatched_prediction_signatures, "by_image_and_prompt"

    if fallback_index:
        missing_predictions = 0
        for meta in meta_rows:
            meta_id = str(meta.get("id"))
            if meta_id not in fallback_index:
                missing_predictions += 1
                continue
            pred_lines = extract_prediction_lines(prediction_rows[fallback_index[meta_id]])
            matched.append((meta, pred_lines))
        return matched, len(matched), missing_predictions, [], "fallback_by_order"

    return matched, 0, int(len(meta_rows)), [], "no_match"


def choose_best_meta_split(
    fixed16_root: Path,
    requested_split: str,
    prediction_rows: Sequence[Dict],
) -> Tuple[str, List[Dict], Dict[str, List[Dict]], Dict[str, List[Dict]], Dict[str, int], int, List[Tuple[Dict, List[Dict]]], int, int, List[str], str, List[Dict]]:
    candidates = ordered_split_candidates(fixed16_root=fixed16_root, requested_split=requested_split)
    if not candidates:
        raise FileNotFoundError(f"No fixed16 meta_*.jsonl files found under {fixed16_root}")

    best_payload = None
    best_score: Optional[Tuple[int, int, int]] = None
    split_probe: List[Dict] = []
    wanted = str(requested_split).strip()

    for candidate_split in candidates:
        meta_rows = load_meta_rows_for_split(fixed16_root=fixed16_root, split=candidate_split)
        if meta_rows is None:
            continue
        pred_by_id, pred_by_signature, fallback_index, parse_count = build_prediction_index(prediction_rows, meta_rows)
        matched_items, matched_predictions, missing_predictions, unmatched_prediction_ids, match_mode = collect_prediction_targets(
            meta_rows=meta_rows,
            prediction_rows=prediction_rows,
            pred_by_id=pred_by_id,
            pred_by_signature=pred_by_signature,
            fallback_index=fallback_index,
        )
        split_probe.append(
            {
                "split": str(candidate_split),
                "meta_row_count": int(len(meta_rows)),
                "matched_predictions": int(matched_predictions),
                "missing_predictions": int(missing_predictions),
                "prediction_match_mode": str(match_mode),
            }
        )
        score = (
            int(matched_predictions),
            1 if candidate_split == wanted else 0,
            -int(missing_predictions),
        )
        if best_payload is None or best_score is None or score > best_score:
            best_score = score
            best_payload = (
                str(candidate_split),
                meta_rows,
                pred_by_id,
                pred_by_signature,
                fallback_index,
                int(parse_count),
                matched_items,
                int(matched_predictions),
                int(missing_predictions),
                unmatched_prediction_ids,
                str(match_mode),
            )

    if best_payload is None:
        raise FileNotFoundError(f"No usable fixed16 meta rows found under {fixed16_root}")
    return (*best_payload, split_probe)


def clip_pred_lines_to_target_box(pred_lines: Sequence[Dict], target_box: Dict[str, int], boundary_tol_px: float) -> List[Dict]:
    rect = (
        float(target_box["x_min"]),
        float(target_box["y_min"]),
        float(target_box["x_max"]),
        float(target_box["y_max"]),
    )
    out: List[Dict] = []
    for line in pred_lines:
        if str(line.get("category", "lane_line")) != "lane_line":
            continue
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
            continue
        pieces = clip_polyline_to_rect(arr, rect)
        for piece in pieces:
            piece = dedup_points(piece)
            if piece.ndim != 2 or piece.shape[0] < 2:
                continue
            start_type = str(line.get("start_type", "start"))
            end_type = str(line.get("end_type", "end"))
            if point_boundary_side(piece[0], rect, float(boundary_tol_px)) is not None:
                start_type = "cut"
            elif start_type not in {"start", "cut"}:
                start_type = "start"
            if point_boundary_side(piece[-1], rect, float(boundary_tol_px)) is not None:
                end_type = "cut"
            elif end_type not in {"end", "cut"}:
                end_type = "end"
            out.append(
                {
                    "category": "lane_line",
                    "start_type": start_type,
                    "end_type": end_type,
                    "points": [[float(x), float(y)] for x, y in piece.tolist()],
                }
            )
    return out


def to_global_lines(pred_lines: Sequence[Dict], crop_box: Dict[str, int]) -> List[Dict]:
    x0 = float(crop_box["x_min"])
    y0 = float(crop_box["y_min"])
    out: List[Dict] = []
    for line in pred_lines:
        arr = np.asarray(line.get("points", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        global_arr = dedup_points(arr + np.asarray([[x0, y0]], dtype=np.float32))
        if global_arr.ndim != 2 or global_arr.shape[0] < 2:
            continue
        out.append(
            {
                "category": "lane_line",
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points_global": global_arr,
            }
        )
    return out


def _merge_key(line: Dict) -> Tuple[str]:
    return (str(line.get("category", "lane_line")),)


def _endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _merge_two_lines(a: Dict, b: Dict, tol_px: float) -> Optional[Dict]:
    if _merge_key(a) != _merge_key(b):
        return None
    a_pts = np.asarray(a["points_global"], dtype=np.float32)
    b_pts = np.asarray(b["points_global"], dtype=np.float32)
    if a_pts.ndim != 2 or b_pts.ndim != 2 or a_pts.shape[0] < 2 or b_pts.shape[0] < 2:
        return None

    candidates = []
    if str(a.get("end_type")) == "cut" and str(b.get("start_type")) == "cut":
        candidates.append(("a_end_b_start", a_pts[-1], b_pts[0]))
    if str(a.get("end_type")) == "cut" and str(b.get("end_type")) == "cut":
        candidates.append(("a_end_b_end", a_pts[-1], b_pts[-1]))
    if str(a.get("start_type")) == "cut" and str(b.get("start_type")) == "cut":
        candidates.append(("a_start_b_start", a_pts[0], b_pts[0]))
    if str(a.get("start_type")) == "cut" and str(b.get("end_type")) == "cut":
        candidates.append(("a_start_b_end", a_pts[0], b_pts[-1]))

    best_mode = None
    best_dist = float("inf")
    for mode, p0, p1 in candidates:
        dist = _endpoint_distance(np.asarray(p0), np.asarray(p1))
        if dist <= float(tol_px) and dist < best_dist:
            best_mode = mode
            best_dist = dist

    if best_mode is None:
        return None

    if best_mode == "a_end_b_start":
        join = 0.5 * (a_pts[-1] + b_pts[0])
        pts = np.vstack([a_pts[:-1], join[None, :], b_pts[1:]])
        start_type = str(a.get("start_type", "start"))
        end_type = str(b.get("end_type", "end"))
    elif best_mode == "a_end_b_end":
        join = 0.5 * (a_pts[-1] + b_pts[-1])
        pts = np.vstack([a_pts[:-1], join[None, :], b_pts[-2::-1]])
        start_type = str(a.get("start_type", "start"))
        end_type = str(b.get("start_type", "start"))
    elif best_mode == "a_start_b_start":
        join = 0.5 * (a_pts[0] + b_pts[0])
        pts = np.vstack([a_pts[:0:-1], join[None, :], b_pts[1:]])
        start_type = str(a.get("end_type", "end"))
        end_type = str(b.get("end_type", "end"))
    else:
        join = 0.5 * (a_pts[0] + b_pts[-1])
        pts = np.vstack([b_pts[:-1], join[None, :], a_pts[1:]])
        start_type = str(b.get("start_type", "start"))
        end_type = str(a.get("end_type", "end"))

    merged_pts = dedup_points(pts)
    if merged_pts.ndim != 2 or merged_pts.shape[0] < 2:
        return None
    return {
        "category": "lane_line",
        "start_type": start_type,
        "end_type": end_type,
        "points_global": merged_pts,
    }


def merge_cut_connected_lines(lines: Sequence[Dict], tol_px: float) -> List[Dict]:
    working = [dict(line) for line in lines]
    changed = True
    while changed:
        changed = False
        for i in range(len(working)):
            if changed:
                break
            for j in range(i + 1, len(working)):
                merged = _merge_two_lines(working[i], working[j], tol_px=float(tol_px))
                if merged is None:
                    continue
                working[i] = merged
                del working[j]
                changed = True
                break
    return working


def build_lane_features(lines: Sequence[Dict], transform) -> List[Dict]:
    features: List[Dict] = []
    for line in lines:
        arr = np.asarray(line.get("points_global", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "category": "lane_line",
                    "start_type": str(line.get("start_type", "start")),
                    "end_type": str(line.get("end_type", "end")),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": pixel_to_world(arr.tolist(), transform),
                },
            }
        )
    return features


def build_lane_pixel_features(lines: Sequence[Dict]) -> List[Dict]:
    features: List[Dict] = []
    for line in lines:
        arr = np.asarray(line.get("points_global", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "category": "lane_line",
                    "start_type": str(line.get("start_type", "start")),
                    "end_type": str(line.get("end_type", "end")),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(x), float(y)] for x, y in arr.tolist()],
                },
            }
        )
    return features


def infer_canvas_size(meta_rows: Sequence[Dict]) -> Tuple[int, int]:
    width = 0
    height = 0
    for meta in meta_rows:
        image_size = meta.get("image_size", [])
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            try:
                width = max(width, int(image_size[0]))
                height = max(height, int(image_size[1]))
            except Exception:
                pass
        crop_box = dict(meta.get("crop_box", {}))
        try:
            width = max(width, int(crop_box.get("x_max", 0)))
            height = max(height, int(crop_box.get("y_max", 0)))
        except Exception:
            pass
    return max(1, width), max(1, height)


def build_pixel_canvas(fixed16_root: Path, meta_rows: Sequence[Dict]) -> Image.Image:
    width, height = infer_canvas_size(meta_rows)
    canvas = Image.new("RGB", (int(width), int(height)), color=(0, 0, 0))
    pasted_keys = set()
    for meta in meta_rows:
        image_rel = str(meta.get("image", "")).strip()
        crop_box = dict(meta.get("crop_box", {}))
        if not image_rel or not crop_box:
            continue
        dedup_key = (image_rel, int(crop_box.get("x_min", 0)), int(crop_box.get("y_min", 0)))
        if dedup_key in pasted_keys:
            continue
        pasted_keys.add(dedup_key)
        patch_path = (fixed16_root / image_rel).resolve()
        if not patch_path.is_file():
            continue
        try:
            with Image.open(patch_path) as patch_image:
                patch_rgb = patch_image.convert("RGB")
                canvas.paste(patch_rgb, (int(crop_box.get("x_min", 0)), int(crop_box.get("y_min", 0))))
        except Exception:
            continue
    return canvas


def draw_endpoint_marker(draw: ImageDraw.ImageDraw, point: np.ndarray, color: Tuple[int, int, int], radius: int = 3) -> None:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_line_set(draw: ImageDraw.ImageDraw, lines: Sequence[Dict], color: Tuple[int, int, int], width: int) -> None:
    for line in lines:
        arr = np.asarray(line.get("points_global", []), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        points = [tuple(float(v) for v in point) for point in arr.tolist()]
        draw.line(points, fill=color, width=width)
        start_type = str(line.get("start_type", "start"))
        end_type = str(line.get("end_type", "end"))
        start_color = (0, 220, 90) if start_type == "start" else (255, 60, 60) if start_type == "cut" else (255, 215, 0)
        end_color = (0, 220, 90) if end_type == "start" else (255, 60, 60) if end_type == "cut" else (255, 215, 0)
        draw_endpoint_marker(draw, arr[0], start_color, radius=3)
        draw_endpoint_marker(draw, arr[-1], end_color, radius=3)


def save_pixel_overlay(
    fixed16_root: Path,
    meta_rows: Sequence[Dict],
    raw_lines: Sequence[Dict],
    merged_lines: Sequence[Dict],
    out_path: Path,
) -> None:
    canvas = build_pixel_canvas(fixed16_root=fixed16_root, meta_rows=meta_rows)
    draw = ImageDraw.Draw(canvas)
    draw_line_set(draw, raw_lines, color=(90, 200, 255), width=2)
    draw_line_set(draw, merged_lines, color=(255, 140, 0), width=4)
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    fixed16_root = Path(args.fixed16_root).resolve()
    predictions_path = Path(args.predictions_path).resolve()
    output_root = Path(args.output_root).resolve()
    source_root = Path(str(args.source_root).strip()).resolve() if str(args.source_root).strip() else None
    ensure_dir(output_root)

    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    family_manifest = resolve_family_manifest_path(fixed16_root=fixed16_root, family_manifest_arg=str(args.family_manifest))
    manifest_rows = load_jsonl(family_manifest) if family_manifest is not None else []
    manifest_map = {str(row["family_id"]): row for row in manifest_rows if isinstance(row, dict) and str(row.get("family_id", "")).strip()}
    prediction_rows = load_prediction_rows(predictions_path)
    (
        resolved_split,
        meta_rows,
        pred_by_id,
        pred_by_signature,
        fallback_index,
        parse_count,
        matched_items,
        matched_predictions,
        missing_predictions,
        unmatched_prediction_ids,
        match_mode,
        split_probe,
    ) = choose_best_meta_split(
        fixed16_root=fixed16_root,
        requested_split=str(args.split),
        prediction_rows=prediction_rows,
    )

    grouped_global_lines: Dict[str, List[Dict]] = defaultdict(list)
    grouped_family: Dict[str, Optional[Dict]] = {}
    grouped_source_image_path: Dict[str, Path] = {}
    grouped_meta_rows: Dict[str, List[Dict]] = defaultdict(list)
    raw_piece_count = 0
    missing_source_image_rows = 0

    for meta, pred_lines in matched_items:
        family = manifest_map.get(str(meta.get("family_id")))
        source_sample_id = meta_source_sample_id(meta=meta, family=family)
        if str(args.source_sample_id).strip() and source_sample_id != str(args.source_sample_id).strip():
            continue
        source_image_path = meta_source_image_path(
            meta=meta,
            family=family,
            source_root=source_root,
            source_image_relpath=str(args.source_image_relpath),
        )
        grouped_family[source_sample_id] = family
        if source_image_path is not None:
            grouped_source_image_path[source_sample_id] = source_image_path
        else:
            missing_source_image_rows += 1
        grouped_meta_rows[source_sample_id].append(meta)
        clipped_local_lines = clip_pred_lines_to_target_box(
            pred_lines=pred_lines,
            target_box=dict(meta.get("target_box", {})),
            boundary_tol_px=float(args.merge_endpoint_tol_px),
        )
        raw_piece_count += len(clipped_local_lines)
        grouped_global_lines[source_sample_id].extend(
            to_global_lines(
                pred_lines=clipped_local_lines,
                crop_box=dict(meta.get("crop_box", {})),
            )
        )

    sample_ids = sorted(grouped_global_lines.keys())
    if int(args.max_samples) > 0:
        sample_ids = sample_ids[: int(args.max_samples)]

    summary: List[Dict] = []
    for sample_id in sample_ids:
        family = grouped_family.get(sample_id)
        source_image_path = grouped_source_image_path.get(sample_id)
        sample_meta_rows = grouped_meta_rows.get(sample_id, [])
        raw_lines = grouped_global_lines[sample_id]
        merged_lines = merge_cut_connected_lines(raw_lines, tol_px=float(args.merge_endpoint_tol_px))

        sample_out = output_root / sample_id
        ensure_dir(sample_out)
        raw_lane_pixel_geojson = build_pixel_feature_collection(build_lane_pixel_features(raw_lines), name="Lane_raw_pixel")
        merged_lane_pixel_geojson = build_pixel_feature_collection(build_lane_pixel_features(merged_lines), name="Lane_pixel")

        raw_lane_pixel_path = sample_out / "Lane.raw.pixel.geojson"
        lane_pixel_path = sample_out / "Lane.pixel.geojson"
        with raw_lane_pixel_path.open("w", encoding="utf-8") as f:
            json.dump(raw_lane_pixel_geojson, f, ensure_ascii=False, indent=2)
        with lane_pixel_path.open("w", encoding="utf-8") as f:
            json.dump(merged_lane_pixel_geojson, f, ensure_ascii=False, indent=2)

        overlay_pixel_path = sample_out / "overlay.pixel.png"
        save_pixel_overlay(
            fixed16_root=fixed16_root,
            meta_rows=sample_meta_rows,
            raw_lines=raw_lines,
            merged_lines=merged_lines,
            out_path=overlay_pixel_path,
        )

        lane_path = ""
        raw_lane_path = ""
        if source_image_path is not None:
            with rasterio_open(source_image_path) as ds:
                transform = ds.transform
                crs_name = str(ds.crs) if ds.crs is not None else "urn:ogc:def:crs:OGC:1.3:CRS84"

            raw_lane_geojson = build_feature_collection(build_lane_features(raw_lines, transform), crs_name=crs_name, name="Lane_raw")
            merged_lane_geojson = build_feature_collection(build_lane_features(merged_lines, transform), crs_name=crs_name, name="Lane")

            raw_lane_path_obj = sample_out / "Lane.raw.geojson"
            lane_path_obj = sample_out / "Lane.geojson"
            with raw_lane_path_obj.open("w", encoding="utf-8") as f:
                json.dump(raw_lane_geojson, f, ensure_ascii=False, indent=2)
            with lane_path_obj.open("w", encoding="utf-8") as f:
                json.dump(merged_lane_geojson, f, ensure_ascii=False, indent=2)
            raw_lane_path = str(raw_lane_path_obj)
            lane_path = str(lane_path_obj)

        summary.append(
            {
                "source_sample_id": sample_id,
                "source_image_path": "" if source_image_path is None else str(source_image_path),
                "family_id": "" if family is None else str(family.get("family_id", "")),
                "raw_piece_count": len(raw_lines),
                "merged_lane_count": len(merged_lines),
                "lane_geojson_path": lane_path,
                "raw_lane_geojson_path": raw_lane_path,
                "lane_pixel_geojson_path": str(lane_pixel_path),
                "raw_lane_pixel_geojson_path": str(raw_lane_pixel_path),
                "overlay_pixel_path": str(overlay_pixel_path),
                "pixel_only": bool(source_image_path is None),
            }
        )

    summary_path = output_root / "reconstruct_fixed16_predictions.summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed16_root": str(fixed16_root),
                "predictions_path": str(predictions_path),
                "family_manifest": "" if family_manifest is None else str(family_manifest),
                "source_root": "" if source_root is None else str(source_root),
                "source_image_relpath": str(args.source_image_relpath),
                "requested_split": str(args.split),
                "resolved_split": str(resolved_split),
                "prediction_match_mode": str(match_mode),
                "matched_predictions": int(matched_predictions),
                "missing_predictions": int(missing_predictions),
                "unmatched_prediction_ids": unmatched_prediction_ids,
                "parsed_prediction_rows": int(parse_count),
                "raw_piece_count": int(raw_piece_count),
                "stitched_sample_count": int(len(summary)),
                "skipped_missing_source_image": 0,
                "missing_source_image_rows": int(missing_source_image_rows),
                "split_probe": split_probe,
                "samples": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Fixed16 Stitch] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
