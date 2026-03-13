import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unimapgen.geo.coord_sequence import (
    points_abs_to_uv,
    uv_feature_records_to_target_items,
    uv_items_to_abs_feature_records,
)
from unimapgen.geo.geometry import build_resize_context
from unimapgen.geo.io import geojson_to_pixel_features, load_geojson, pixel_features_to_geojson, read_raster_meta
from unimapgen.geo.pipeline import load_geo_task_schemas
from unimapgen.geo.tokenizer import GeoCoordTokenizer
from unimapgen.utils import load_yaml


def _find_sample_dir(dataset_root: str, split: str, sample_id: str = "") -> str:
    split_dir = os.path.join(dataset_root, split)
    if sample_id:
        sample_dir = os.path.join(split_dir, sample_id)
        if not os.path.isdir(sample_dir):
            raise FileNotFoundError(f"sample_id not found: {sample_dir}")
        return sample_dir
    candidates = sorted(
        [
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"no sample dirs found under: {split_dir}")
    return candidates[0]


def _clip_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.size == 0:
        return pts.reshape(0, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(0, width - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(0, height - 1)))
    return pts.astype(np.float32)


def _duplicate_features(
    feature_records: Sequence[Dict],
    width: int,
    height: int,
    offsets: Sequence[Tuple[float, float]],
) -> List[Dict]:
    if not feature_records:
        return []
    base = copy.deepcopy(feature_records[0])
    out: List[Dict] = []
    for dup_index, (dx, dy) in enumerate(offsets):
        rec = copy.deepcopy(base)
        points = np.asarray(rec.get("points", []), dtype=np.float32)
        shifted = points.copy()
        shifted[:, 0] += float(dx)
        shifted[:, 1] += float(dy)
        rec["points"] = _clip_points(shifted, width=width, height=height)
        props = dict(rec.get("properties", {}))
        if "Id" in props:
            props["Id"] = f"{props['Id']}_dup{dup_index + 1}"
        else:
            props["Id"] = f"dup_{dup_index + 1}"
        rec["properties"] = props
        out.append(rec)
    return out


def _build_grid_offsets(
    count: int,
    step_x: float,
    step_y: float,
    cols: int,
) -> List[Tuple[float, float]]:
    total = max(1, int(count))
    num_cols = max(1, int(cols))
    offsets: List[Tuple[float, float]] = []
    for index in range(total):
        row = index // num_cols
        col = index % num_cols
        offsets.append((float(col) * float(step_x), float(row) * float(step_y)))
    return offsets


def _feature_records_to_uv(feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
    out: List[Dict] = []
    for feature in feature_records:
        out.append(
            {
                "properties": dict(feature.get("properties", {})),
                "points": points_abs_to_uv(
                    points_xy=np.asarray(feature.get("points", []), dtype=np.float32),
                    resize_ctx=resize_ctx,
                ),
            }
        )
    return out


def _save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-trip test for multi-feature Lane/Intersection GeoJSON.")
    parser.add_argument("--config", required=True, help="Path to geo config yaml")
    parser.add_argument("--split", default="train", help="Dataset split to sample from")
    parser.add_argument("--sample_id", default="", help="Optional sample id; defaults to first sample in split")
    parser.add_argument("--lane_count", type=int, default=8, help="Synthetic lane feature count for the round-trip test")
    parser.add_argument(
        "--intersection_count",
        type=int,
        default=6,
        help="Synthetic intersection feature count for the round-trip test",
    )
    parser.add_argument("--offset_step_x", type=float, default=4.0, help="Synthetic duplicate x-offset in pixels")
    parser.add_argument("--offset_step_y", type=float, default=4.0, help="Synthetic duplicate y-offset in pixels")
    parser.add_argument("--grid_cols", type=int, default=3, help="How many synthetic objects to place per row")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to write report and round-trip GeoJSON outputs",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_schemas = load_geo_task_schemas(cfg)
    tokenizer = GeoCoordTokenizer(
        qwen_model_path=str(cfg["model"]["qwen_model_path"]),
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        trust_remote_code=True,
        coord_bins=int(cfg.get("serialization", {}).get("coord_bins", 1024)),
    )

    dataset_root = str(cfg["data"]["dataset_root"])
    sample_dir = _find_sample_dir(dataset_root=dataset_root, split=str(args.split), sample_id=str(args.sample_id))
    image_path = os.path.join(sample_dir, str(cfg["data"]["image_relpath"]))
    raster_meta = read_raster_meta(image_path)
    resize_ctx = build_resize_context(
        width=int(raster_meta.width),
        height=int(raster_meta.height),
        target_size=int(cfg["data"]["image_size"]),
        crop_bbox=None,
    )

    if args.output_dir:
        output_dir = str(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", "geo_vector_smoke", f"multi_feature_roundtrip_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "config": str(args.config),
        "split": str(args.split),
        "sample_dir": sample_dir,
        "image_path": image_path,
        "image_size": int(cfg["data"]["image_size"]),
        "coord_bins": int(cfg.get("serialization", {}).get("coord_bins", 1024)),
        "lane_count": int(args.lane_count),
        "intersection_count": int(args.intersection_count),
        "offset_step_x": float(args.offset_step_x),
        "offset_step_y": float(args.offset_step_y),
        "grid_cols": int(args.grid_cols),
        "tasks": {},
    }

    offsets_by_task = {
        "lane": _build_grid_offsets(
            count=int(args.lane_count),
            step_x=float(args.offset_step_x),
            step_y=float(args.offset_step_y),
            cols=int(args.grid_cols),
        ),
        "intersection": _build_grid_offsets(
            count=int(args.intersection_count),
            step_x=float(args.offset_step_x),
            step_y=float(args.offset_step_y),
            cols=int(args.grid_cols),
        ),
    }

    for task_name, task_schema in task_schemas.items():
        label_relpath = str(cfg["data"]["label_relpaths"][task_name])
        label_path = os.path.join(sample_dir, label_relpath)
        geojson_dict = load_geojson(label_path)
        feature_records = geojson_to_pixel_features(
            geojson_dict=geojson_dict,
            task_schema=task_schema,
            raster_meta=raster_meta,
        )
        synthetic_records = _duplicate_features(
            feature_records=feature_records,
            width=int(raster_meta.width),
            height=int(raster_meta.height),
            offsets=offsets_by_task.get(task_name, [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]),
        )
        feature_records_uv = _feature_records_to_uv(synthetic_records, resize_ctx=resize_ctx)
        target_items = uv_feature_records_to_target_items(
            feature_records=feature_records_uv,
            task_schema=task_schema,
            image_size=int(cfg["data"]["image_size"]),
        )
        token_ids = tokenizer.encode_map_items(
            map_items=target_items,
            image_size=int(cfg["data"]["image_size"]),
            max_length=None,
            append_eos=True,
        )
        decoded_items, decode_info = tokenizer.decode_map_items(
            token_ids=token_ids,
            task_schema=task_schema,
            image_size=int(cfg["data"]["image_size"]),
        )
        decoded_abs = uv_items_to_abs_feature_records(
            items=decoded_items,
            task_schema=task_schema,
            resize_ctx=resize_ctx,
        )
        roundtrip_geojson = pixel_features_to_geojson(
            task_schema=task_schema,
            feature_records=decoded_abs,
            raster_meta=raster_meta,
        )

        _save_json(os.path.join(output_dir, f"{task_schema.collection_name}.synthetic_input.geojson"), pixel_features_to_geojson(
            task_schema=task_schema,
            feature_records=synthetic_records,
            raster_meta=raster_meta,
        ))
        _save_json(os.path.join(output_dir, f"{task_schema.collection_name}.roundtrip.geojson"), roundtrip_geojson)

        task_report = {
            "input_feature_count": int(len(synthetic_records)),
            "target_item_count": int(len(target_items)),
            "token_count": int(len(token_ids)),
            "decoded_item_count": int(len(decoded_items)),
            "decode_info": dict(decode_info),
            "roundtrip_feature_count": int(len(roundtrip_geojson.get("features", []))),
        }
        report["tasks"][task_name] = task_report

        expected_min = int(args.lane_count) if task_name == "lane" else int(args.intersection_count)
        if int(task_report["roundtrip_feature_count"]) < expected_min:
            raise RuntimeError(
                f"task={task_name} roundtrip_feature_count={task_report['roundtrip_feature_count']} < expected_min={expected_min}"
            )

    _save_json(os.path.join(output_dir, "report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
