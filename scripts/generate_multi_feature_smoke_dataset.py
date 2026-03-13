import argparse
import copy
import json
import os
import shutil
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unimapgen.geo.io import geojson_to_pixel_features, load_geojson, pixel_features_to_geojson, read_raster_meta
from unimapgen.geo.pipeline import load_geo_task_schemas
from unimapgen.utils import ensure_dir, load_yaml


def _find_source_sample(dataset_root: str, split: str) -> str:
    split_dir = os.path.join(dataset_root, split)
    candidates = sorted(
        os.path.join(split_dir, name)
        for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )
    if not candidates:
        raise FileNotFoundError(f"no source sample found under: {split_dir}")
    return candidates[0]


def _clip_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.size == 0:
        return pts.reshape(0, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(0, width - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(0, height - 1)))
    return pts.astype(np.float32)


def _build_grid_offsets(
    count: int,
    step_x: float,
    step_y: float,
    cols: int,
    variant_index: int,
) -> List[Tuple[float, float]]:
    total = max(1, int(count))
    num_cols = max(1, int(cols))
    base_dx = float(variant_index % 3) * float(step_x) * 0.35
    base_dy = float((variant_index // 3) % 3) * float(step_y) * 0.35
    offsets: List[Tuple[float, float]] = []
    for index in range(total):
        row = index // num_cols
        col = index % num_cols
        offsets.append(
            (
                base_dx + float(col) * float(step_x),
                base_dy + float(row) * float(step_y),
            )
        )
    return offsets


def _shift_feature(feature: Dict, dx: float, dy: float, width: int, height: int, suffix: str) -> Dict:
    rec = copy.deepcopy(feature)
    points = np.asarray(rec.get("points", []), dtype=np.float32)
    if points.ndim == 2 and points.shape[0] > 0:
        shifted = points.copy()
        shifted[:, 0] += float(dx)
        shifted[:, 1] += float(dy)
        rec["points"] = _clip_points(shifted, width=width, height=height)
    if rec.get("rings"):
        shifted_rings = []
        for ring in rec.get("rings", []):
            ring_pts = np.asarray(ring, dtype=np.float32)
            shifted_ring = ring_pts.copy()
            shifted_ring[:, 0] += float(dx)
            shifted_ring[:, 1] += float(dy)
            shifted_rings.append(_clip_points(shifted_ring, width=width, height=height))
        rec["rings"] = shifted_rings
        if shifted_rings:
            rec["points"] = shifted_rings[0]
    props = dict(rec.get("properties", {}))
    if "Id" in props:
        props["Id"] = f"{props['Id']}_{suffix}"
    else:
        props["Id"] = suffix
    rec["properties"] = props
    return rec


def _synthesize_task_features(
    base_features: Sequence[Dict],
    count: int,
    width: int,
    height: int,
    offsets: Sequence[Tuple[float, float]],
    task_name: str,
) -> List[Dict]:
    if not base_features:
        return []
    out: List[Dict] = []
    for index in range(max(1, int(count))):
        source = base_features[index % len(base_features)]
        dx, dy = offsets[index]
        out.append(
            _shift_feature(
                feature=source,
                dx=float(dx),
                dy=float(dy),
                width=width,
                height=height,
                suffix=f"{task_name}_{index + 1:03d}",
            )
        )
    return out


def _copy_sample_rasters(src_sample_dir: str, dst_sample_dir: str) -> None:
    src_patch_dir = os.path.join(src_sample_dir, "patch_tif")
    dst_patch_dir = os.path.join(dst_sample_dir, "patch_tif")
    ensure_dir(dst_patch_dir)
    for name in ("0.tif", "0_edit_poly.tif"):
        shutil.copy2(os.path.join(src_patch_dir, name), os.path.join(dst_patch_dir, name))


def _write_sample_labels(
    cfg: Dict,
    src_sample_dir: str,
    dst_sample_dir: str,
    task_schemas: Dict,
    lane_count: int,
    intersection_count: int,
    cols: int,
    lane_step_x: float,
    lane_step_y: float,
    intersection_step_x: float,
    intersection_step_y: float,
    variant_index: int,
) -> Dict:
    ensure_dir(os.path.join(dst_sample_dir, "label_check_crop"))
    raster_meta = read_raster_meta(os.path.join(src_sample_dir, str(cfg["data"]["image_relpath"])))
    width = int(raster_meta.width)
    height = int(raster_meta.height)
    summary: Dict[str, int] = {}
    for task_name, task_schema in task_schemas.items():
        relpath = str(cfg["data"]["label_relpaths"][task_name])
        src_label_path = os.path.join(src_sample_dir, relpath)
        base_features = geojson_to_pixel_features(
            geojson_dict=load_geojson(src_label_path),
            task_schema=task_schema,
            raster_meta=raster_meta,
        )
        if task_name == "lane":
            offsets = _build_grid_offsets(
                count=int(lane_count),
                step_x=float(lane_step_x),
                step_y=float(lane_step_y),
                cols=int(cols),
                variant_index=int(variant_index),
            )
            features = _synthesize_task_features(
                base_features=base_features,
                count=int(lane_count),
                width=width,
                height=height,
                offsets=offsets,
                task_name=task_name,
            )
        else:
            offsets = _build_grid_offsets(
                count=int(intersection_count),
                step_x=float(intersection_step_x),
                step_y=float(intersection_step_y),
                cols=int(cols),
                variant_index=int(variant_index),
            )
            features = _synthesize_task_features(
                base_features=base_features,
                count=int(intersection_count),
                width=width,
                height=height,
                offsets=offsets,
                task_name=task_name,
            )
        geojson_dict = pixel_features_to_geojson(
            task_schema=task_schema,
            feature_records=features,
            raster_meta=raster_meta,
        )
        dst_label_path = os.path.join(dst_sample_dir, relpath)
        ensure_dir(os.path.dirname(dst_label_path))
        with open(dst_label_path, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, ensure_ascii=False, indent=2)
        summary[task_name] = int(len(features))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a multi-feature smoke dataset from the base smoke sample.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_root", default="data/geo_vector_dataset_smoke")
    parser.add_argument("--output_root", default="data/geo_vector_dataset_smoke_multi")
    parser.add_argument("--train_variants", type=int, default=6)
    parser.add_argument("--val_variants", type=int, default=2)
    parser.add_argument("--lane_count", type=int, default=6)
    parser.add_argument("--intersection_count", type=int, default=4)
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--lane_step_x", type=float, default=26.0)
    parser.add_argument("--lane_step_y", type=float, default=16.0)
    parser.add_argument("--intersection_step_x", type=float, default=24.0)
    parser.add_argument("--intersection_step_y", type=float, default=24.0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_schemas = load_geo_task_schemas(cfg)
    output_root = os.path.abspath(str(args.output_root))
    ensure_dir(output_root)

    summary = {
        "source_root": os.path.abspath(str(args.source_root)),
        "output_root": output_root,
        "train_variants": int(args.train_variants),
        "val_variants": int(args.val_variants),
        "lane_count": int(args.lane_count),
        "intersection_count": int(args.intersection_count),
        "samples": [],
    }

    for split, variant_count in (("train", int(args.train_variants)), ("val", int(args.val_variants))):
        src_sample_dir = _find_source_sample(dataset_root=str(args.source_root), split=split)
        split_out_dir = os.path.join(output_root, split)
        ensure_dir(split_out_dir)
        for variant_index in range(max(1, int(variant_count))):
            sample_id = f"sample_{split}_{variant_index + 1:04d}"
            dst_sample_dir = os.path.join(split_out_dir, sample_id)
            ensure_dir(dst_sample_dir)
            _copy_sample_rasters(src_sample_dir=src_sample_dir, dst_sample_dir=dst_sample_dir)
            counts = _write_sample_labels(
                cfg=cfg,
                src_sample_dir=src_sample_dir,
                dst_sample_dir=dst_sample_dir,
                task_schemas=task_schemas,
                lane_count=int(args.lane_count),
                intersection_count=int(args.intersection_count),
                cols=int(args.grid_cols),
                lane_step_x=float(args.lane_step_x),
                lane_step_y=float(args.lane_step_y),
                intersection_step_x=float(args.intersection_step_x),
                intersection_step_y=float(args.intersection_step_y),
                variant_index=int(variant_index),
            )
            summary["samples"].append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "sample_dir": dst_sample_dir,
                    "lane_features": int(counts.get("lane", 0)),
                    "intersection_features": int(counts.get("intersection", 0)),
                }
            )

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
