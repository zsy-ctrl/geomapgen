import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio import open as rasterio_open

from geo_current_dataset_v1_common import load_jsonl


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest_map(path: Path) -> Dict[str, Dict]:
    return {str(row["family_id"]): row for row in load_jsonl(path)}


def load_dataset_image_map(path: Path) -> Dict[str, str]:
    rows = load_jsonl(path)
    out: Dict[str, str] = {}
    for row in rows:
        image_list = row.get("images", [])
        if isinstance(image_list, list) and len(image_list) > 0:
            out[str(row.get("id"))] = str(image_list[0])
    return out


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


def _line_color(category: str) -> Tuple[int, int, int]:
    if str(category) == "intersection_boundary":
        return (255, 180, 0)
    return (0, 255, 255)


def build_overlay_image(
    canvas: np.ndarray,
    visual_lines: List[Dict],
) -> Image.Image:
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for row in visual_lines:
        points = row.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        color = _line_color(str(row.get("category", "")))
        xy = [(int(pt[0]), int(pt[1])) for pt in points]
        draw.line(xy, fill=color, width=3)

        start_xy = xy[0]
        end_xy = xy[-1]
        start_type = str(row.get("start_type", "")).strip() or "start"
        end_type = str(row.get("end_type", "")).strip() or "end"

        start_r = 5
        end_r = 5
        draw.ellipse(
            (start_xy[0] - start_r, start_xy[1] - start_r, start_xy[0] + start_r, start_xy[1] + start_r),
            fill=(0, 255, 0),
            outline=(0, 0, 0),
        )
        draw.ellipse(
            (end_xy[0] - end_r, end_xy[1] - end_r, end_xy[0] + end_r, end_xy[1] + end_r),
            fill=(255, 80, 80),
            outline=(0, 0, 0),
        )
        draw.text((start_xy[0] + 6, start_xy[1] - 10), start_type, fill=(0, 255, 0), font=font)
        draw.text((end_xy[0] + 6, end_xy[1] - 10), end_type, fill=(255, 80, 80), font=font)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct full-size masked GeoTIFF and GeoJSON from processed geo-current training dataset."
    )
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stage", type=str, default="stage_a", choices=["stage_a", "stage_b"])
    parser.add_argument("--family-manifest", type=str, default="")
    parser.add_argument("--source-sample-id", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    family_manifest = Path(args.family_manifest).resolve() if str(args.family_manifest).strip() else processed_root / "family_manifest.jsonl"
    dataset_root = processed_root / str(args.stage) / "dataset"
    meta_path = dataset_root / f"meta_{args.split}.jsonl"
    rows_path = dataset_root / f"{args.split}.jsonl"

    manifest_map = load_manifest_map(family_manifest)
    meta_rows = load_jsonl(meta_path)
    image_map = load_dataset_image_map(rows_path)

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    source_meta: Dict[str, Dict] = {}
    for row in meta_rows:
        family = manifest_map.get(str(row["family_id"]))
        if family is None:
            continue
        sample_id = str(family.get("source_sample_id", family.get("source_image", row["family_id"])))
        if str(args.source_sample_id).strip() and sample_id != str(args.source_sample_id).strip():
            continue
        grouped[sample_id].append(row)
        source_meta[sample_id] = family

    sample_ids = sorted(grouped.keys())
    if int(args.max_samples) > 0:
        sample_ids = sample_ids[: int(args.max_samples)]

    summary: List[Dict] = []
    for sample_id in sample_ids:
        family = source_meta[sample_id]
        source_image_path = Path(str(family["source_image_path"])).resolve()
        with rasterio_open(source_image_path) as ds:
            width = int(ds.width)
            height = int(ds.height)
            profile = ds.profile.copy()
            transform = ds.transform
            crs_name = str(ds.crs) if ds.crs is not None else "urn:ogc:def:crs:OGC:1.3:CRS84"

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        lane_features: List[Dict] = []
        inter_features: List[Dict] = []
        visual_lines: List[Dict] = []
        seen_lane = set()
        seen_inter = set()

        sample_out = output_root / sample_id
        ensure_dir(sample_out)

        for row in grouped[sample_id]:
            row_id = str(row["id"])
            image_rel = image_map.get(row_id, "")
            if not image_rel:
                continue
            patch_path = dataset_root / Path(image_rel)
            if not patch_path.is_file():
                continue
            crop_box = row["crop_box"]
            x0 = int(crop_box["x_min"])
            y0 = int(crop_box["y_min"])
            x1 = int(crop_box["x_max"])
            y1 = int(crop_box["y_max"])
            patch = rasterio_open(patch_path).read() if patch_path.suffix.lower() in {".tif", ".tiff"} else None
            if patch is None:
                from PIL import Image

                patch_img = np.asarray(Image.open(patch_path).convert("RGB"), dtype=np.uint8)
            else:
                patch_img = np.transpose(patch, (1, 2, 0)).astype(np.uint8)
            canvas[y0:y1, x0:x1] = patch_img[: y1 - y0, : x1 - x0]

            for line in row.get("target_lines", []):
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[int(p[0]) + x0, int(p[1]) + y0] for p in points]
                key = (
                    str(line.get("category", "")),
                    str(line.get("start_type", "")),
                    str(line.get("end_type", "")),
                    tuple((int(p[0]), int(p[1])) for p in global_points),
                )
                feature = {
                    "type": "Feature",
                    "properties": {
                        "category": str(line.get("category", "")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": pixel_to_world(global_points, transform),
                    },
                }
                visual_lines.append(
                    {
                        "category": str(line.get("category", "")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                        "points": global_points,
                    }
                )
                if str(line.get("category", "")) == "intersection_boundary":
                    if key not in seen_inter:
                        seen_inter.add(key)
                        inter_features.append(feature)
                else:
                    if key not in seen_lane:
                        seen_lane.add(key)
                        lane_features.append(feature)

        tif_path = sample_out / "masked_reconstructed.tif"
        overlay_path = sample_out / "masked_reconstructed_overlay.png"
        out_profile = profile.copy()
        out_profile.update(count=3, dtype="uint8")
        with rasterio_open(tif_path, "w", **out_profile) as dst:
            dst.write(np.transpose(canvas, (2, 0, 1)))
        build_overlay_image(canvas=canvas, visual_lines=visual_lines).save(overlay_path)

        lane_geojson = build_feature_collection(lane_features, crs_name=crs_name, name="Lane")
        inter_geojson = build_feature_collection(inter_features, crs_name=crs_name, name="IntersectionBoundary")
        lane_path = sample_out / "Lane.geojson"
        inter_path = sample_out / "IntersectionBoundary.geojson"
        with lane_path.open("w", encoding="utf-8") as f:
            json.dump(lane_geojson, f, ensure_ascii=False, indent=2)
        with inter_path.open("w", encoding="utf-8") as f:
            json.dump(inter_geojson, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "source_sample_id": sample_id,
                "source_image_path": str(source_image_path),
                "patch_count": len(grouped[sample_id]),
                "lane_feature_count": len(lane_features),
                "intersection_boundary_count": len(inter_features),
                "tif_path": str(tif_path),
                "overlay_path": str(overlay_path),
                "lane_geojson_path": str(lane_path),
                "intersection_geojson_path": str(inter_path),
            }
        )
        print(
            f"[Verify] sample={sample_id} patches={len(grouped[sample_id])} "
            f"lane={len(lane_features)} inter={len(inter_features)} -> {sample_out}",
            flush=True,
        )

    summary_path = output_root / "reconstruct_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Verify] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
