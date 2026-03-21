import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio import open as rasterio_open

from geo_current_dataset_v1_common import load_jsonl, uv_lines_to_local


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


def _origin_sort_key(point_xy: Iterable[float]) -> Tuple[float, float, float]:
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (x * x + y * y, y, x)


def annotate_patch_endpoint_order_labels(lines: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    endpoint_groups: Dict[str, List[Dict]] = defaultdict(list)
    for line_idx, row in enumerate(lines):
        copied = dict(row)
        out.append(copied)
        if str(copied.get("geometry_type", "line")) == "polygon":
            continue
        local_points = copied.get("local_points", copied.get("points", []))
        if not isinstance(local_points, list) or len(local_points) < 2:
            continue
        start_type = str(copied.get("start_type", "")).strip() or "start"
        end_type = str(copied.get("end_type", "")).strip() or "end"
        endpoint_groups[start_type].append(
            {
                "line_idx": int(line_idx),
                "label_key": "start_label",
                "point": local_points[0],
            }
        )
        endpoint_groups[end_type].append(
            {
                "line_idx": int(line_idx),
                "label_key": "end_label",
                "point": local_points[-1],
            }
        )
    for endpoint_type, refs in endpoint_groups.items():
        ordered = sorted(refs, key=lambda item: _origin_sort_key(item["point"]))
        for rank, ref in enumerate(ordered, start=1):
            out[int(ref["line_idx"])][str(ref["label_key"])] = f"{endpoint_type}{rank}"
    return out


def _line_color(category: str) -> Tuple[int, int, int]:
    if str(category) == "intersection_polygon":
        return (255, 180, 0)
    return (0, 255, 255)


def _line_color_compare(kind: str) -> Tuple[int, int, int]:
    if str(kind) == "float":
        return (80, 255, 80)
    return (255, 80, 255)


def build_overlay_image(
    canvas: np.ndarray,
    visual_lines: List[Dict],
    color_mode: str = "category",
    keep_boxes: List[Dict] | None = None,
) -> Image.Image:
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for keep_box in list(keep_boxes or []):
        x_min = int(round(float(keep_box.get("x_min", 0))))
        y_min = int(round(float(keep_box.get("y_min", 0))))
        x_max = int(round(float(keep_box.get("x_max", 0))))
        y_max = int(round(float(keep_box.get("y_max", 0))))
        label = str(keep_box.get("label", "keep_box"))
        draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 255, 255), width=2)
        draw.text((x_min + 4, max(0, y_min - 12)), label, fill=(255, 255, 255), font=font)

    for row in visual_lines:
        points = row.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        color = _line_color(str(row.get("category", ""))) if color_mode == "category" else _line_color_compare(str(row.get("viz_kind", "quantized")))
        xy = [(int(round(pt[0])), int(round(pt[1]))) for pt in points]
        if str(row.get("geometry_type", "line")) == "polygon" and len(xy) >= 3:
            draw.polygon(xy, outline=color, width=3)
            first_xy = xy[0]
            radius = 5
            draw.ellipse(
                (first_xy[0] - radius, first_xy[1] - radius, first_xy[0] + radius, first_xy[1] + radius),
                fill=(255, 180, 0),
                outline=(0, 0, 0),
            )
            label = "polygon" if color_mode == "category" else str(row.get("viz_kind", "polygon"))
            draw.text((first_xy[0] + 6, first_xy[1] - 10), label, fill=color, font=font)
            continue
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
        if color_mode == "category":
            start_label = str(row.get("start_label", start_type))
            end_label = str(row.get("end_label", end_type))
            draw.text((start_xy[0] + 6, start_xy[1] - 10), start_label, fill=(0, 255, 0), font=font)
            draw.text((end_xy[0] + 6, end_xy[1] - 10), end_label, fill=(255, 80, 80), font=font)
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
        visual_lines_float: List[Dict] = []
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

            export_lines = row.get("target_lines_float", [])
            if not export_lines:
                export_lines = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in export_lines:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[float(p[0]) + x0, float(p[1]) + y0] for p in points]
                key = (
                    str(line.get("category", "")),
                    str(line.get("geometry_type", "line")),
                    str(line.get("start_type", "")),
                    str(line.get("end_type", "")),
                    tuple((round(float(p[0]), 3), round(float(p[1]), 3)) for p in global_points),
                )
                geometry_type = str(line.get("geometry_type", "line"))
                if str(line.get("category", "")) == "intersection_polygon":
                    ring_world = pixel_to_world(global_points, transform)
                    if len(ring_world) >= 3 and ring_world[0] != ring_world[-1]:
                        ring_world.append(list(ring_world[0]))
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [ring_world],
                        },
                    }
                else:
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "category": str(line.get("category", "")),
                            "start_type": str(line.get("start_type", "")),
                            "end_type": str(line.get("end_type", "")),
                            "geometry_type": geometry_type,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": pixel_to_world(global_points, transform),
                        },
                    }
                if str(line.get("category", "")) == "intersection_polygon":
                    if key not in seen_inter:
                        seen_inter.add(key)
                        inter_features.append(feature)
                else:
                    if key not in seen_lane:
                        seen_lane.add(key)
                        lane_features.append(feature)

            patch_visual_lines: List[Dict] = []
            visual_quantized_source = row.get("target_lines_quantized", [])
            if not visual_quantized_source:
                visual_quantized_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in visual_quantized_source:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[int(p[0]) + x0, int(p[1]) + y0] for p in points]
                patch_visual_lines.append(
                    {
                        "category": str(line.get("category", "")),
                        "geometry_type": str(line.get("geometry_type", "line")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                        "points": global_points,
                        "local_points": [[float(p[0]), float(p[1])] for p in points],
                        "viz_kind": "quantized",
                    }
                )
            visual_lines.extend(annotate_patch_endpoint_order_labels(patch_visual_lines))

            patch_visual_lines_float: List[Dict] = []
            float_source = row.get("target_lines_float", [])
            if not float_source:
                float_source = uv_lines_to_local(row.get("target_lines", []), patch=row)
            for line in float_source:
                points = line.get("points", [])
                if not isinstance(points, list) or len(points) < 2:
                    continue
                global_points = [[float(p[0]) + x0, float(p[1]) + y0] for p in points]
                patch_visual_lines_float.append(
                    {
                        "category": str(line.get("category", "")),
                        "geometry_type": str(line.get("geometry_type", "line")),
                        "start_type": str(line.get("start_type", "")),
                        "end_type": str(line.get("end_type", "")),
                        "points": global_points,
                        "local_points": [[float(p[0]), float(p[1])] for p in points],
                        "viz_kind": "float",
                    }
                )
            visual_lines_float.extend(annotate_patch_endpoint_order_labels(patch_visual_lines_float))

        tif_path = sample_out / "masked_reconstructed.tif"
        overlay_path = sample_out / "masked_reconstructed_overlay.png"
        overlay_float_path = sample_out / "masked_reconstructed_overlay_float.png"
        overlay_compare_path = sample_out / "masked_reconstructed_overlay_compare.png"
        out_profile = profile.copy()
        out_profile.update(count=3, dtype="uint8")
        with rasterio_open(tif_path, "w", **out_profile) as dst:
            dst.write(np.transpose(canvas, (2, 0, 1)))
        build_overlay_image(canvas=canvas, visual_lines=visual_lines).save(overlay_path)
        build_overlay_image(canvas=canvas, visual_lines=visual_lines_float, color_mode="compare").save(overlay_float_path)
        build_overlay_image(canvas=canvas, visual_lines=[*visual_lines_float, *visual_lines], color_mode="compare").save(overlay_compare_path)

        lane_geojson = build_feature_collection(lane_features, crs_name=crs_name, name="Lane")
        inter_geojson = build_feature_collection(inter_features, crs_name=crs_name, name="Intersection")
        lane_path = sample_out / "Lane.geojson"
        inter_path = sample_out / "Intersection.geojson"
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
                "intersection_polygon_count": len(inter_features),
                "tif_path": str(tif_path),
                "overlay_path": str(overlay_path),
                "overlay_float_path": str(overlay_float_path),
                "overlay_compare_path": str(overlay_compare_path),
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
