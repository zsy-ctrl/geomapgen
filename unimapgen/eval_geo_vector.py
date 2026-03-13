import argparse
import math
import os

from tqdm import tqdm

from unimapgen.geo.artifacts import export_eval_sample_geojsons
from unimapgen.geo.errors import raise_geo_error, run_with_geo_error_boundary
from unimapgen.geo.inference import run_tiled_sample_prediction
from unimapgen.geo.io import geojson_to_pixel_features, load_geojson, read_binary_mask
from unimapgen.geo.metrics import (
    evaluate_intersection_predictions,
    evaluate_lane_predictions,
    filter_features_by_review_mask,
)
from unimapgen.geo.pipeline import (
    build_geo_components,
    maybe_load_model_checkpoint,
    save_json,
    select_enabled_task_schemas,
)
from unimapgen.utils import load_yaml, select_torch_device


def _mean_metrics(records):
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    out = {}
    for key in keys:
        values = []
        for record in records:
            if key not in record:
                continue
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def _collect_samples(cfg, split: str, max_samples: int):
    split_root = os.path.join(str(cfg["data"]["dataset_root"]), split)
    if not os.path.isdir(split_root):
        raise_geo_error("GEO-1101", f"dataset split not found: {split_root}")
    out = []
    for name in sorted(os.listdir(split_root)):
        sample_dir = os.path.join(split_root, name)
        if not os.path.isdir(sample_dir):
            continue
        image_path = os.path.join(sample_dir, str(cfg["data"]["image_relpath"]))
        if not os.path.isfile(image_path):
            continue
        out.append({"sample_id": name, "sample_dir": sample_dir, "image_path": image_path})
    if int(max_samples) > 0:
        out = out[: int(max_samples)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--output", type=str, default="outputs/geo_eval_metrics.json")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    split = str(args.split or cfg["data"]["val_split"])
    task_schemas, text_tokenizer, _, model = build_geo_components(cfg)
    task_schemas = select_enabled_task_schemas(cfg=cfg, task_schemas=task_schemas)
    maybe_load_model_checkpoint(model, args.checkpoint)
    device = select_torch_device(prefer_cuda=True)
    model.to(device)
    model.eval()

    samples = _collect_samples(cfg=cfg, split=split, max_samples=int(args.max_samples))
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})
    lane_records = []
    intersection_records = []
    sample_metrics = []
    parse_rates = []
    eval_artifact_root = os.path.splitext(args.output)[0] + "_artifacts"

    for sample in tqdm(samples, desc=f"Eval {split}", leave=False):
        pred_result = run_tiled_sample_prediction(
            cfg=cfg,
            image_path=sample["image_path"],
            task_schemas=task_schemas,
            text_tokenizer=text_tokenizer,
            model=model,
            device=device,
            decode_cfg=cfg.get("decode", {}),
            stage="eval",
        )
        raster_meta = pred_result["raster_meta"]
        parse_stats = pred_result["parse_stats"]
        review_mask_path = str(cfg["data"].get("review_mask_relpath", "")).strip()
        review_mask_full_path = os.path.join(sample["sample_dir"], review_mask_path) if review_mask_path else ""
        review_mask = None
        if review_mask_full_path and os.path.isfile(review_mask_full_path):
            review_mask = read_binary_mask(
                path=review_mask_full_path,
                threshold=int(cfg["data"].get("mask_threshold", 127)),
            )

        gt_by_task = {}
        for task_name, task_schema in task_schemas.items():
            label_relpath = str(cfg["data"]["label_relpaths"][task_name])
            label_path = os.path.join(sample["sample_dir"], label_relpath)
            if not os.path.isfile(label_path):
                continue
            gt_features = geojson_to_pixel_features(
                geojson_dict=load_geojson(label_path),
                task_schema=task_schema,
                raster_meta=raster_meta,
            )
            pred_features = pred_result["task_predictions"].get(task_name, [])
            if review_mask is not None and bool(data_cfg.get("eval_filter_features_by_review_mask", False)):
                gt_features = filter_features_by_review_mask(
                    feature_records=gt_features,
                    mask=review_mask,
                    min_inside_ratio=float(eval_cfg.get("review_mask_min_inside_ratio", 0.5)),
                )
                pred_features = filter_features_by_review_mask(
                    feature_records=pred_features,
                    mask=review_mask,
                    min_inside_ratio=float(eval_cfg.get("review_mask_min_inside_ratio", 0.5)),
                )
            gt_by_task[task_name] = gt_features
            parse_rate = 0.0
            if parse_stats.get(task_name, {}).get("tile_count", 0) > 0:
                parse_rate = float(parse_stats[task_name]["decoded_ok_count"]) / float(parse_stats[task_name]["tile_count"])
            parse_rates.append(parse_rate)

            if task_name == "lane":
                metrics = evaluate_lane_predictions(
                    gt_features=gt_features,
                    pred_features=pred_features,
                    raster_meta=raster_meta,
                    task_schema=task_schema,
                    distance_threshold_m=float(eval_cfg.get("lane_match_distance_m", 2.0)),
                )
                lane_records.append(metrics)
            else:
                metrics = evaluate_intersection_predictions(
                    gt_features=gt_features,
                    pred_features=pred_features,
                    raster_meta=raster_meta,
                    task_schema=task_schema,
                    iou_threshold=float(eval_cfg.get("intersection_iou_threshold", 0.3)),
                )
                intersection_records.append(metrics)
            metrics["sample_id"] = sample["sample_id"]
            metrics["task_name"] = task_name
            metrics["parse_ok_rate"] = parse_rate
            sample_metrics.append(metrics)
        export_eval_sample_geojsons(
            cfg=cfg,
            sample_id=sample["sample_id"],
            image_path=sample["image_path"],
            pred_result=pred_result,
            task_schemas=task_schemas,
            gt_by_task=gt_by_task,
            output_dir=os.path.join(eval_artifact_root, sample["sample_id"]),
        )

    output = {
        "split": split,
        "samples": len(samples),
        "task_records": len(sample_metrics),
        "parse_success_rate": float(sum(parse_rates) / len(parse_rates)) if parse_rates else 0.0,
        "lane_metrics": _mean_metrics(lane_records),
        "intersection_metrics": _mean_metrics(intersection_records),
        "sample_metrics": sample_metrics,
    }
    save_json(args.output, output)
    print(f"saved eval metrics to {args.output}")


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1004")
