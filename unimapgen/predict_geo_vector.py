import argparse
import glob
import os
import time

from unimapgen.geo.artifacts import export_prediction_tile_geojsons
from unimapgen.geo.errors import raise_geo_error, run_with_geo_error_boundary
from unimapgen.geo.io import geojson_dumps, pixel_features_to_geojson, save_text
from unimapgen.geo.inference import run_tiled_sample_prediction
from unimapgen.geo.pipeline import (
    build_geo_components,
    maybe_load_model_checkpoint,
    save_json,
    select_enabled_task_schemas,
)
from unimapgen.utils import ensure_dir, load_yaml, select_torch_device


def _resolve_inputs(cfg, args):
    records = []
    if args.input_image:
        records.append(
            {
                "sample_id": os.path.splitext(os.path.basename(args.input_image))[0],
                "image_path": args.input_image,
            }
        )
        return records
    if args.input_dir:
        pattern = os.path.join(args.input_dir, args.glob)
        for path in sorted(glob.glob(pattern)):
            records.append({"sample_id": os.path.splitext(os.path.basename(path))[0], "image_path": path})
        return records

    split = str(args.split or cfg["data"].get("test_split") or cfg["data"]["val_split"])
    split_root = os.path.join(str(cfg["data"]["dataset_root"]), split)
    if not os.path.isdir(split_root):
        fallback_split = str(cfg["data"]["val_split"])
        split = fallback_split
        split_root = os.path.join(str(cfg["data"]["dataset_root"]), split)
    if not os.path.isdir(split_root):
        raise_geo_error("GEO-1102", f"unable to resolve split directory for inference: {split_root}")
    for name in sorted(os.listdir(split_root)):
        sample_dir = os.path.join(split_root, name)
        if not os.path.isdir(sample_dir):
            continue
        image_path = os.path.join(sample_dir, str(cfg["data"]["image_relpath"]))
        if os.path.isfile(image_path):
            records.append({"sample_id": name, "image_path": image_path})
    return records


def _format_raw_text_dump(task_name: str, tile_records) -> str:
    lines = [f"task={task_name}", f"tile_count={len(tile_records)}", ""]
    for tile_record in tile_records:
        tile_index = int(tile_record.get("tile_index", 0))
        crop_bbox = tile_record.get("crop_bbox", [])
        pred_text = str(tile_record.get("pred_text", ""))
        lines.extend(
            [
                f"=== tile_index={tile_index} crop_bbox={crop_bbox} ===",
                pred_text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--glob", type=str, default="*.tif")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/geo_predictions")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=-1.0)
    parser.add_argument("--verbose_progress", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_schemas, text_tokenizer, _, model = build_geo_components(cfg)
    task_schemas = select_enabled_task_schemas(cfg=cfg, task_schemas=task_schemas)
    maybe_load_model_checkpoint(model, args.checkpoint)

    device = select_torch_device(prefer_cuda=True)
    model.to(device)
    model.eval()
    ensure_dir(args.output_dir)

    dec_cfg = cfg.get("decode", {})
    max_new_tokens = int(args.max_new_tokens or dec_cfg.get("max_new_tokens", 512))
    min_new_tokens = int(args.min_new_tokens or dec_cfg.get("min_new_tokens", 8))
    temperature = float(args.temperature if args.temperature > 0 else dec_cfg.get("temperature", 1.0))
    top_k = int(args.top_k or dec_cfg.get("top_k", 1))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty > 0 else dec_cfg.get("repetition_penalty", 1.05)
    )
    use_grammar_constraint = bool(dec_cfg.get("use_grammar_constraint", True))
    grammar_min_points = int(dec_cfg.get("grammar_min_points_per_line", 2))
    grammar_max_lines = dec_cfg.get("grammar_max_lines")

    records = _resolve_inputs(cfg, args)
    if int(args.max_samples) > 0:
        records = records[: int(args.max_samples)]

    summary = []
    total_t0 = time.time()
    for index, record in enumerate(records, start=1):
        sample_t0 = time.time()
        print(
            f"[Predict] sample {index}/{len(records)} id={record['sample_id']} image={record['image_path']}",
            flush=True,
        )
        sample_out_dir = os.path.join(args.output_dir, record["sample_id"])
        ensure_dir(sample_out_dir)
        sample_result = {"sample_id": record["sample_id"], "image_path": record["image_path"], "outputs": {}}
        pred_result = run_tiled_sample_prediction(
            cfg=cfg,
            image_path=record["image_path"],
            task_schemas=task_schemas,
            text_tokenizer=text_tokenizer,
            model=model,
            device=device,
            decode_cfg={
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "use_grammar_constraint": use_grammar_constraint,
                "grammar_min_points_per_line": grammar_min_points,
                "grammar_max_lines": grammar_max_lines,
            },
            stage="predict",
            progress_label=f"sample {index}/{len(records)} id={record['sample_id']}",
            log_progress=bool(args.verbose_progress),
        )
        export_prediction_tile_geojsons(
            cfg=cfg,
            sample_id=record["sample_id"],
            image_path=record["image_path"],
            pred_result=pred_result,
            output_dir=os.path.join(sample_out_dir, "artifacts"),
        )
        raster_meta = pred_result["raster_meta"]
        sample_result["parse_stats"] = pred_result.get("parse_stats", {})
        failed_tasks = []
        for task_name, task_schema in task_schemas.items():
            pred_original_features = pred_result["task_predictions"].get(task_name, [])
            raw_tile_outputs = pred_result.get("raw_outputs", {}).get(task_name, [])
            raw_output_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.raw_tiles.json")
            raw_text_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.raw_before_json.txt")
            save_json(raw_output_path, raw_tile_outputs)
            save_text(raw_text_path, _format_raw_text_dump(task_name=task_name, tile_records=raw_tile_outputs))
            parse_stats = pred_result.get("parse_stats", {}).get(task_name, {})
            if int(parse_stats.get("decoded_ok_count", 0)) == 0:
                failed_tasks.append(str(task_name))
                save_json(
                    os.path.join(sample_out_dir, f"{task_schema.collection_name}.parse_failed.json"),
                    {
                        "task_name": str(task_name),
                        "parse_stats": parse_stats,
                        "raw_tiles_path": raw_output_path,
                        "raw_text_path": raw_text_path,
                    },
                )
                sample_result.setdefault("debug_outputs", {})[task_name] = {
                    "raw_tiles": raw_output_path,
                    "raw_text": raw_text_path,
                }
                continue
            geojson_dict = pixel_features_to_geojson(
                task_schema=task_schema,
                feature_records=pred_original_features,
                raster_meta=raster_meta,
            )
            output_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.geojson")
            save_text(output_path, geojson_dumps(geojson_dict))
            sample_result["outputs"][task_name] = output_path
            sample_result.setdefault("debug_outputs", {})[task_name] = {
                "raw_tiles": raw_output_path,
                "raw_text": raw_text_path,
            }
        if failed_tasks:
            raise_geo_error(
                "GEO-1701",
                f"inference produced no decodable tasks={failed_tasks} for sample={record['sample_id']}",
            )
        summary.append(sample_result)
        print(
            f"[Predict] sample {index}/{len(records)} finished in {time.time() - sample_t0:.1f}s",
            flush=True,
        )

    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(
        f"saved {len(summary)} samples to {args.output_dir} "
        f"in {time.time() - total_t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1003")
