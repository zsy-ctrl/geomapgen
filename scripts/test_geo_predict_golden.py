import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unimapgen.geo.artifacts import export_prediction_tile_geojsons
from unimapgen.geo.coord_sequence import points_abs_to_uv, rings_abs_to_uv, uv_feature_records_to_target_items
from unimapgen.geo.geometry import build_resize_context
from unimapgen.geo.inference import run_tiled_sample_prediction
from unimapgen.geo.io import geojson_dumps, geojson_to_pixel_features, load_geojson, pixel_features_to_geojson, read_raster_meta, save_text
from unimapgen.geo.pipeline import load_geo_task_schemas, save_json, select_enabled_task_schemas
from unimapgen.geo.tokenizer import GeoCoordTokenizer
from unimapgen.utils import ensure_dir, load_yaml


class _DummySatEncoder:
    def __init__(self, out_hw: Tuple[int, int]) -> None:
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))


class GoldenReplayModel:
    def __init__(self, token_batches: Dict[str, List[int]], out_hw: Tuple[int, int]) -> None:
        self._token_batches = {str(k): [int(x) for x in v] for k, v in token_batches.items()}
        self._call_index = 0
        self.sat_encoder = _DummySatEncoder(out_hw=out_hw)

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(
        self,
        image,
        prompt_input_ids,
        prompt_attention_mask,
        pv_images=None,
        state_input_ids=None,
        state_attention_mask=None,
        max_new_tokens=0,
        min_new_tokens=0,
        temperature=1.0,
        top_k=1,
        repetition_penalty=1.0,
        grammar_helper=None,
        use_kv_cache=True,
        return_token_meta=False,
    ):
        if self._call_index >= len(self._token_batches):
            raise RuntimeError(f"GoldenReplayModel exhausted token batches at call_index={self._call_index}")
        task_name = list(self._token_batches.keys())[self._call_index]
        token_ids = self._token_batches[task_name]
        self._call_index += 1
        tensor = torch.tensor(token_ids, dtype=torch.long, device=image.device).unsqueeze(0)
        if return_token_meta:
            return tensor, [[]]
        return tensor


def _clip_points(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).copy()
    if pts.size == 0:
        return pts.reshape(0, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(0, width - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(0, height - 1)))
    return pts.astype(np.float32)


def _build_grid_offsets(count: int, step_x: float, step_y: float, cols: int) -> List[Tuple[float, float]]:
    total = max(1, int(count))
    num_cols = max(1, int(cols))
    offsets: List[Tuple[float, float]] = []
    for index in range(total):
        row = index // num_cols
        col = index % num_cols
        offsets.append((float(col) * float(step_x), float(row) * float(step_y)))
    return offsets


def _duplicate_feature_records(
    feature_records: Sequence[Dict],
    width: int,
    height: int,
    offsets: Sequence[Tuple[float, float]],
    task_name: str,
) -> List[Dict]:
    if not feature_records:
        return []
    base = feature_records[0]
    out: List[Dict] = []
    for dup_index, (dx, dy) in enumerate(offsets):
        rec = copy.deepcopy(base)
        pts = np.asarray(rec.get("points", []), dtype=np.float32)
        shifted = pts.copy()
        shifted[:, 0] += float(dx)
        shifted[:, 1] += float(dy)
        rec["points"] = _clip_points(shifted, width=width, height=height)
        if rec.get("rings"):
            shifted_rings = []
            for ring in rec.get("rings", []):
                ring_pts = np.asarray(ring, dtype=np.float32)
                ring_shifted = ring_pts.copy()
                ring_shifted[:, 0] += float(dx)
                ring_shifted[:, 1] += float(dy)
                shifted_rings.append(_clip_points(ring_shifted, width=width, height=height))
            rec["rings"] = shifted_rings
            if shifted_rings:
                rec["points"] = shifted_rings[0]
        props = dict(rec.get("properties", {}))
        props["Id"] = f"{props.get('Id', task_name)}_{task_name}_{dup_index + 1:03d}"
        rec["properties"] = props
        out.append(rec)
    return out


def _feature_records_to_uv(feature_records: Sequence[Dict], resize_ctx) -> List[Dict]:
    out = []
    for feature in feature_records:
        record = {
            "properties": dict(feature.get("properties", {})),
            "points": points_abs_to_uv(np.asarray(feature.get("points", []), dtype=np.float32), resize_ctx=resize_ctx),
        }
        if feature.get("rings"):
            record["rings"] = rings_abs_to_uv(feature.get("rings", []), resize_ctx=resize_ctx)
        out.append(record)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden predict replay for multi-lane and multi-intersection output.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample_id", default="")
    parser.add_argument("--lane_count", type=int, default=6)
    parser.add_argument("--intersection_count", type=int, default=4)
    parser.add_argument("--offset_step_x", type=float, default=18.0)
    parser.add_argument("--offset_step_y", type=float, default=18.0)
    parser.add_argument("--grid_cols", type=int, default=3)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task_schemas = select_enabled_task_schemas(cfg=cfg, task_schemas=load_geo_task_schemas(cfg))
    text_tokenizer = GeoCoordTokenizer(
        qwen_model_path=str(cfg["model"]["qwen_model_path"]),
        local_files_only=bool(cfg["model"].get("local_files_only", True)),
        trust_remote_code=True,
        coord_bins=int(cfg.get("serialization", {}).get("coord_bins", 1024)),
    )

    split_root = os.path.join(str(cfg["data"]["dataset_root"]), str(args.split))
    sample_id = str(args.sample_id).strip()
    if not sample_id:
        sample_id = sorted(name for name in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, name)))[0]
    sample_dir = os.path.join(split_root, sample_id)
    image_path = os.path.join(sample_dir, str(cfg["data"]["image_relpath"]))
    raster_meta = read_raster_meta(image_path)
    resize_ctx = build_resize_context(
        width=int(raster_meta.width),
        height=int(raster_meta.height),
        target_size=int(cfg["data"]["image_size"]),
        crop_bbox=None,
    )

    offsets_by_task = {
        "lane": _build_grid_offsets(int(args.lane_count), float(args.offset_step_x), float(args.offset_step_y), int(args.grid_cols)),
        "intersection": _build_grid_offsets(int(args.intersection_count), float(args.offset_step_x), float(args.offset_step_y), int(args.grid_cols)),
    }
    synthetic_by_task: Dict[str, List[Dict]] = {}
    token_batches: Dict[str, List[int]] = {}
    for task_name, task_schema in task_schemas.items():
        label_path = os.path.join(sample_dir, str(cfg["data"]["label_relpaths"][task_name]))
        base_features = geojson_to_pixel_features(
            geojson_dict=load_geojson(label_path),
            task_schema=task_schema,
            raster_meta=raster_meta,
        )
        synthetic_records = _duplicate_feature_records(
            feature_records=base_features,
            width=int(raster_meta.width),
            height=int(raster_meta.height),
            offsets=offsets_by_task[task_name],
            task_name=task_name,
        )
        synthetic_by_task[task_name] = synthetic_records
        target_items = uv_feature_records_to_target_items(
            feature_records=_feature_records_to_uv(synthetic_records, resize_ctx=resize_ctx),
            task_schema=task_schema,
            image_size=int(cfg["data"]["image_size"]),
        )
        token_batches[task_name] = text_tokenizer.encode_map_items(
            map_items=target_items,
            image_size=int(cfg["data"]["image_size"]),
            max_length=None,
            append_eos=True,
        )

    model = GoldenReplayModel(
        token_batches=token_batches,
        out_hw=tuple(cfg["model"].get("sat_token_hw", [4, 4])),
    )
    device = torch.device("cpu")
    ensure_dir(args.output_dir)

    pred_result = run_tiled_sample_prediction(
        cfg=cfg,
        image_path=image_path,
        task_schemas=task_schemas,
        text_tokenizer=text_tokenizer,
        model=model,
        device=device,
        decode_cfg=cfg.get("decode", {}),
        stage="predict",
        progress_label=f"golden sample={sample_id}",
        log_progress=True,
    )

    sample_out_dir = os.path.join(str(args.output_dir), sample_id)
    ensure_dir(sample_out_dir)
    export_prediction_tile_geojsons(
        cfg=cfg,
        sample_id=sample_id,
        image_path=image_path,
        pred_result=pred_result,
        output_dir=os.path.join(sample_out_dir, "artifacts"),
    )

    summary = {
        "sample_id": sample_id,
        "image_path": image_path,
        "generated_at": time.time(),
        "tasks": {},
    }
    for task_name, task_schema in task_schemas.items():
        pred_features = pred_result["task_predictions"].get(task_name, [])
        output_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.geojson")
        save_text(
            output_path,
            geojson_dumps(
                pixel_features_to_geojson(
                    task_schema=task_schema,
                    feature_records=pred_features,
                    raster_meta=raster_meta,
                )
            ),
        )
        golden_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.golden_input.geojson")
        save_text(
            golden_path,
            geojson_dumps(
                pixel_features_to_geojson(
                    task_schema=task_schema,
                    feature_records=synthetic_by_task[task_name],
                    raster_meta=raster_meta,
                )
            ),
        )
        raw_tiles_path = os.path.join(sample_out_dir, f"{task_schema.collection_name}.raw_tiles.json")
        save_json(raw_tiles_path, pred_result.get("raw_outputs", {}).get(task_name, []))
        summary["tasks"][task_name] = {
            "golden_feature_count": int(len(synthetic_by_task[task_name])),
            "pred_feature_count": int(len(pred_features)),
            "output_path": output_path,
            "golden_path": golden_path,
            "raw_tiles_path": raw_tiles_path,
        }

    save_json(os.path.join(str(args.output_dir), "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
