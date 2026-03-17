import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Subset
from tqdm import tqdm

from unimapgen.geo.artifacts import (
    export_batch_geojson_snapshots,
    export_tile_audit_records,
    get_artifact_export_cfg,
    save_geojson_snapshot,
    save_json,
)
from unimapgen.geo.errors import run_with_geo_error_boundary, wrap_geo_error
from unimapgen.geo.io import assign_incremental_feature_ids, geojson_dumps, pixel_features_to_geojson, pixel_features_to_uv_geojson
from unimapgen.geo.geometry import build_resize_context
from unimapgen.geo.metrics import deduplicate_feature_records
from unimapgen.geo.pipeline import (
    atomic_torch_save,
    build_checkpoint_obj,
    build_geo_components,
    build_geo_dataset,
    compute_shift_metrics,
    make_output_dir,
    maybe_load_model_checkpoint,
    maybe_resume_training_state,
)
from unimapgen.utils import cosine_lr, load_yaml, select_torch_device, set_seed


def _build_sample_batches(items, batch_size: int, task_order: dict[str, int], drop_last: bool = False) -> list[list[int]]:
    batch_size = max(1, int(batch_size))
    drop_last = bool(drop_last)
    grouped: OrderedDict[str, list[tuple[int, dict]]] = OrderedDict()
    for index, item in enumerate(items):
        sample_id = str(item.get("sample_id", ""))
        grouped.setdefault(sample_id, []).append((int(index), item))

    batches: list[list[int]] = []
    for _, entries in grouped.items():
        entries.sort(
            key=lambda pair: (
                int(pair[1].get("tile_index", 0)),
                int(task_order.get(str(pair[1].get("task_name", "")), 10**6)),
                int(pair[0]),
            )
        )
        indices = [int(index) for index, _ in entries]
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            batches.append(batch)
    return batches


class SampleSequentialBatchSampler(BatchSampler):
    def __init__(self, items, batch_size: int, task_order: dict[str, int], drop_last: bool = False) -> None:
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self._batches = _build_sample_batches(
            items=items,
            batch_size=self.batch_size,
            task_order=task_order,
            drop_last=self.drop_last,
        )

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class PrecomputedBatchSampler(BatchSampler):
    def __init__(self, batches: list[list[int]]) -> None:
        self._batches = [list(map(int, batch)) for batch in batches]

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def _count_sample_batches(items, batch_size: int) -> dict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for item in items:
        sample_id = str(item.get("sample_id", ""))
        counts[sample_id] = int(counts.get(sample_id, 0)) + 1
    out: dict[str, int] = {}
    batch_size = max(1, int(batch_size))
    for sample_id, item_count in counts.items():
        out[sample_id] = max(1, (int(item_count) + batch_size - 1) // batch_size)
    return out


def _resolve_tile_audit_records(dataset_obj) -> list[dict]:
    records = list(getattr(dataset_obj, "tile_audit_records", []) or [])
    if records:
        return records

    items = list(getattr(dataset_obj, "items", []) or [])
    if not items:
        return []

    fallback_records: list[dict] = []
    seen_keys: set[tuple[str, tuple[int, int, int, int]]] = set()
    for item in items:
        sample_id = str(item.get("sample_id", "sample"))
        sample_dir = str(item.get("sample_dir", ""))
        image_path = str(item.get("image_path", ""))
        tile_index = int(item.get("tile_index", 0))
        tile_window = item.get("tile_window")
        crop_bbox = item.get("crop_bbox")
        if isinstance(tile_window, dict):
            bbox = (
                int(tile_window["x0"]),
                int(tile_window["y0"]),
                int(tile_window["x1"]),
                int(tile_window["y1"]),
            )
            keep_bbox = (
                int(tile_window.get("keep_x0", tile_window["x0"])),
                int(tile_window.get("keep_y0", tile_window["y0"])),
                int(tile_window.get("keep_x1", tile_window["x1"])),
                int(tile_window.get("keep_y1", tile_window["y1"])),
            )
            mask_ratio = float(tile_window.get("mask_ratio", 0.0))
            mask_pixels = int(tile_window.get("mask_pixels", 0))
        elif crop_bbox is not None:
            bbox = tuple(int(v) for v in crop_bbox)
            keep_bbox = bbox
            mask_ratio = 0.0
            mask_pixels = 0
        else:
            continue

        dedupe_key = (sample_id, bbox)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        fallback_records.append(
            {
                "stage": str(getattr(getattr(dataset_obj, "cfg", None), "stage", "")),
                "split": str(getattr(getattr(dataset_obj, "cfg", None), "split", "")),
                "sample_id": sample_id,
                "sample_dir": sample_dir,
                "image_path": image_path,
                "crop_bbox": None if crop_bbox is None else [int(v) for v in crop_bbox],
                "candidate_index": int(tile_index),
                "selected": True,
                "reason": "dataset_item_fallback",
                "bbox": [int(v) for v in bbox],
                "keep_bbox": [int(v) for v in keep_bbox],
                "mask_ratio": float(mask_ratio),
                "mask_pixels": int(mask_pixels),
            }
        )
    return fallback_records


def _group_sample_indices(items, task_order: dict[str, int]) -> OrderedDict[str, list[int]]:
    grouped: OrderedDict[str, list[tuple[int, dict]]] = OrderedDict()
    for index, item in enumerate(items):
        sample_id = str(item.get("sample_id", ""))
        grouped.setdefault(sample_id, []).append((int(index), item))
    ordered: OrderedDict[str, list[int]] = OrderedDict()
    for sample_id, entries in grouped.items():
        entries.sort(
            key=lambda pair: (
                int(pair[1].get("tile_index", 0)),
                int(task_order.get(str(pair[1].get("task_name", "")), 10**6)),
                int(pair[0]),
            )
        )
        ordered[sample_id] = [int(index) for index, _ in entries]
    return ordered


def _build_subset_loader(dataset, indices: list[int], batch_size: int, num_workers: int, collate_fn):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(int(num_workers) > 0),
        collate_fn=collate_fn,
    )


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _init_distributed_training(prefer_cuda: bool = True) -> dict:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = bool(world_size > 1)
    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = select_torch_device(prefer_cuda=prefer_cuda)
    return {
        "enabled": enabled,
        "world_size": int(world_size),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "device": device,
    }


def _destroy_distributed_training(dist_state: dict) -> None:
    if bool(dist_state.get("enabled", False)) and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _is_main_process(dist_state: dict) -> bool:
    return int(dist_state.get("rank", 0)) == 0


def _maybe_barrier(dist_state: dict) -> None:
    if bool(dist_state.get("enabled", False)) and dist.is_initialized():
        dist.barrier()


def _broadcast_object(value, dist_state: dict):
    if not bool(dist_state.get("enabled", False)):
        return value
    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _reduce_sum(value: float, device: torch.device, dist_state: dict) -> float:
    if not bool(dist_state.get("enabled", False)):
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _assign_samples_to_ranks(
    sample_ids_in_order: list[str],
    sample_batch_counts: dict[str, int],
    world_size: int,
) -> tuple[list[list[str]], list[int], int]:
    if int(world_size) <= 1:
        total = sum(int(sample_batch_counts.get(sample_id, 1)) for sample_id in sample_ids_in_order)
        return [list(sample_ids_in_order)], [int(total)], int(total)

    positions = {sample_id: idx for idx, sample_id in enumerate(sample_ids_in_order)}
    assigned: list[list[str]] = [[] for _ in range(int(world_size))]
    loads = [0 for _ in range(int(world_size))]
    sorted_ids = sorted(
        sample_ids_in_order,
        key=lambda sample_id: (-int(sample_batch_counts.get(sample_id, 1)), int(positions[sample_id])),
    )
    for sample_id in sorted_ids:
        target_rank = min(range(int(world_size)), key=lambda idx: (int(loads[idx]), len(assigned[idx]), int(idx)))
        assigned[target_rank].append(sample_id)
        loads[target_rank] += int(sample_batch_counts.get(sample_id, 1))

    if sample_ids_in_order:
        for rank_idx in range(int(world_size)):
            if assigned[rank_idx]:
                continue
            sample_id = sample_ids_in_order[rank_idx % len(sample_ids_in_order)]
            assigned[rank_idx].append(sample_id)
            loads[rank_idx] += int(sample_batch_counts.get(sample_id, 1))

    for rank_idx in range(int(world_size)):
        assigned[rank_idx].sort(key=lambda sample_id: int(positions[sample_id]))

    max_batches = max(loads) if loads else 0
    return assigned, loads, int(max_batches)


def _pad_batches_to_length(batches: list[list[int]], target_len: int) -> list[list[int]]:
    padded = [list(batch) for batch in batches]
    if int(target_len) <= len(padded):
        return padded
    if not padded:
        raise RuntimeError("Cannot pad empty batch list to distributed target length.")
    cursor = 0
    while len(padded) < int(target_len):
        padded.append(list(padded[cursor % len(batches)]))
        cursor += 1
    return padded


def _build_rank_subset_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    collate_fn,
    sample_patch_sequential: bool,
    task_order: dict[str, int],
    target_num_batches: int | None = None,
):
    subset = Subset(dataset, indices)
    if bool(sample_patch_sequential):
        subset_items = [getattr(dataset, "items", [])[idx] for idx in indices]
        batch_lists = _build_sample_batches(
            items=subset_items,
            batch_size=batch_size,
            task_order=task_order,
            drop_last=False,
        )
        if target_num_batches is not None:
            batch_lists = _pad_batches_to_length(batch_lists=batch_lists, target_len=int(target_num_batches))
        return DataLoader(
            subset,
            batch_sampler=PrecomputedBatchSampler(batch_lists),
            num_workers=int(num_workers),
            pin_memory=True,
            persistent_workers=bool(int(num_workers) > 0),
            collate_fn=collate_fn,
        )
    return DataLoader(
        subset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(int(num_workers) > 0),
        collate_fn=collate_fn,
    )


def _cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _estimate_memory_risk(cfg: dict, batch_size: int) -> dict:
    data_cfg = cfg.get("data", {})
    text_cfg = cfg.get("text", {})
    model_cfg = cfg.get("model", {})
    state_cfg = cfg.get("state_update", {})
    task_cfg = cfg.get("serialization", {}).get("tasks", {})
    train_cfg = cfg.get("train", {})

    image_size = int(data_cfg.get("image_size", 0) or 0)
    prompt_max = int(text_cfg.get("prompt_max_tokens", 0) or 0)
    state_max = int(text_cfg.get("state_max_tokens", 0) or 0)
    target_max = int(text_cfg.get("target_max_tokens", 0) or 0)
    lane_max = int(task_cfg.get("lane", {}).get("max_features", 0) or 0)
    intersection_max = int(task_cfg.get("intersection", {}).get("max_features", 0) or 0)
    state_max_features = int(state_cfg.get("max_features", 0) or 0)
    amp_enabled = _cfg_bool(train_cfg.get("amp", False), default=False)
    grad_ckpt = _cfg_bool(model_cfg.get("gradient_checkpointing", False), default=False)
    llm_mode = str(model_cfg.get("llm_train_mode", "full")).strip().lower()

    score = 0
    reasons = []
    suggestions = []

    if llm_mode == "full":
        score += 5
        reasons.append("llm_train_mode=full")
        suggestions.append("switch to LoRA if possible")
    elif llm_mode == "lora":
        score += 1
        reasons.append("llm_train_mode=lora")

    if image_size >= 448:
        score += 3
        reasons.append(f"image_size={image_size}")
        suggestions.append("reduce image_size to 336 or 288")
    elif image_size >= 384:
        score += 2
        reasons.append(f"image_size={image_size}")
    elif image_size >= 336:
        score += 1
        reasons.append(f"image_size={image_size}")

    if target_max == 0:
        score += 3
        reasons.append("target_max_tokens=unlimited")
        suggestions.append("set target_max_tokens to 1024 or lower")
    elif target_max > 1024:
        score += 2
        reasons.append(f"target_max_tokens={target_max}")

    if state_max == 0:
        score += 2
        reasons.append("state_max_tokens=unlimited")
        suggestions.append("set state_max_tokens to 512 or lower")
    elif state_max > 512:
        score += 1
        reasons.append(f"state_max_tokens={state_max}")

    if prompt_max == 0:
        score += 1
        reasons.append("prompt_max_tokens=unlimited")

    if lane_max == 0:
        score += 1
        reasons.append("lane.max_features=unlimited")
        suggestions.append("cap lane.max_features")
    elif lane_max > 64:
        score += 1
        reasons.append(f"lane.max_features={lane_max}")

    if intersection_max == 0:
        score += 1
        reasons.append("intersection.max_features=unlimited")
        suggestions.append("cap intersection.max_features")
    elif intersection_max > 32:
        score += 1
        reasons.append(f"intersection.max_features={intersection_max}")

    if state_max_features > 16:
        score += 1
        reasons.append(f"state_update.max_features={state_max_features}")

    if batch_size > 1:
        score += max(2, batch_size)
        reasons.append(f"batch_size={batch_size}")
        suggestions.append("keep batch_size at 1")

    if not amp_enabled:
        score += 2
        reasons.append("amp=false")
        suggestions.append("enable amp")

    if not grad_ckpt:
        score += 2
        reasons.append("gradient_checkpointing=false")
        suggestions.append("enable gradient checkpointing")

    level = "low"
    if score >= 10:
        level = "high"
    elif score >= 6:
        level = "medium"

    uniq_suggestions = []
    for item in suggestions:
        if item not in uniq_suggestions:
            uniq_suggestions.append(item)

    return {
        "level": level,
        "score": int(score),
        "reasons": reasons,
        "suggestions": uniq_suggestions,
        "llm_mode": llm_mode,
    }


def run_val(
    model,
    loader,
    device,
    desc: str = "",
    cfg: dict | None = None,
    task_schemas: dict | None = None,
    text_tokenizer=None,
    out_dir: str = "",
    epoch: int = 0,
    predict_model=None,
):
    model.eval()
    predict_model = _unwrap_model(predict_model if predict_model is not None else model)
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    total_tok = 0
    artifact_cfg = get_artifact_export_cfg(cfg or {})
    exported_batches = 0
    iterator = loader
    if desc:
        iterator = tqdm(loader, desc=desc, leave=False)
    use_amp = bool((cfg or {}).get("train", {}).get("amp", False)) and device.type == "cuda"
    max_batches_cfg = int(artifact_cfg["max_batches_per_epoch"])
    with torch.inference_mode():
        for batch_index, batch in enumerate(iterator):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    image=batch["image"].to(device),
                    prompt_input_ids=batch["prompt_input_ids"].to(device),
                    prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                    pv_images=None,
                    state_input_ids=batch["state_input_ids"].to(device),
                    state_attention_mask=batch["state_attention_mask"].to(device),
                    map_input_ids=batch["map_input_ids"].to(device),
                    map_attention_mask=batch["map_attention_mask"].to(device),
                    return_logits=True,
                )
            total_loss += float(out["loss"].item()) * batch["image"].shape[0]
            total_count += batch["image"].shape[0]
            correct, total = compute_shift_metrics(out["logits"], out["labels"])
            total_correct += correct
            total_tok += total
            if (
                cfg is not None
                and task_schemas is not None
                and text_tokenizer is not None
                and bool(artifact_cfg["enabled"])
                and bool(artifact_cfg["save_val_batch_geojson"])
                and (max_batches_cfg <= 0 or exported_batches < max_batches_cfg)
            ):
                export_batch_geojson_snapshots(
                    cfg=cfg,
                    task_schemas=task_schemas,
                    text_tokenizer=text_tokenizer,
                    model=predict_model,
                    batch=batch,
                    device=device,
                    decode_cfg=cfg.get("decode", {}),
                    output_dir=out_dir,
                    stage="val",
                    epoch=int(epoch),
                    batch_index=int(batch_index),
                )
                exported_batches += 1
    return total_loss / max(total_count, 1), float(total_correct) / float(max(total_tok, 1))


def run_training(config_path: str, mode_override: str = "") -> None:
    dist_state = _init_distributed_training(prefer_cuda=True)
    is_main_process = _is_main_process(dist_state)
    cfg = load_yaml(config_path)
    if mode_override:
        cfg.setdefault("model", {})
        cfg["model"]["llm_train_mode"] = str(mode_override)

    set_seed(int(cfg["seed"]))

    train_cfg = cfg["train"]
    resume_checkpoint = str(train_cfg.get("init_checkpoint", "")).strip()
    if bool(dist_state["enabled"]):
        out_dir = make_output_dir(train_cfg=train_cfg, resume_checkpoint=resume_checkpoint) if is_main_process else ""
        out_dir = str(_broadcast_object(out_dir, dist_state))
        os.makedirs(out_dir, exist_ok=True)
        _maybe_barrier(dist_state)
    else:
        out_dir = make_output_dir(train_cfg=train_cfg, resume_checkpoint=resume_checkpoint)
        os.makedirs(out_dir, exist_ok=True)

    if is_main_process:
        try:
            with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
            with open(os.path.join(out_dir, "run_meta.txt"), "w", encoding="utf-8") as f:
                f.write("command: " + " ".join(sys.argv) + "\n")
                f.write(f"seed: {cfg['seed']}\n")
                f.write(f"init_checkpoint: {resume_checkpoint}\n")
                f.write(f"llm_train_mode: {cfg['model'].get('llm_train_mode', '')}\n")
                f.write(f"state_update_enabled: {bool(cfg.get('state_update', {}).get('enabled', True))}\n")
                f.write(f"optimize_per_sample: {bool(train_cfg.get('optimize_per_sample', True))}\n")
                f.write(f"distributed: {bool(dist_state['enabled'])}\n")
                f.write(f"world_size: {int(dist_state['world_size'])}\n")
        except Exception as exc:
            wrap_geo_error(
                code="GEO-1105",
                message=f"failed to write training run metadata into {out_dir}",
                exc=exc,
            )
    _maybe_barrier(dist_state)

    task_schemas, text_tokenizer, collator, model = build_geo_components(cfg)
    artifact_cfg = get_artifact_export_cfg(cfg)


    train_set = build_geo_dataset(
        cfg=cfg,
        split=str(cfg["data"]["train_split"]),
        task_schemas=task_schemas,
        max_samples=cfg["data"].get("max_train_samples"),
        train_augment=bool(cfg["data"].get("train_augment", False)),
        crop_to_review_mask=bool(cfg["data"].get("train_crop_to_review_mask", cfg["data"].get("crop_to_review_mask", True))),
        stage="train",
    )
    val_set = build_geo_dataset(
        cfg=cfg,
        split=str(cfg["data"]["val_split"]),
        task_schemas=task_schemas,
        max_samples=cfg["data"].get("max_val_samples"),
        train_augment=False,
        crop_to_review_mask=bool(cfg["data"].get("val_crop_to_review_mask", False)),
        stage="eval",
    )

    checkpoint_obj = maybe_load_model_checkpoint(model, resume_checkpoint)


    if bool(artifact_cfg["enabled"]) and is_main_process:
        band_indices = [int(x) for x in cfg["data"].get("band_indices", [1, 2, 3])]
        image_size = int(cfg["data"]["image_size"])
        train_audit_records = _resolve_tile_audit_records(train_set)
        val_audit_records = _resolve_tile_audit_records(val_set)
        if len(train_audit_records) == 0:
            print(
                "[Audit] train tile audit records are empty even after fallback "
                f"(records={len(train_set)} items={len(getattr(train_set, 'items', []) or [])})",
                flush=True,
            )
        if len(val_audit_records) == 0:
            print(
                "[Audit] val tile audit records are empty even after fallback "
                f"(records={len(val_set)} items={len(getattr(val_set, 'items', []) or [])} split={cfg['data']['val_split']})",
                flush=True,
            )
        export_tile_audit_records(
            audit_records=train_audit_records,
            output_dir=os.path.join(out_dir, "artifacts", "train_patch_audit"),
            band_indices=band_indices,
            image_size=image_size,
            save_kept_patches=bool(artifact_cfg["save_kept_patches"]),
            save_discarded_patches=bool(artifact_cfg["save_discarded_patches"]),
            save_resized_patch_inputs=bool(artifact_cfg["save_resized_patch_inputs"]),
            max_patch_images_per_sample=int(artifact_cfg["max_patch_images_per_sample"]),
        )
        export_tile_audit_records(
            audit_records=val_audit_records,
            output_dir=os.path.join(out_dir, "artifacts", "val_patch_audit"),
            band_indices=band_indices,
            image_size=image_size,
            save_kept_patches=bool(artifact_cfg["save_kept_patches"]),
            save_discarded_patches=bool(artifact_cfg["save_discarded_patches"]),
            save_resized_patch_inputs=bool(artifact_cfg["save_resized_patch_inputs"]),
            max_patch_images_per_sample=int(artifact_cfg["max_patch_images_per_sample"]),
        )
    _maybe_barrier(dist_state)


    batch_size = int(train_cfg["batch_size"])
    val_batch_size = int(train_cfg.get("val_batch_size", batch_size))
    num_workers = int(cfg["data"].get("num_workers", 0))
    val_num_workers = int(cfg["data"].get("val_num_workers", 0))
    sample_patch_sequential = _cfg_bool(train_cfg.get("sample_patch_sequential", True), default=True)
    val_sample_patch_sequential = _cfg_bool(
        train_cfg.get("val_sample_patch_sequential", sample_patch_sequential),
        default=sample_patch_sequential,
    )


    optimize_per_sample = _cfg_bool(
        train_cfg.get("optimize_per_sample", sample_patch_sequential),
        default=sample_patch_sequential,
    ) and bool(sample_patch_sequential)
    task_order = {name: idx for idx, name in enumerate(task_schemas.keys())}
    epoch_is_single_sample = _cfg_bool(train_cfg.get("epoch_is_single_sample", True), default=True)
    if bool(dist_state["enabled"]) and bool(epoch_is_single_sample):
        if is_main_process:
            print(
                "[DDP] epoch_is_single_sample=true is incompatible with multi-GPU sample partitioning; overriding to false.",
                flush=True,
            )
        epoch_is_single_sample = False
    sample_to_indices = _group_sample_indices(getattr(train_set, "items", []), task_order=task_order)
    sample_ids_in_order = list(sample_to_indices.keys())
    sample_batch_counts = _count_sample_batches(getattr(train_set, "items", []), batch_size=batch_size)
    rank_sample_ids_by_rank, rank_batch_loads, max_rank_batches = _assign_samples_to_ranks(
        sample_ids_in_order=sample_ids_in_order,
        sample_batch_counts=sample_batch_counts,
        world_size=int(dist_state["world_size"]),
    )


    if val_sample_patch_sequential:
        val_loader = DataLoader(
            val_set,
            batch_sampler=SampleSequentialBatchSampler(
                items=getattr(val_set, "items", []),
                batch_size=val_batch_size,
                task_order=task_order,
            ),
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=bool(val_num_workers > 0),
            collate_fn=collator,
        )
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=bool(val_num_workers > 0),
            collate_fn=collator,
        )

    device = dist_state["device"]
    model.to(device)
    raw_model = model
    if bool(dist_state["enabled"]):
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [int(dist_state["local_rank"])]
            ddp_kwargs["output_device"] = int(dist_state["local_rank"])
        model = DDP(raw_model, find_unused_parameters=False, **ddp_kwargs)


    llm_dtype = "unknown"
    llm_param_dtype = "unknown"
    sat_proj_dtype = "unknown"
    try:
        llm_dtype = str(getattr(raw_model.llm, "dtype", "unknown"))
    except Exception:
        pass
    try:
        llm_param_dtype = str(next(raw_model.llm.parameters()).dtype)
    except Exception:
        pass
    try:
        sat_proj_dtype = str(next(raw_model.sat_proj.parameters()).dtype)
    except Exception:
        pass
    trainable_params = [param for param in raw_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for geo training.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", False)) and device.type == "cuda")


    resume_state = maybe_resume_training_state(
        optimizer=optimizer,
        scaler=scaler,
        checkpoint_obj=checkpoint_obj,
        load_optimizer=bool(train_cfg.get("resume_optimizer", True)),
        load_scaler=bool(train_cfg.get("resume_scaler", True)),
    )

    start_epoch = int(resume_state["epoch"]) + 1 if resume_checkpoint else 1
    global_step = int(resume_state["global_step"]) if resume_checkpoint else 0
    best_val = float(resume_state["best_val"]) if resume_state["best_val"] is not None else 1e9
    epochs = int(train_cfg["epochs"])
    if epoch_is_single_sample and sample_ids_in_order:
        total_steps = 0
        for epoch_id in range(1, epochs + 1):
            sample_id = sample_ids_in_order[(epoch_id - 1) % len(sample_ids_in_order)]
            total_steps += int(sample_batch_counts.get(sample_id, 1))
        total_steps = max(1, total_steps)
    elif bool(dist_state["enabled"]):
        total_steps = max(1, epochs * max(1, int(max_rank_batches)))
    else:
        full_train_loader = DataLoader(
            train_set,
            batch_sampler=SampleSequentialBatchSampler(
                items=getattr(train_set, "items", []),
                batch_size=batch_size,
                task_order=task_order,
            ),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0),
            collate_fn=collator,
        ) if sample_patch_sequential else DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers > 0),
            collate_fn=collator,
        )
        total_steps = max(
            1,
            epochs
            * (
                max(1, len(sample_batch_counts))
                if optimize_per_sample
                else max(1, len(full_train_loader))
            ),
        )
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    mem_risk = _estimate_memory_risk(cfg=cfg, batch_size=batch_size)
    if epoch_is_single_sample and optimize_per_sample:
        if is_main_process:
            print("[Init] epoch_is_single_sample=true forces optimize_per_sample=false for patch-level optimizer steps.", flush=True)
        optimize_per_sample = False
    if is_main_process:
        print(f"[Init] Train records={len(train_set)} Val records={len(val_set)}", flush=True)
        print(
            f"[Init] Train items={len(getattr(train_set, 'items', []) or [])} "
            f"Val items={len(getattr(val_set, 'items', []) or [])} "
            f"train_split={cfg['data']['train_split']} val_split={cfg['data']['val_split']}",
            flush=True,
        )
        train_cache = getattr(train_set, "cache_stats", None)
        if isinstance(train_cache, dict):
            print(
                f"[Init] Train cache enabled={train_cache.get('enabled', False)} "
                f"root={train_cache.get('cache_root', '') or '-'} "
                f"existing={train_cache.get('existing_records', 0)}/{train_cache.get('total_records', 0)} "
                f"missing={train_cache.get('missing_records', 0)} "
                f"write_enabled={train_cache.get('write_enabled', False)}",
                flush=True,
            )
        val_cache = getattr(val_set, "cache_stats", None)
        if isinstance(val_cache, dict):
            print(
                f"[Init] Val cache enabled={val_cache.get('enabled', False)} "
                f"root={val_cache.get('cache_root', '') or '-'} "
                f"existing={val_cache.get('existing_records', 0)}/{val_cache.get('total_records', 0)} "
                f"missing={val_cache.get('missing_records', 0)} "
                f"write_enabled={val_cache.get('write_enabled', False)}",
                flush=True,
            )
        print(f"[Init] Device={device}", flush=True)
        if device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device)
                total_gb = float(props.total_memory) / float(1024 ** 3)
                print(f"[Init] GPU={props.name} total_vram_gb={total_gb:.1f}", flush=True)
                allocated_gb = float(torch.cuda.memory_allocated(device)) / float(1024 ** 3)
                reserved_gb = float(torch.cuda.memory_reserved(device)) / float(1024 ** 3)
                print(
                    f"[Init] CUDA after model.to allocated_gb={allocated_gb:.2f} reserved_gb={reserved_gb:.2f}",
                    flush=True,
                )
            except Exception:
                pass
        print(
            f"[Init] Memory risk level={mem_risk['level']} score={mem_risk['score']} "
            f"mode={mem_risk['llm_mode']} reasons={mem_risk['reasons']}",
            flush=True,
        )
        if mem_risk["suggestions"]:
            print(f"[Init] Memory suggestions={mem_risk['suggestions']}", flush=True)
        print(f"[Init] Tokenizer vocab={text_tokenizer.vocab_size}", flush=True)
        print(
            f"[Init] LLM dtype={llm_dtype} llm_param_dtype={llm_param_dtype} sat_proj_dtype={sat_proj_dtype}",
            flush=True,
        )
        print(f"[Init] Trainable params={raw_model.trainable_parameter_summary()}", flush=True)
        print(f"[Init] Output dir={out_dir}", flush=True)
        print(f"[Init] State update cfg={cfg.get('state_update', {})}", flush=True)
        print(
            f"[Init] Train order sample_patch_sequential={sample_patch_sequential} "
            f"val_sample_patch_sequential={val_sample_patch_sequential} "
            f"optimize_per_sample={optimize_per_sample} "
            f"epoch_is_single_sample={epoch_is_single_sample}",
            flush=True,
        )
        print(
            f"[Init] Train samples={len(sample_ids_in_order)} sample_ids_preview={sample_ids_in_order[:5]}",
            flush=True,
        )
        if bool(dist_state["enabled"]):
            print(f"[DDP] rank_batch_loads={rank_batch_loads} max_batches_per_rank={int(max_rank_batches)}", flush=True)


    for epoch in range(start_epoch, epochs + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        model.train()
        ep_loss = 0.0
        ep_count = 0
        exported_train_batches = 0
        t0 = time.time()
        current_sample_id = ""
        if epoch_is_single_sample:
            if not sample_ids_in_order:
                raise RuntimeError("No train samples available for sample-epoch training.")
            current_sample_id = sample_ids_in_order[(epoch - 1) % len(sample_ids_in_order)]
            current_indices = sample_to_indices.get(current_sample_id, [])
            train_loader = _build_subset_loader(
                dataset=train_set,
                indices=current_indices,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collator,
            )
            if is_main_process:
                print(
                    f"[Epoch {epoch}] Training sample_id={current_sample_id} patch_batches={len(train_loader)}",
                    flush=True,
                )
        elif bool(dist_state["enabled"]):
            assigned_sample_ids = rank_sample_ids_by_rank[int(dist_state["rank"])]
            current_indices = []
            for sample_id in assigned_sample_ids:
                current_indices.extend(sample_to_indices.get(sample_id, []))
            train_loader = _build_rank_subset_loader(
                dataset=train_set,
                indices=current_indices,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collator,
                sample_patch_sequential=sample_patch_sequential,
                task_order=task_order,
                target_num_batches=max_rank_batches,
            )
            if is_main_process:
                print(
                    f"[Epoch {epoch}] DDP assigned_samples={assigned_sample_ids[:8]} "
                    f"sample_count={len(assigned_sample_ids)} patch_batches={len(train_loader)}",
                    flush=True,
                )
        else:
            train_loader = DataLoader(
                train_set,
                batch_sampler=SampleSequentialBatchSampler(
                    items=getattr(train_set, "items", []),
                    batch_size=batch_size,
                    task_order=task_order,
                ),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=bool(num_workers > 0),
                collate_fn=collator,
            ) if sample_patch_sequential else DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=bool(num_workers > 0),
                collate_fn=collator,
            )
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not is_main_process)
        current_lr = float(train_cfg["lr"])
        epoch_pred_features: dict[str, list[dict]] = {name: [] for name in task_schemas.keys()}
        epoch_raster_meta = None
        max_train_batches_cfg = int(artifact_cfg["max_batches_per_epoch"])


        for batch_index, batch in enumerate(pbar):
            batch_sample_id = str(batch["sample_ids"][0]) if batch.get("sample_ids") else f"batch_{batch_index}"
            current_lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=float(train_cfg["lr"]),
                warmup_steps=int(train_cfg.get("warmup_steps", 0)),
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            if batch_index == 0 and is_main_process:
                prompt_lens = batch["prompt_attention_mask"].sum(dim=1).tolist()
                state_lens = batch["state_attention_mask"].sum(dim=1).tolist()
                target_lens = batch["map_attention_mask"].sum(dim=1).tolist()
                print(
                    f"[Epoch {epoch}] Batch0 image_shape={tuple(batch['image'].shape)} "
                    f"prompt_lens={prompt_lens} state_lens={state_lens} target_lens={target_lens}",
                    flush=True,
                )
                if device.type == "cuda":
                    try:
                        alloc_gb = float(torch.cuda.memory_allocated(device)) / float(1024 ** 3)
                        reserved_gb = float(torch.cuda.memory_reserved(device)) / float(1024 ** 3)
                        print(
                            f"[Epoch {epoch}] Batch0 preforward allocated_gb={alloc_gb:.2f} "
                            f"reserved_gb={reserved_gb:.2f}",
                            flush=True,
                        )
                    except Exception:
                        pass
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(train_cfg.get("amp", False)) and device.type == "cuda"):
                out = model(
                    image=batch["image"].to(device),
                    prompt_input_ids=batch["prompt_input_ids"].to(device),
                    prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                    pv_images=None,
                    state_input_ids=batch["state_input_ids"].to(device),
                    state_attention_mask=batch["state_attention_mask"].to(device),
                    map_input_ids=batch["map_input_ids"].to(device),
                    map_attention_mask=batch["map_attention_mask"].to(device),
                    return_logits=False,
                )
                loss = out["loss"]
            raw_loss_value = float(loss.item())
            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if device.type == "cuda" and batch_index == 0 and is_main_process:
                try:
                    alloc_gb = float(torch.cuda.memory_allocated(device)) / float(1024 ** 3)
                    reserved_gb = float(torch.cuda.memory_reserved(device)) / float(1024 ** 3)
                    peak_alloc_gb = float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 3)
                    peak_reserved_gb = float(torch.cuda.max_memory_reserved(device)) / float(1024 ** 3)
                    print(
                        f"[Epoch {epoch}] Batch0 CUDA allocated_gb={alloc_gb:.2f} reserved_gb={reserved_gb:.2f} "
                        f"peak_alloc_gb={peak_alloc_gb:.2f} peak_reserved_gb={peak_reserved_gb:.2f}",
                        flush=True,
                    )
                except Exception:
                    pass


            ep_loss += raw_loss_value * batch["image"].shape[0]
            ep_count += batch["image"].shape[0]
            postfix = {
                "loss": f"{raw_loss_value:.4f}",
                "lr": f"{current_lr:.2e}",
                "sample": str(batch_sample_id),
                "tile": f"{int(batch['tile_indices'][0]) + 1}/{int(batch['tile_counts'][0])}",
            }
            if is_main_process:
                pbar.set_postfix(**postfix)
            if (
                is_main_process
                and
                bool(artifact_cfg["enabled"])
                and bool(artifact_cfg["save_train_batch_geojson"])
                and (max_train_batches_cfg <= 0 or exported_train_batches < max_train_batches_cfg)
            ):
                print(
                    f"[Epoch {epoch}] Preparing train artifact export "
                    f"sample={batch_sample_id} tile={int(batch['tile_indices'][0]) + 1}/{int(batch['tile_counts'][0])} "
                    f"save_predictions={bool(artifact_cfg.get('save_train_batch_predictions', False))}",
                    flush=True,
                )
                if bool(artifact_cfg.get("save_train_batch_predictions", False)):
                    print(
                        f"[Epoch {epoch}] Exporting train prediction snapshot "
                        f"{exported_train_batches + 1}/"
                        f"{'all' if max_train_batches_cfg <= 0 else max_train_batches_cfg} "
                        f"for sample={batch_sample_id} tile={int(batch['tile_indices'][0]) + 1}/{int(batch['tile_counts'][0])}",
                        flush=True,
                    )
                exported = export_batch_geojson_snapshots(
                    cfg=cfg,
                    task_schemas=task_schemas,
                    text_tokenizer=text_tokenizer,
                    model=raw_model,
                    batch=batch,
                    device=device,
                    decode_cfg=cfg.get("decode", {}),
                    output_dir=out_dir,
                    stage="train",
                    epoch=int(epoch),
                    batch_index=int(batch_index),
                )
                exported_train_batches += 1
                if bool(artifact_cfg.get("save_train_batch_predictions", False)):
                    print(
                        f"[Epoch {epoch}] Train prediction snapshot finished",
                        flush=True,
                    )
                for record in exported:
                    epoch_raster_meta = record.get("raster_meta", epoch_raster_meta)
                    epoch_pred_features.setdefault(str(record["task_name"]), []).extend(record.get("feature_records", []))

        if bool(dist_state["enabled"]):
            ep_loss = _reduce_sum(ep_loss, device, dist_state)
            ep_count = int(round(_reduce_sum(ep_count, device, dist_state)))
        train_loss = ep_loss / max(ep_count, 1)
        train_sec = time.time() - t0
        if is_main_process and epoch_is_single_sample and current_sample_id and epoch_raster_meta is not None:
            stitched_out_dir = os.path.join(out_dir, "artifacts", "train", f"epoch_{int(epoch):04d}", f"sample_{current_sample_id}_stitched")
            os.makedirs(stitched_out_dir, exist_ok=True)
            stitched_summary = {
                "sample_id": current_sample_id,
                "epoch": int(epoch),
                "tasks": {},
            }
            stitched_records_by_task = {}
            for task_name, task_schema in task_schemas.items():
                pred_records = epoch_pred_features.get(task_name, [])
                stitched_records = deduplicate_feature_records(
                    task_schema=task_schema,
                    feature_records=pred_records,
                    raster_meta=epoch_raster_meta,
                    line_distance_threshold_m=float(cfg.get("postprocess", {}).get("line_dedup_distance_m", 1.0)),
                    polygon_iou_threshold=float(cfg.get("postprocess", {}).get("polygon_dedup_iou", 0.5)),
                )
                stitched_records_by_task[task_name] = stitched_records
                stitched_summary["tasks"][task_name] = {
                    "patch_pred_feature_count": int(len(pred_records)),
                    "stitched_feature_count": int(len(stitched_records)),
                }

            stitched_geojson_by_task = {}
            for task_name, task_schema in task_schemas.items():
                stitched_records = stitched_records_by_task.get(task_name, [])
                stitched_geojson_by_task[task_name] = pixel_features_to_geojson(
                    task_schema=task_schema,
                    feature_records=stitched_records,
                    raster_meta=epoch_raster_meta,
                )
            stitched_geojson_by_task = assign_incremental_feature_ids(stitched_geojson_by_task)
            for task_name, task_schema in task_schemas.items():
                geojson_dict = stitched_geojson_by_task.get(task_name)
                if geojson_dict is None:
                    continue
                with open(os.path.join(stitched_out_dir, f"{task_schema.collection_name}.stitched.pred.geojson"), "w", encoding="utf-8") as f:
                    f.write(geojson_dumps(geojson_dict))
                with open(os.path.join(stitched_out_dir, f"{task_schema.collection_name}.geojson"), "w", encoding="utf-8") as f:
                    f.write(geojson_dumps(geojson_dict))
                with open(os.path.join(stitched_out_dir, f"{task_schema.collection_name}.uv.geojson"), "w", encoding="utf-8") as f:
                    f.write(
                        geojson_dumps(
                            pixel_features_to_uv_geojson(
                                task_schema=task_schema,
                                feature_records=stitched_records_by_task.get(task_name, []),
                                resize_ctx=build_resize_context(
                                    width=int(epoch_raster_meta.width),
                                    height=int(epoch_raster_meta.height),
                                    target_size=int(cfg["data"]["image_size"]),
                                    crop_bbox=None,
                                ),
                            )
                        )
                    )
            save_json(os.path.join(stitched_out_dir, "summary.json"), stitched_summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_t0 = time.time()


        if bool(dist_state["enabled"]):
            if is_main_process:
                val_loss, val_token_acc = run_val(
                    model,
                    val_loader,
                    device,
                    desc=f"Val {epoch}/{epochs}",
                    cfg=cfg,
                    task_schemas=task_schemas,
                    text_tokenizer=text_tokenizer,
                    out_dir=out_dir,
                    epoch=int(epoch),
                    predict_model=raw_model,
                )
            else:
                val_loss = 0.0
                val_token_acc = 0.0
            val_loss = float(_broadcast_object(val_loss, dist_state))
            val_token_acc = float(_broadcast_object(val_token_acc, dist_state))
            _maybe_barrier(dist_state)
        else:
            val_loss, val_token_acc = run_val(
                model,
                val_loader,
                device,
                desc=f"Val {epoch}/{epochs}",
                cfg=cfg,
                task_schemas=task_schemas,
                text_tokenizer=text_tokenizer,
                out_dir=out_dir,
                epoch=int(epoch),
                predict_model=raw_model,
            )
        val_sec = time.time() - val_t0

        best_updated = False
        checkpoint_sec = 0.0
        if is_main_process:
            if val_loss < best_val:
                best_val = val_loss
                best_updated = True
            checkpoint_obj = build_checkpoint_obj(
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val,
                cfg=cfg,
            )
            latest_path = os.path.join(out_dir, "latest.pt")
            best_path = os.path.join(out_dir, "best.pt")
            save_t0 = time.time()
            if bool(train_cfg.get("save_latest", True)):
                atomic_torch_save(checkpoint_obj, latest_path)
            if best_updated:
                atomic_torch_save(checkpoint_obj, best_path)
            checkpoint_sec = time.time() - save_t0

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_token_acc": val_token_acc,
                "train_sec": train_sec,
                "val_sec": val_sec,
                "checkpoint_sec": checkpoint_sec,
                "global_step": global_step,
                "best_updated": best_updated,
                "output_dir": out_dir,
                "world_size": int(dist_state["world_size"]),
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(record, flush=True)
            train_runtime_hits = int(getattr(train_set, "cache_runtime_hits", 0))
            train_runtime_misses = int(getattr(train_set, "cache_runtime_misses", 0))
            val_runtime_hits = int(getattr(val_set, "cache_runtime_hits", 0))
            val_runtime_misses = int(getattr(val_set, "cache_runtime_misses", 0))
            print(
                f"[Epoch {epoch}] Cache runtime train_hit={train_runtime_hits} train_miss={train_runtime_misses} "
                f"val_hit={val_runtime_hits} val_miss={val_runtime_misses}",
                flush=True,
            )
        if bool(dist_state["enabled"]):
            best_val = float(_broadcast_object(best_val, dist_state))
            _maybe_barrier(dist_state)
    _destroy_distributed_training(dist_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="")
    args = parser.parse_args()
    run_training(config_path=args.config, mode_override=str(args.mode or ""))


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1000")
