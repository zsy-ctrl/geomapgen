import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import torch
import yaml
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


class SampleSequentialBatchSampler(BatchSampler):
    def __init__(self, items, batch_size: int, task_order: dict[str, int], drop_last: bool = False) -> None:
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        grouped: OrderedDict[str, list[tuple[int, dict]]] = OrderedDict()
        for index, item in enumerate(items):
            sample_id = str(item.get("sample_id", ""))
            grouped.setdefault(sample_id, []).append((int(index), item))

        self._batches: list[list[int]] = []
        for _, entries in grouped.items():
            entries.sort(
                key=lambda pair: (
                    int(pair[1].get("tile_index", 0)),
                    int(task_order.get(str(pair[1].get("task_name", "")), 10**6)),
                    int(pair[0]),
                )
            )
            indices = [int(index) for index, _ in entries]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                self._batches.append(batch)

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
):
    model.eval()
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
                and exported_batches < max(0, int(artifact_cfg["max_batches_per_epoch"]))
            ):
                export_batch_geojson_snapshots(
                    cfg=cfg,
                    task_schemas=task_schemas,
                    text_tokenizer=text_tokenizer,
                    model=model,
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
    cfg = load_yaml(config_path)
    if mode_override:
        cfg.setdefault("model", {})
        cfg["model"]["llm_train_mode"] = str(mode_override)

    set_seed(int(cfg["seed"]))
    train_cfg = cfg["train"]
    resume_checkpoint = str(train_cfg.get("init_checkpoint", "")).strip()
    out_dir = make_output_dir(train_cfg=train_cfg, resume_checkpoint=resume_checkpoint)
    os.makedirs(out_dir, exist_ok=True)

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
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1105",
            message=f"failed to write training run metadata into {out_dir}",
            exc=exc,
        )

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

    if bool(artifact_cfg["enabled"]):
        band_indices = [int(x) for x in cfg["data"].get("band_indices", [1, 2, 3])]
        image_size = int(cfg["data"]["image_size"])
        export_tile_audit_records(
            audit_records=getattr(train_set, "tile_audit_records", []),
            output_dir=os.path.join(out_dir, "artifacts", "train_patch_audit"),
            band_indices=band_indices,
            image_size=image_size,
            save_kept_patches=bool(artifact_cfg["save_kept_patches"]),
            save_discarded_patches=bool(artifact_cfg["save_discarded_patches"]),
            save_resized_patch_inputs=bool(artifact_cfg["save_resized_patch_inputs"]),
            max_patch_images_per_sample=int(artifact_cfg["max_patch_images_per_sample"]),
        )
        export_tile_audit_records(
            audit_records=getattr(val_set, "tile_audit_records", []),
            output_dir=os.path.join(out_dir, "artifacts", "val_patch_audit"),
            band_indices=band_indices,
            image_size=image_size,
            save_kept_patches=bool(artifact_cfg["save_kept_patches"]),
            save_discarded_patches=bool(artifact_cfg["save_discarded_patches"]),
            save_resized_patch_inputs=bool(artifact_cfg["save_resized_patch_inputs"]),
            max_patch_images_per_sample=int(artifact_cfg["max_patch_images_per_sample"]),
        )

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
    sample_to_indices = _group_sample_indices(getattr(train_set, "items", []), task_order=task_order)
    sample_ids_in_order = list(sample_to_indices.keys())
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

    device = select_torch_device(prefer_cuda=True)
    model.to(device)
    llm_dtype = "unknown"
    llm_param_dtype = "unknown"
    sat_proj_dtype = "unknown"
    try:
        llm_dtype = str(getattr(model.llm, "dtype", "unknown"))
    except Exception:
        pass
    try:
        llm_param_dtype = str(next(model.llm.parameters()).dtype)
    except Exception:
        pass
    try:
        sat_proj_dtype = str(next(model.sat_proj.parameters()).dtype)
    except Exception:
        pass
    trainable_params = [param for param in model.parameters() if param.requires_grad]
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
    sample_batch_counts = _count_sample_batches(getattr(train_set, "items", []), batch_size=batch_size)
    if epoch_is_single_sample and sample_ids_in_order:
        total_steps = 0
        for epoch_id in range(1, epochs + 1):
            sample_id = sample_ids_in_order[(epoch_id - 1) % len(sample_ids_in_order)]
            total_steps += int(sample_batch_counts.get(sample_id, 1))
        total_steps = max(1, total_steps)
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

    print(f"[Init] Train records={len(train_set)} Val records={len(val_set)}", flush=True)
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
    mem_risk = _estimate_memory_risk(cfg=cfg, batch_size=batch_size)
    if epoch_is_single_sample and optimize_per_sample:
        print("[Init] epoch_is_single_sample=true forces optimize_per_sample=false for patch-level optimizer steps.", flush=True)
        optimize_per_sample = False
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
    print(f"[Init] Trainable params={model.trainable_parameter_summary()}", flush=True)
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
            print(
                f"[Epoch {epoch}] Training sample_id={current_sample_id} patch_batches={len(train_loader)}",
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        current_lr = float(train_cfg["lr"])
        epoch_pred_features: dict[str, list[dict]] = {name: [] for name in task_schemas.keys()}
        epoch_raster_meta = None
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
            if batch_index == 0:
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
            if device.type == "cuda" and batch_index == 0:
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
            pbar.set_postfix(**postfix)
            if bool(artifact_cfg["enabled"]) and bool(artifact_cfg["save_train_batch_geojson"]):
                exported = export_batch_geojson_snapshots(
                    cfg=cfg,
                    task_schemas=task_schemas,
                    text_tokenizer=text_tokenizer,
                    model=model,
                    batch=batch,
                    device=device,
                    decode_cfg=cfg.get("decode", {}),
                    output_dir=out_dir,
                    stage="train",
                    epoch=int(epoch),
                    batch_index=int(batch_index),
                )
                exported_train_batches += 1
                for record in exported:
                    epoch_raster_meta = record.get("raster_meta", epoch_raster_meta)
                    epoch_pred_features.setdefault(str(record["task_name"]), []).extend(record.get("feature_records", []))

        train_loss = ep_loss / max(ep_count, 1)
        train_sec = time.time() - t0
        if epoch_is_single_sample and current_sample_id and epoch_raster_meta is not None:
            stitched_out_dir = os.path.join(out_dir, "artifacts", "train", f"epoch_{int(epoch):04d}", f"sample_{current_sample_id}_stitched")
            os.makedirs(stitched_out_dir, exist_ok=True)
            stitched_summary = {
                "sample_id": current_sample_id,
                "epoch": int(epoch),
                "tasks": {},
            }
            for task_name, task_schema in task_schemas.items():
                pred_records = epoch_pred_features.get(task_name, [])
                stitched_records = deduplicate_feature_records(
                    task_schema=task_schema,
                    feature_records=pred_records,
                    raster_meta=epoch_raster_meta,
                    line_distance_threshold_m=float(cfg.get("postprocess", {}).get("line_dedup_distance_m", 1.0)),
                    polygon_iou_threshold=float(cfg.get("postprocess", {}).get("polygon_dedup_iou", 0.5)),
                )
                save_geojson_snapshot(
                    path=os.path.join(stitched_out_dir, f"{task_schema.collection_name}.stitched.pred.geojson"),
                    task_schema=task_schema,
                    feature_records=stitched_records,
                    raster_meta=epoch_raster_meta,
                )
                stitched_summary["tasks"][task_name] = {
                    "patch_pred_feature_count": int(len(pred_records)),
                    "stitched_feature_count": int(len(stitched_records)),
                }
            save_json(os.path.join(stitched_out_dir, "summary.json"), stitched_summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_t0 = time.time()
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
        )
        val_sec = time.time() - val_t0

        best_updated = False
        if val_loss < best_val:
            best_val = val_loss
            best_updated = True
        checkpoint_obj = build_checkpoint_obj(
            model=model,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="")
    args = parser.parse_args()
    run_training(config_path=args.config, mode_override=str(args.mode or ""))


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1000")
