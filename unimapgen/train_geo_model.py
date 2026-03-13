import argparse
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from unimapgen.geo.artifacts import (
    export_batch_geojson_snapshots,
    export_tile_audit_records,
    get_artifact_export_cfg,
)
from unimapgen.geo.errors import run_with_geo_error_boundary, wrap_geo_error
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
    with torch.inference_mode():
        for batch_index, batch in enumerate(iterator):
            out = model(
                image=batch["image"].to(device),
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                pv_images=None,
                state_input_ids=batch["state_input_ids"].to(device),
                state_attention_mask=batch["state_attention_mask"].to(device),
                map_input_ids=batch["map_input_ids"].to(device),
                map_attention_mask=batch["map_attention_mask"].to(device),
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
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        collate_fn=collator,
    )
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
    total_steps = max(1, epochs * max(1, len(train_loader)))
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    print(f"[Init] Train records={len(train_set)} Val records={len(val_set)}", flush=True)
    print(f"[Init] Device={device}", flush=True)
    print(f"[Init] Tokenizer vocab={text_tokenizer.vocab_size}", flush=True)
    print(f"[Init] Trainable params={model.trainable_parameter_summary()}", flush=True)
    print(f"[Init] Output dir={out_dir}", flush=True)
    print(f"[Init] State update cfg={cfg.get('state_update', {})}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_count = 0
        exported_train_batches = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_index, batch in enumerate(pbar):
            lr = cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                base_lr=float(train_cfg["lr"]),
                warmup_steps=int(train_cfg.get("warmup_steps", 0)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

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
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(loss.item()) * batch["image"].shape[0]
            ep_count += batch["image"].shape[0]
            global_step += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{lr:.2e}")
            if (
                bool(artifact_cfg["enabled"])
                and bool(artifact_cfg["save_train_batch_geojson"])
                and exported_train_batches < max(0, int(artifact_cfg["max_batches_per_epoch"]))
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
                    stage="train",
                    epoch=int(epoch),
                    batch_index=int(batch_index),
                )
                exported_train_batches += 1

        train_loss = ep_loss / max(ep_count, 1)
        train_sec = time.time() - t0
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="")
    args = parser.parse_args()
    run_training(config_path=args.config, mode_override=str(args.mode or ""))


if __name__ == "__main__":
    run_with_geo_error_boundary(main, default_code="GEO-1000")
