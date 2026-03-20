import argparse
import json
from pathlib import Path
from typing import Dict, List

from export_llamafactory_state_sft_from_raw_family_manifest import build_sample_rng, choose_state_mode
from geo_current_dataset_v1_common import (
    DEFAULT_STAGEA_PROMPT_TEMPLATE,
    DEFAULT_STAGEA_SYSTEM_PROMPT,
    DEFAULT_STAGEB_PROMPT_TEMPLATE,
    DEFAULT_STAGEB_SYSTEM_PROMPT,
    apply_state_mode,
    build_owned_segments_by_patch,
    build_patch_image,
    build_patch_only_record,
    build_patch_target_lines,
    build_patch_target_lines_float,
    build_state_record,
    ensure_dir,
    extract_state_lines,
    family_global_lines,
    load_family_raster_and_mask,
    load_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage A and Stage B datasets together from current-dataset family manifests.")
    parser.add_argument("--family-manifest", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--band-indices", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--resample-step-px", type=float, default=4.0)
    parser.add_argument("--boundary-tol-px", type=float, default=2.5)
    parser.add_argument("--trace-points", type=int, default=8)
    parser.add_argument("--state-mixture-mode", type=str, default="full", choices=["full", "mixed"])
    parser.add_argument("--state-no-state-ratio", type=float, default=0.30)
    parser.add_argument("--state-weak-ratio", type=float, default=0.40)
    parser.add_argument("--state-full-ratio", type=float, default=0.30)
    parser.add_argument("--state-weak-trace-points", type=int, default=3)
    parser.add_argument("--state-line-dropout", type=float, default=0.40)
    parser.add_argument("--state-point-jitter-px", type=float, default=2.0)
    parser.add_argument("--state-truncate-prob", type=float, default=0.30)
    parser.add_argument("--include-lane", action="store_true")
    parser.add_argument("--include-intersection-boundary", action="store_true")
    parser.add_argument("--max-families-per-split", type=int, default=0)
    parser.add_argument("--use-system-prompt", action="store_true")
    parser.add_argument("--stagea-system-prompt", type=str, default=DEFAULT_STAGEA_SYSTEM_PROMPT)
    parser.add_argument("--stagea-prompt-template", type=str, default=DEFAULT_STAGEA_PROMPT_TEMPLATE)
    parser.add_argument("--stageb-system-prompt", type=str, default=DEFAULT_STAGEB_SYSTEM_PROMPT)
    parser.add_argument("--stageb-prompt-template", type=str, default=DEFAULT_STAGEB_PROMPT_TEMPLATE)
    return parser.parse_args()


def build_dataset_registry(output_root: Path, prefix: str) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}
    for split in ("train", "val"):
        dataset_file = output_root / f"{split}.jsonl"
        if not dataset_file.is_file():
            continue
        registry[f"{prefix}_{split}"] = {
            "file_name": str(dataset_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    return registry


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    stage_a_root = output_root / "stage_a" / "dataset"
    stage_b_root = output_root / "stage_b" / "dataset"
    ensure_dir(stage_a_root)
    ensure_dir(stage_b_root)
    families = load_jsonl(Path(args.family_manifest).resolve())
    include_lane = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_lane)
    include_intersection_boundary = True if (not bool(args.include_lane) and not bool(args.include_intersection_boundary)) else bool(args.include_intersection_boundary)
    stagea_system_prompt = str(args.stagea_system_prompt).strip() if bool(args.use_system_prompt) else ""
    stageb_system_prompt = str(args.stageb_system_prompt).strip() if bool(args.use_system_prompt) else ""

    stagea_rows: Dict[str, List[Dict]] = {str(split): [] for split in args.splits}
    stagea_meta_rows: Dict[str, List[Dict]] = {str(split): [] for split in args.splits}
    stageb_rows: Dict[str, List[Dict]] = {str(split): [] for split in args.splits}
    stageb_meta_rows: Dict[str, List[Dict]] = {str(split): [] for split in args.splits}
    family_counts: Dict[str, int] = {str(split): 0 for split in args.splits}

    for family in families:
        split = str(family.get("split"))
        if split not in stagea_rows:
            continue
        family_counts[split] += 1
        if int(args.max_families_per_split) > 0 and family_counts[split] > int(args.max_families_per_split):
            continue
        raw_image_hwc, raster_meta, review_mask = load_family_raster_and_mask(
            family=family,
            band_indices=[int(x) for x in args.band_indices],
            mask_threshold=int(args.mask_threshold),
        )
        global_lines = family_global_lines(
            family=family,
            raster_meta=raster_meta,
            include_lane=include_lane,
            include_intersection=include_intersection_boundary,
        )
        owned_segments_by_patch = build_owned_segments_by_patch(
            family=family,
            global_lines=global_lines,
            review_mask=review_mask,
            resample_step_px=float(args.resample_step_px),
            boundary_tol_px=float(args.boundary_tol_px),
        )
        for patch in sorted(list(family["patches"]), key=lambda item: int(item["patch_id"])):
            patch_id = int(patch["patch_id"])
            patch_image = build_patch_image(raw_image_hwc=raw_image_hwc, patch=patch)
            target_lines = build_patch_target_lines(owned_segments_by_patch.get(patch_id, []), patch=patch)
            target_lines_float = build_patch_target_lines_float(owned_segments_by_patch.get(patch_id, []), patch=patch)
            image_rel = Path("images") / split / str(family["family_id"]) / f"p{patch_id:04d}.png"

            out_stagea_image = stage_a_root / image_rel
            ensure_dir(out_stagea_image.parent)
            patch_image.save(out_stagea_image)

            out_stageb_image = stage_b_root / image_rel
            ensure_dir(out_stageb_image.parent)
            patch_image.save(out_stageb_image)

            sample_id = f"{family['family_id']}_p{patch_id:04d}"
            stagea_rows[split].append(
                build_patch_only_record(
                    image_rel_path=image_rel.as_posix(),
                    target_lines=target_lines,
                    sample_id=sample_id,
                    system_prompt=stagea_system_prompt,
                    prompt_template=str(args.stagea_prompt_template),
                )
            )
            stagea_meta_rows[split].append(
                {
                    "id": sample_id,
                    "split": split,
                    "family_id": family["family_id"],
                    "source_image": family["source_image"],
                    "patch_id": patch_id,
                    "row": int(patch["row"]),
                    "col": int(patch["col"]),
                    "crop_box": patch["crop_box"],
                    "keep_box": patch["keep_box"],
                    "mask_ratio": float(patch.get("mask_ratio", 0.0)),
                    "mask_pixels": int(patch.get("mask_pixels", 0)),
                    "num_target_lines": len(target_lines),
                    "target_lines": target_lines,
                    "target_lines_float": target_lines_float,
                }
            )

            raw_state_lines = extract_state_lines(
                patch=patch,
                family=family,
                owned_segments_by_patch=owned_segments_by_patch,
                trace_points=int(args.trace_points),
                boundary_tol_px=float(args.boundary_tol_px),
            )
            sample_rng = build_sample_rng(sample_id)
            state_mode = choose_state_mode(
                rng=sample_rng,
                mixture_mode=str(args.state_mixture_mode),
                raw_state_lines=raw_state_lines,
                no_state_ratio=float(args.state_no_state_ratio),
                weak_ratio=float(args.state_weak_ratio),
                full_ratio=float(args.state_full_ratio),
            )
            patch_size = int(patch["crop_box"]["x_max"] - patch["crop_box"]["x_min"])
            state_lines = apply_state_mode(
                raw_state_lines=raw_state_lines,
                state_mode=state_mode,
                patch_size=patch_size,
                weak_trace_points=int(args.state_weak_trace_points),
                state_line_dropout=float(args.state_line_dropout),
                state_point_jitter_px=float(args.state_point_jitter_px),
                state_truncate_prob=float(args.state_truncate_prob),
                rng=sample_rng,
            )
            stageb_rows[split].append(
                build_state_record(
                    image_rel_path=image_rel.as_posix(),
                    state_lines=state_lines,
                    target_lines=target_lines,
                    sample_id=sample_id,
                    system_prompt=stageb_system_prompt,
                    prompt_template=str(args.stageb_prompt_template),
                )
            )
            stageb_meta_rows[split].append(
                {
                    "id": sample_id,
                    "split": split,
                    "family_id": family["family_id"],
                    "source_image": family["source_image"],
                    "patch_id": patch_id,
                    "row": int(patch["row"]),
                    "col": int(patch["col"]),
                    "crop_box": patch["crop_box"],
                    "keep_box": patch["keep_box"],
                    "state_mode": str(state_mode),
                    "num_state_lines": len(state_lines),
                    "num_target_lines": len(target_lines),
                    "state_lines": state_lines,
                    "target_lines": target_lines,
                    "target_lines_float": target_lines_float,
                }
            )

    stage_a_summary: Dict[str, Dict[str, int]] = {}
    stage_b_summary: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        split = str(split)
        stage_a_summary[split] = {
            "families": int(family_counts.get(split, 0)),
            "samples": write_jsonl(stage_a_root / f"{split}.jsonl", stagea_rows[split]),
            "meta_samples": write_jsonl(stage_a_root / f"meta_{split}.jsonl", stagea_meta_rows[split]),
        }
        stage_b_summary[split] = {
            "families": int(family_counts.get(split, 0)),
            "samples": write_jsonl(stage_b_root / f"{split}.jsonl", stageb_rows[split]),
            "meta_samples": write_jsonl(stage_b_root / f"meta_{split}.jsonl", stageb_meta_rows[split]),
        }

    with (stage_a_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(build_dataset_registry(stage_a_root, "unimapgen_geo_current_patch_only"), f, ensure_ascii=False, indent=2)
    with (stage_b_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(build_dataset_registry(stage_b_root, "unimapgen_geo_current_state"), f, ensure_ascii=False, indent=2)

    with (stage_a_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name_prefix": "unimapgen_geo_current_patch_only",
                "source_family_manifest": str(Path(args.family_manifest).resolve()),
                "summary": stage_a_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (stage_b_root / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name_prefix": "unimapgen_geo_current_state",
                "source_family_manifest": str(Path(args.family_manifest).resolve()),
                "summary": stage_b_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for split in args.splits:
        split = str(split)
        print(
            f"[{split}] stage_a_samples={stage_a_summary[split]['samples']} "
            f"stage_b_samples={stage_b_summary[split]['samples']}",
            flush=True,
        )
    print(f"Saved Stage A and Stage B datasets under {output_root}", flush=True)


if __name__ == "__main__":
    main()
