import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from geo_current_dataset_v1_common import ensure_dir, load_json, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller converted dataset by selecting source_sample_id units from the original raw dataset."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Input converted dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for the subset dataset.")
    parser.add_argument("--splits", type=str, nargs="*", default=[], help="Splits to subset. Defaults to auto-detect.")
    parser.add_argument("--train-source-root", type=str, default="", help="Original raw train root used to derive sample units.")
    parser.add_argument("--val-source-root", type=str, default="", help="Original raw val root used to derive sample units.")
    parser.add_argument(
        "--keep-source-sample-ids",
        type=str,
        nargs="*",
        default=[],
        help="Optional explicit source_sample_id list to keep across splits.",
    )
    parser.add_argument(
        "--source-sample-list",
        type=str,
        default="",
        help="Optional txt/json file listing source_sample_id values to keep.",
    )
    parser.add_argument(
        "--max-source-samples-per-split",
        type=int,
        default=0,
        help="Maximum number of source samples to keep per split. 0 means no explicit cap.",
    )
    parser.add_argument(
        "--source-sample-ratio",
        type=float,
        default=0.0,
        help="Optional ratio in (0,1]. Used together with or instead of --max-source-samples-per-split.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="ordered",
        choices=["ordered", "random"],
        help="How to choose samples when not using an explicit source sample list.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-root-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "none"],
        help="How to materialize images in the subset dataset.",
    )
    return parser.parse_args()


def detect_splits(input_root: Path, requested: Sequence[str]) -> List[str]:
    if requested:
        return [str(x) for x in requested]
    splits: List[str] = []
    for path in sorted(input_root.glob("*.jsonl")):
        stem = path.stem
        if stem.startswith("meta_"):
            continue
        splits.append(stem)
    return splits


def normalize_images_field(images_value) -> List[str]:
    if isinstance(images_value, list):
        out: List[str] = []
        for item in images_value:
            if isinstance(item, dict):
                text = str(item.get("path", "") or item.get("image", "") or item.get("url", "")).strip()
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(images_value, str) and images_value.strip():
        return [images_value.strip()]
    return []


def resolve_image_path(dataset_root: Path, image_path_value: str) -> Path:
    image_path = Path(str(image_path_value))
    if image_path.is_absolute():
        return image_path
    return (dataset_root / image_path).resolve(strict=False)


def build_dataset_info_generic(output_root: Path, splits: Sequence[str]) -> Dict[str, Dict]:
    base = output_root.name.strip() or "dataset"
    info: Dict[str, Dict] = {}
    for split in splits:
        info[f"{base}_{split}"] = {
            "file_name": str((output_root / f"{split}.jsonl").resolve()),
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
    return info


def rebuild_dataset_info(input_root: Path, output_root: Path, splits: Sequence[str]) -> Dict[str, Dict]:
    dataset_info_path = input_root / "dataset_info.json"
    if dataset_info_path.is_file():
        existing = load_json(dataset_info_path)
        if isinstance(existing, dict):
            rebuilt: Dict[str, Dict] = {}
            wanted = {str(split) for split in splits}
            for key, value in existing.items():
                if not isinstance(value, dict):
                    continue
                file_name = str(value.get("file_name", "")).strip()
                split_name = Path(file_name).stem if file_name else ""
                if split_name not in wanted:
                    continue
                copied = dict(value)
                copied["file_name"] = str((output_root / f"{split_name}.jsonl").resolve())
                rebuilt[str(key)] = copied
            if rebuilt:
                return rebuilt
    return build_dataset_info_generic(output_root=output_root, splits=splits)


def load_source_sample_list(path_value: str) -> List[str]:
    path = Path(str(path_value).strip())
    if not path.is_file():
        raise FileNotFoundError(f"source sample list not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        for key in ("source_sample_ids", "sample_ids", "ids", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def list_raw_sample_ids(root_value: str) -> List[str]:
    root_text = str(root_value).strip()
    if not root_text:
        return []
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw source root not found: {root}")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def determine_target_count(total: int, max_count: int, ratio: float) -> int:
    if total <= 0:
        return 0
    candidates: List[int] = [int(total)]
    if int(max_count) > 0:
        candidates.append(max(0, min(int(total), int(max_count))))
    if float(ratio) > 0.0:
        ratio_count = int(round(float(total) * float(ratio)))
        if ratio_count <= 0:
            ratio_count = 1
        candidates.append(max(0, min(int(total), int(ratio_count))))
    return min(candidates)


def choose_source_sample_ids(
    available_ids: List[str],
    explicit_ids: List[str],
    max_count: int,
    ratio: float,
    selection_mode: str,
    seed: int,
    split: str,
) -> List[str]:
    if explicit_ids:
        explicit_set = {str(x).strip() for x in explicit_ids if str(x).strip()}
        return [sample_id for sample_id in available_ids if sample_id in explicit_set]
    target_count = determine_target_count(total=len(available_ids), max_count=int(max_count), ratio=float(ratio))
    if target_count >= len(available_ids):
        return list(available_ids)
    if str(selection_mode) == "random":
        rng = random.Random(f"{int(seed)}::{split}")
        chosen = list(available_ids)
        rng.shuffle(chosen)
        chosen = chosen[: int(target_count)]
        chosen_set = set(chosen)
        return [sample_id for sample_id in available_ids if sample_id in chosen_set]
    return list(available_ids[: int(target_count)])


def copy_or_link_images(input_root: Path, output_root: Path, image_paths: Iterable[str], mode: str) -> Dict[str, int]:
    materialized = 0
    missing = 0
    mode = str(mode)
    for image_rel in sorted({str(x).strip() for x in image_paths if str(x).strip()}):
        src = resolve_image_path(input_root, image_rel)
        if not src.is_file():
            missing += 1
            continue
        if mode == "none":
            continue
        dst = output_root / Path(image_rel)
        ensure_dir(dst.parent)
        if mode == "symlink":
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(str(src), str(dst))
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        materialized += 1
    return {
        "materialized_images": int(materialized),
        "missing_images": int(missing),
        "image_root_mode": mode,
    }


def resolve_available_source_ids(split: str, meta_rows: Sequence[Dict], train_root: str, val_root: str) -> List[str]:
    if str(split) == "train":
        raw_ids = list_raw_sample_ids(train_root)
    elif str(split) == "val":
        raw_ids = list_raw_sample_ids(val_root)
    else:
        raw_ids = []
    meta_ids = {str(row.get("source_sample_id", "")).strip() for row in meta_rows if str(row.get("source_sample_id", "")).strip()}
    if raw_ids:
        return [sample_id for sample_id in raw_ids if sample_id in meta_ids]
    return sorted(meta_ids)


def process_dataset(
    input_root: Path,
    output_root: Path,
    splits: Sequence[str],
    train_source_root: str,
    val_source_root: str,
    explicit_ids: List[str],
    max_source_samples_per_split: int,
    source_sample_ratio: float,
    selection_mode: str,
    seed: int,
    image_root_mode: str,
) -> Dict[str, object]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root == input_root:
        raise ValueError("--output-root must be different from --input-root")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output root already exists and is not empty: {output_root}")
    ensure_dir(output_root)

    split_summaries: Dict[str, Dict[str, object]] = {}
    all_images: Set[str] = set()

    for split in splits:
        rows_path = input_root / f"{split}.jsonl"
        meta_path = input_root / f"meta_{split}.jsonl"
        if not rows_path.is_file():
            continue
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing meta file required for source-sample splitting: {meta_path}")

        rows = load_jsonl(rows_path)
        meta_rows = load_jsonl(meta_path)
        available_source_ids = resolve_available_source_ids(
            split=split,
            meta_rows=meta_rows,
            train_root=train_source_root,
            val_root=val_source_root,
        )
        selected_source_ids = choose_source_sample_ids(
            available_ids=available_source_ids,
            explicit_ids=explicit_ids,
            max_count=int(max_source_samples_per_split),
            ratio=float(source_sample_ratio),
            selection_mode=str(selection_mode),
            seed=int(seed),
            split=str(split),
        )
        selected_set = set(selected_source_ids)

        kept_meta_rows = [
            dict(row)
            for row in meta_rows
            if str(row.get("source_sample_id", "")).strip() in selected_set
        ]
        kept_id_set = {str(row.get("id", "")).strip() for row in kept_meta_rows if str(row.get("id", "")).strip()}
        kept_rows = [dict(row) for row in rows if str(row.get("id", "")).strip() in kept_id_set]

        write_jsonl(output_root / f"{split}.jsonl", kept_rows)
        write_jsonl(output_root / f"meta_{split}.jsonl", kept_meta_rows)

        (output_root / f"selected_source_sample_ids_{split}.txt").write_text(
            "\n".join(selected_source_ids) + ("\n" if selected_source_ids else ""),
            encoding="utf-8",
        )

        split_images: Set[str] = set()
        for row in kept_rows:
            split_images.update(normalize_images_field(row.get("images", [])))
        all_images.update(split_images)

        split_summaries[str(split)] = {
            "input_rows": int(len(rows)),
            "kept_rows": int(len(kept_rows)),
            "input_meta_rows": int(len(meta_rows)),
            "kept_meta_rows": int(len(kept_meta_rows)),
            "available_source_samples": int(len(available_source_ids)),
            "selected_source_samples": int(len(selected_source_ids)),
            "referenced_images": int(len(split_images)),
            "selection_mode": str(selection_mode),
        }

    image_summary = copy_or_link_images(
        input_root=input_root,
        output_root=output_root,
        image_paths=all_images,
        mode=image_root_mode,
    )
    dataset_info = rebuild_dataset_info(input_root=input_root, output_root=output_root, splits=splits)
    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": split_summaries,
        "selection": {
            "explicit_ids_count": int(len(explicit_ids)),
            "max_source_samples_per_split": int(max_source_samples_per_split),
            "source_sample_ratio": float(source_sample_ratio),
            "selection_mode": str(selection_mode),
            "seed": int(seed),
        },
        "totals": {
            "referenced_images": int(len(all_images)),
            **image_summary,
        },
    }
    (output_root / "subset_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    explicit_ids: List[str] = [str(x).strip() for x in args.keep_source_sample_ids if str(x).strip()]
    if str(args.source_sample_list).strip():
        explicit_ids.extend(load_source_sample_list(str(args.source_sample_list)))
    seen: Set[str] = set()
    deduped_ids: List[str] = []
    for sample_id in explicit_ids:
        if sample_id and sample_id not in seen:
            seen.add(sample_id)
            deduped_ids.append(sample_id)

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    splits = detect_splits(input_root=input_root, requested=args.splits)
    summary = process_dataset(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        train_source_root=str(args.train_source_root),
        val_source_root=str(args.val_source_root),
        explicit_ids=deduped_ids,
        max_source_samples_per_split=int(args.max_source_samples_per_split),
        source_sample_ratio=float(args.source_sample_ratio),
        selection_mode=str(args.selection_mode),
        seed=int(args.seed),
        image_root_mode=str(args.image_root_mode),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
