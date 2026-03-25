import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from geo_current_dataset_v1_common import ensure_dir, load_json, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a ShareGPT-style dataset by dropping rows whose referenced images are missing."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Input dataset root containing train/val jsonl files.")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root for the cleaned dataset. Use a new directory to avoid mixing old and cleaned files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=[],
        help="Splits to clean. Defaults to auto-detecting train/val/test jsonl files under input root.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy kept images into the cleaned dataset. Default behavior is to copy kept images; this flag is accepted for clarity.",
    )
    parser.add_argument(
        "--image-root-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "none"],
        help="How to expose images under the cleaned dataset root.",
    )
    parser.add_argument(
        "--dropped-report-limit",
        type=int,
        default=200,
        help="Maximum number of dropped-row details to store per split in cleanup_summary.json.",
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
        out = [str(x).strip() for x in images_value if str(x).strip()]
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
            "columns": {
                "messages": "messages",
                "images": "images",
            },
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


def filter_rows_by_existing_images(
    rows: Sequence[Dict],
    dataset_root: Path,
    dropped_report_limit: int,
) -> Tuple[List[Dict], Set[str], Set[str], Dict[str, object]]:
    kept_rows: List[Dict] = []
    kept_ids: Set[str] = set()
    kept_image_paths: Set[str] = set()
    dropped_details: List[Dict[str, object]] = []
    missing_image_count = 0
    missing_images_by_row: Dict[str, List[str]] = {}

    for row in rows:
        row_id = str(row.get("id", "")).strip()
        image_values = normalize_images_field(row.get("images", []))
        missing_for_row: List[str] = []
        if not image_values:
            missing_for_row.append("<missing images field>")
        else:
            for image_rel in image_values:
                image_abs = resolve_image_path(dataset_root=dataset_root, image_path_value=image_rel)
                if not image_abs.is_file():
                    missing_for_row.append(str(image_rel))
        if missing_for_row:
            missing_image_count += len(missing_for_row)
            if row_id:
                missing_images_by_row[row_id] = list(missing_for_row)
            if len(dropped_details) < int(dropped_report_limit):
                dropped_details.append(
                    {
                        "id": row_id,
                        "missing_images": list(missing_for_row),
                    }
                )
            continue
        kept_rows.append(dict(row))
        if row_id:
            kept_ids.add(row_id)
        kept_image_paths.update(image_values)

    summary = {
        "input_rows": int(len(rows)),
        "kept_rows": int(len(kept_rows)),
        "dropped_rows": int(len(rows) - len(kept_rows)),
        "missing_image_references": int(missing_image_count),
        "dropped_examples": dropped_details,
        "missing_images_by_row": missing_images_by_row,
    }
    return kept_rows, kept_ids, kept_image_paths, summary


def filter_meta_rows(meta_rows: Sequence[Dict], kept_ids: Set[str]) -> List[Dict]:
    if not meta_rows:
        return []
    out: List[Dict] = []
    for row in meta_rows:
        row_id = str(row.get("id", "")).strip()
        if row_id and row_id in kept_ids:
            out.append(dict(row))
    return out


def copy_kept_images(input_root: Path, output_root: Path, image_paths: Iterable[str]) -> Dict[str, int]:
    copied = 0
    skipped_missing = 0
    seen: Set[str] = set()
    for image_rel in sorted({str(x) for x in image_paths if str(x).strip()}):
        if image_rel in seen:
            continue
        seen.add(image_rel)
        src = resolve_image_path(dataset_root=input_root, image_path_value=image_rel)
        if not src.is_file():
            skipped_missing += 1
            continue
        dst = output_root / Path(image_rel)
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        copied += 1
    return {
        "copied_images": int(copied),
        "skipped_missing_images": int(skipped_missing),
    }


def materialize_images(
    input_root: Path,
    output_root: Path,
    image_paths: Iterable[str],
    image_root_mode: str,
) -> Dict[str, object]:
    src_root = input_root / "images"
    dst_root = output_root / "images"
    mode = str(image_root_mode).strip().lower() or "copy"
    if mode == "none" or not src_root.exists():
        return {
            "image_root_mode": "none",
            "copied_images": 0,
            "skipped_missing_images": 0,
        }
    if mode == "symlink":
        try:
            dst_root.symlink_to(src_root, target_is_directory=True)
            return {
                "image_root_mode": "symlink",
                "copied_images": 0,
                "skipped_missing_images": 0,
            }
        except OSError:
            copy_summary = copy_kept_images(input_root=input_root, output_root=output_root, image_paths=image_paths)
            return {
                "image_root_mode": "copy_fallback",
                **copy_summary,
            }
    copy_summary = copy_kept_images(input_root=input_root, output_root=output_root, image_paths=image_paths)
    return {
        "image_root_mode": "copy",
        **copy_summary,
    }


def clean_dataset_root(
    input_root: Path,
    output_root: Path,
    splits: Sequence[str],
    dropped_report_limit: int,
    image_root_mode: str = "copy",
) -> Dict[str, object]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if output_root == input_root:
        raise ValueError("--output-root must be different from --input-root for safety.")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output root already exists and is not empty: {output_root}")
    ensure_dir(output_root)

    split_summaries: Dict[str, Dict[str, object]] = {}
    all_kept_images: Set[str] = set()

    for split in splits:
        rows_path = input_root / f"{split}.jsonl"
        if not rows_path.is_file():
            continue
        rows = load_jsonl(rows_path)
        kept_rows, kept_ids, kept_image_paths, row_summary = filter_rows_by_existing_images(
            rows=rows,
            dataset_root=input_root,
            dropped_report_limit=int(dropped_report_limit),
        )
        write_jsonl(output_root / f"{split}.jsonl", kept_rows)

        meta_path = input_root / f"meta_{split}.jsonl"
        kept_meta_rows: List[Dict] = []
        if meta_path.is_file():
            meta_rows = load_jsonl(meta_path)
            kept_meta_rows = filter_meta_rows(meta_rows=meta_rows, kept_ids=kept_ids)
            write_jsonl(output_root / f"meta_{split}.jsonl", kept_meta_rows)

        split_summaries[str(split)] = {
            **row_summary,
            "input_meta_rows": int(len(load_jsonl(meta_path))) if meta_path.is_file() else 0,
            "kept_meta_rows": int(len(kept_meta_rows)),
            "dropped_meta_rows": int((len(load_jsonl(meta_path)) - len(kept_meta_rows))) if meta_path.is_file() else 0,
        }
        all_kept_images.update(kept_image_paths)

    image_copy_summary = materialize_images(
        input_root=input_root,
        output_root=output_root,
        image_paths=all_kept_images,
        image_root_mode=str(image_root_mode),
    )
    dataset_info = rebuild_dataset_info(input_root=input_root, output_root=output_root, splits=splits)
    (output_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": split_summaries,
        "totals": {
            "kept_images": int(len(all_kept_images)),
            **image_copy_summary,
        },
    }
    (output_root / "cleanup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    splits = detect_splits(input_root=input_root, requested=[str(x) for x in args.splits])
    if not splits:
        raise ValueError(f"No split jsonl files found under: {input_root}")
    summary = clean_dataset_root(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        dropped_report_limit=int(args.dropped_report_limit),
        image_root_mode=str(args.image_root_mode),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
