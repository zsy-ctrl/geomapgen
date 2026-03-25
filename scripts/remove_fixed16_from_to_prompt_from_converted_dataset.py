import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from geo_current_dataset_v1_common import ensure_dir, load_json, load_jsonl, write_jsonl


FROM_TO_SENTENCE_RE = re.compile(
    r"Please construct the road map from\s*\([^)]*\)\s*to\s*\([^)]*\)\s*in the satellite image\.",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove fixed16 from->to anchor text from an already converted dataset while keeping the dataset structure unchanged."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Input converted dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for the cleaned dataset.")
    parser.add_argument("--splits", type=str, nargs="*", default=[], help="Splits to process. Defaults to auto-detect.")
    parser.add_argument(
        "--image-root-mode",
        type=str,
        default="symlink",
        choices=["copy", "symlink", "none"],
        help="How to materialize images under the output dataset root.",
    )
    return parser.parse_args()


def detect_splits(input_root: Path, requested: Sequence[str]) -> List[str]:
    if requested:
        return [str(x) for x in requested]
    out: List[str] = []
    for path in sorted(input_root.glob("*.jsonl")):
        stem = path.stem
        if stem.startswith("meta_"):
            continue
        out.append(stem)
    return out


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


def strip_from_to_prompt_text(text: str) -> str:
    raw = str(text or "")
    replaced = FROM_TO_SENTENCE_RE.sub("Please construct the road map in the satellite image.", raw, count=1)
    replaced = replaced.replace("Please construct the road map in the satellite image.Please construct", "Please construct")
    return replaced


def filter_messages(messages_value) -> List[Dict]:
    if not isinstance(messages_value, list):
        return []
    out: List[Dict] = []
    for msg in messages_value:
        if not isinstance(msg, dict):
            continue
        copied = dict(msg)
        role = str(copied.get("role", copied.get("from", ""))).strip().lower()
        if role in {"user", "human"}:
            if "content" in copied:
                copied["content"] = strip_from_to_prompt_text(str(copied.get("content", "")))
            elif "value" in copied:
                copied["value"] = strip_from_to_prompt_text(str(copied.get("value", "")))
        out.append(copied)
    return out


def filter_row(row: Dict) -> Dict:
    copied = dict(row)
    if "messages" in copied:
        copied["messages"] = filter_messages(copied.get("messages", []))
    for field in ("prompt", "query", "instruction", "input"):
        if field in copied:
            copied[field] = strip_from_to_prompt_text(str(copied.get(field, "")))
    return copied


def filter_meta_row(row: Dict) -> Dict:
    copied = dict(row)
    if "prompt_text" in copied:
        copied["prompt_text"] = strip_from_to_prompt_text(str(copied.get("prompt_text", "")))
    for field in ("anchor_source", "anchor_start_xy", "anchor_end_xy", "anchor_piece_points"):
        copied.pop(field, None)
    return copied


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


def process_dataset(input_root: Path, output_root: Path, splits: Sequence[str], image_root_mode: str) -> Dict[str, object]:
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
        if not rows_path.is_file():
            continue
        rows = load_jsonl(rows_path)
        kept_rows = [filter_row(row) for row in rows]
        write_jsonl(output_root / f"{split}.jsonl", kept_rows)

        meta_path = input_root / f"meta_{split}.jsonl"
        kept_meta: List[Dict] = []
        if meta_path.is_file():
            meta_rows = load_jsonl(meta_path)
            kept_meta = [filter_meta_row(row) for row in meta_rows]
            write_jsonl(output_root / f"meta_{split}.jsonl", kept_meta)

        split_images: Set[str] = set()
        changed_rows = 0
        for old_row, new_row in zip(rows, kept_rows):
            split_images.update(normalize_images_field(new_row.get("images", [])))
            if json.dumps(old_row, ensure_ascii=False, sort_keys=True) != json.dumps(new_row, ensure_ascii=False, sort_keys=True):
                changed_rows += 1
        all_images.update(split_images)

        split_summaries[str(split)] = {
            "rows": int(len(kept_rows)),
            "meta_rows": int(len(kept_meta)),
            "changed_rows": int(changed_rows),
            "referenced_images": int(len(split_images)),
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
        "dataset_info_keys": list(dataset_info.keys()),
    }
    summary.update(image_summary)
    (output_root / "remove_fixed16_from_to_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    splits = detect_splits(input_root=input_root, requested=args.splits)
    summary = process_dataset(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        image_root_mode=str(args.image_root_mode),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
