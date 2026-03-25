import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from geo_current_dataset_v1_common import ensure_dir, load_json, load_jsonl, parse_generated_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lane-only copy from an already converted ShareGPT-style dataset by removing all intersection_polygon fields."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Input converted dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for the lane-only dataset copy.")
    parser.add_argument("--splits", type=str, nargs="*", default=[], help="Splits to process. Defaults to auto-detect.")
    parser.add_argument(
        "--image-root-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "none"],
        help="How to materialize images in the output dataset.",
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


def is_intersection_line(line: Dict) -> bool:
    return str(line.get("category", "")).strip() == "intersection_polygon"


def filter_line_list(lines_value) -> List[Dict]:
    if not isinstance(lines_value, list):
        return []
    out: List[Dict] = []
    for line in lines_value:
        if not isinstance(line, dict):
            continue
        if is_intersection_line(line):
            continue
        out.append(dict(line))
    return out


def normalize_system_text(text: str) -> str:
    out = str(text or "")
    out = out.replace(
        "Use category lane_line for roads and intersection_polygon for intersections.\n",
        "Use category lane_line for roads.\n",
    )
    out = out.replace(
        "Use category lane_line for roads and intersection_polygon for intersections.",
        "Use category lane_line for roads.",
    )
    return out


def filter_json_like_value(value):
    if isinstance(value, dict):
        copied = dict(value)
        if isinstance(copied.get("lines"), list):
            copied["lines"] = filter_line_list(copied.get("lines", []))
        return copied
    if isinstance(value, list):
        return filter_line_list(value)
    text = str(value or "")
    stripped = text.strip()
    if not stripped:
        return text
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("lines"), list):
        return json.dumps({"lines": filter_line_list(parsed.get("lines", []))}, ensure_ascii=False, separators=(",", ":"))
    if isinstance(parsed, list):
        return json.dumps(filter_line_list(parsed), ensure_ascii=False, separators=(",", ":"))

    parsed_obj, cleaned = parse_generated_json(text)
    if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("lines"), list) and cleaned:
        replacement = json.dumps({"lines": filter_line_list(parsed_obj.get("lines", []))}, ensure_ascii=False, separators=(",", ":"))
        return text.replace(cleaned, replacement, 1)
    if isinstance(parsed_obj, list) and cleaned:
        replacement = json.dumps(filter_line_list(parsed_obj), ensure_ascii=False, separators=(",", ":"))
        return text.replace(cleaned, replacement, 1)
    return text


def filter_messages(messages_value) -> List[Dict]:
    if not isinstance(messages_value, list):
        return []
    out: List[Dict] = []
    for msg in messages_value:
        if not isinstance(msg, dict):
            continue
        copied = dict(msg)
        role = str(copied.get("role", copied.get("from", ""))).strip().lower()
        content = copied.get("content", copied.get("value", ""))
        if role == "system":
            new_content = normalize_system_text(str(content))
        else:
            new_content = filter_json_like_value(content)
        if "content" in copied:
            copied["content"] = new_content
        elif "value" in copied:
            copied["value"] = new_content
        out.append(copied)
    return out


def filter_row(row: Dict) -> Dict:
    copied = dict(row)
    if "messages" in copied:
        copied["messages"] = filter_messages(copied.get("messages", []))
    for key in ("labels", "label", "ground_truth", "target", "gt", "response", "predict", "prediction", "pred_text", "generated_text", "output", "assistant", "text"):
        if key in copied:
            copied[key] = filter_json_like_value(copied.get(key))
    return copied


def filter_meta_row(row: Dict) -> Dict:
    copied = dict(row)
    line_fields = (
        "target_lines",
        "target_lines_float",
        "target_lines_quantized",
        "state_lines",
        "state_lines_float",
        "pred_lines",
        "prediction_lines",
    )
    for field in line_fields:
        if field in copied:
            copied[field] = filter_line_list(copied.get(field, []))
    if "num_target_lines" in copied:
        target_value = copied.get("target_lines")
        if not isinstance(target_value, list):
            target_value = copied.get("target_lines_float", [])
        copied["num_target_lines"] = int(len(target_value)) if isinstance(target_value, list) else 0
    if "num_state_lines" in copied:
        state_value = copied.get("state_lines")
        if not isinstance(state_value, list):
            state_value = copied.get("state_lines_float", [])
        copied["num_state_lines"] = int(len(state_value)) if isinstance(state_value, list) else 0
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
        else:
            meta_rows = []

        split_images: Set[str] = set()
        for row in kept_rows:
            split_images.update(normalize_images_field(row.get("images", [])))
        all_images.update(split_images)

        original_intersections = 0
        remaining_intersections = 0
        for row in meta_rows:
            original_intersections += sum(1 for line in row.get("target_lines", []) if is_intersection_line(line))
            original_intersections += sum(1 for line in row.get("state_lines", []) if is_intersection_line(line))
        for row in kept_meta:
            remaining_intersections += sum(1 for line in row.get("target_lines", []) if is_intersection_line(line))
            remaining_intersections += sum(1 for line in row.get("state_lines", []) if is_intersection_line(line))

        split_summaries[str(split)] = {
            "rows": int(len(kept_rows)),
            "meta_rows": int(len(kept_meta)),
            "referenced_images": int(len(split_images)),
            "original_intersection_records": int(original_intersections),
            "remaining_intersection_records": int(remaining_intersections),
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
        "totals": {
            "referenced_images": int(len(all_images)),
            **image_summary,
        },
    }
    (output_root / "remove_intersection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    splits = detect_splits(input_root=input_root, requested=args.splits)
    summary = process_dataset(
        input_root=input_root,
        output_root=output_root,
        splits=splits,
        image_root_mode=str(args.image_root_mode),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
