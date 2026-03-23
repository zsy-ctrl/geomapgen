import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from export_llamafactory_patch_only_from_raw_family_manifest import clip_polyline_to_rect
from geo_current_dataset_v1_common import parse_generated_json


def merge_two_images(img1, img2, direction="horizontal"):
    """
    拼接两张图片
    Args:
        img1: 第一张图片（PIL Image对象）
        img2: 第二张图片（PIL Image对象）
        direction: 拼接方向 - horizontal(横向)/vertical(纵向)
    Returns:
        拼接后的新图片
    """
    w1, h1 = img1.size
    w2, h2 = img2.size

    if direction == "horizontal":
        new_width = w1 + w2
        new_height = max(h1, h2)
        new_img = Image.new("RGB", (new_width, new_height), "white")
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (w1, 0))
    else:
        new_width = max(w1, w2)
        new_height = h1 + h2
        new_img = Image.new("RGB", (new_width, new_height), "white")
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, h1))
    return new_img


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_path_text(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def _split_agnostic_tail(value: str) -> str:
    text = _normalize_path_text(value)
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 3 and parts[0].lower() == "images":
        return "/".join(parts[2:])
    return text


def _extract_image_ref(data: Dict) -> str:
    images = data.get("images", [])
    if isinstance(images, list):
        for item in images:
            if isinstance(item, dict):
                for key in ("path", "image", "url"):
                    value = _normalize_path_text(item.get(key, ""))
                    if value:
                        return value
            else:
                value = _normalize_path_text(item)
                if value:
                    return value
    image = data.get("image")
    if isinstance(image, dict):
        for key in ("path", "image", "url"):
            value = _normalize_path_text(image.get(key, ""))
            if value:
                return value
    if isinstance(image, str):
        return _normalize_path_text(image)
    return ""


def _candidate_roots_from_jsonl(jsonl_path: Path) -> List[Path]:
    resolved = jsonl_path.resolve()
    out: List[Path] = []
    for path in [resolved.parent, *resolved.parents]:
        if path not in out:
            out.append(path)
    cwd = Path.cwd().resolve()
    if cwd not in out:
        out.append(cwd)
    return out


def _resolve_image_path(jsonl_path: Path, image_ref: str) -> str:
    ref = _normalize_path_text(image_ref)
    if not ref:
        return ""
    direct = Path(ref)
    if direct.is_file():
        return str(direct.resolve())

    roots = _candidate_roots_from_jsonl(jsonl_path)
    rel_path = Path(ref)
    for root in roots:
        candidate = (root / rel_path).resolve()
        if candidate.is_file():
            return str(candidate)

    splitless_tail = _split_agnostic_tail(ref)
    if splitless_tail:
        tail_path = Path(splitless_tail)
        for root in roots:
            images_root = root / "images"
            if not images_root.is_dir():
                continue
            for split_dir in images_root.iterdir():
                if not split_dir.is_dir():
                    continue
                candidate = (split_dir / tail_path).resolve()
                if candidate.is_file():
                    return str(candidate)
    return ""


def _decode_json_payload(value) -> Optional[object]:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    current = text
    for _ in range(3):
        try:
            parsed = json.loads(current)
        except Exception:
            parsed, _ = parse_generated_json(current)
        if isinstance(parsed, (dict, list)):
            return parsed
        if isinstance(parsed, str):
            current = parsed.strip()
            if not current:
                return None
            continue
        break
    return None


def _coerce_line_dicts(lines: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for line in list(lines or []):
        if not isinstance(line, dict):
            continue
        points = line.get("points", [])
        if not isinstance(points, list):
            continue
        clean_points: List[List[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                clean_points.append([float(point[0]), float(point[1])])
            except Exception:
                continue
        if len(clean_points) < 2:
            continue
        start_type = str(line.get("start_type", "start")).strip().lower() or "start"
        end_type = str(line.get("end_type", "end")).strip().lower() or "end"
        if start_type not in {"start", "cut", "closed"}:
            start_type = "start"
        if end_type not in {"end", "cut", "closed"}:
            end_type = "end"
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "geometry_type": str(line.get("geometry_type", "line")),
                "start_type": start_type,
                "end_type": end_type,
                "points": clean_points,
            }
        )
    return out


def _extract_lines_from_payload(payload) -> List[Dict]:
    if isinstance(payload, dict):
        if isinstance(payload.get("lines"), list):
            return _coerce_line_dicts(payload.get("lines", []))
        return []
    if isinstance(payload, list):
        return _coerce_line_dicts(payload)
    return []


def _extract_pred_lines(data: Dict) -> List[Dict]:
    for key in ("response", "predict", "prediction", "pred_text", "generated_text", "output", "text"):
        if key not in data:
            continue
        lines = _extract_lines_from_payload(_decode_json_payload(data.get(key)))
        if lines:
            return lines
    for msg in data.get("messages", []):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        lines = _extract_lines_from_payload(_decode_json_payload(msg.get("content", "")))
        if lines:
            return lines
    return []


def _extract_gt_lines(data: Dict) -> List[Dict]:
    for key in ("labels", "label", "ground_truth", "target", "gt"):
        if key not in data:
            continue
        lines = _extract_lines_from_payload(_decode_json_payload(data.get(key)))
        if lines:
            return lines
    return []


def _extract_user_prompt(data: Dict) -> str:
    messages = data.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", msg.get("from", ""))).strip().lower()
            if role not in {"user", "human"}:
                continue
            content = str(msg.get("content", msg.get("value", ""))).strip()
            if content:
                return content
    for key in ("prompt", "query", "instruction", "input"):
        value = str(data.get(key, "")).strip()
        if value:
            return value
    return ""


def _extract_target_box(prompt: str) -> Optional[Tuple[float, float, float, float]]:
    text = str(prompt or "")
    match = re.search(
        r"target\s+box\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return tuple(float(match.group(idx)) for idx in range(1, 5))


def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _endpoint_color(endpoint_type: str) -> Tuple[int, int, int]:
    endpoint_type = str(endpoint_type).strip().lower()
    if endpoint_type == "cut":
        return (255, 60, 60)
    if endpoint_type == "end":
        return (255, 220, 0)
    return (0, 255, 0)


def _spread_endpoint_positions(
    visual_lines: List[Dict],
    cluster_dist_px: float = 14.0,
    base_offset_px: float = 14.0,
) -> Dict[Tuple[int, str], Tuple[float, float]]:
    endpoints: List[Dict] = []
    for line_idx, row in enumerate(visual_lines):
        points = row.get("points", [])
        if str(row.get("geometry_type", "line")) == "polygon" or len(points) < 2:
            continue
        endpoints.append(
            {
                "line_idx": int(line_idx),
                "endpoint_key": "start",
                "point": (float(points[0][0]), float(points[0][1])),
            }
        )
        endpoints.append(
            {
                "line_idx": int(line_idx),
                "endpoint_key": "end",
                "point": (float(points[-1][0]), float(points[-1][1])),
            }
        )

    clusters: List[List[Dict]] = []
    for endpoint in endpoints:
        assigned = False
        for cluster in clusters:
            if _distance_xy(endpoint["point"], cluster[0]["point"]) <= float(cluster_dist_px):
                cluster.append(endpoint)
                assigned = True
                break
        if not assigned:
            clusters.append([endpoint])

    out: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            item = cluster[0]
            out[(int(item["line_idx"]), str(item["endpoint_key"]))] = item["point"]
            continue
        ordered = sorted(
            cluster,
            key=lambda item: (
                float(item["point"][0]) * float(item["point"][0]) + float(item["point"][1]) * float(item["point"][1]),
                float(item["point"][1]),
                float(item["point"][0]),
            ),
        )
        count = len(ordered)
        angle_step = (2.0 * math.pi) / float(max(1, count))
        radius = float(base_offset_px) + max(0.0, float(count - 2) * 2.0)
        for idx, item in enumerate(ordered):
            angle = (math.pi / 2.0) + (float(idx) * angle_step)
            px = float(item["point"][0]) + radius * math.cos(angle)
            py = float(item["point"][1]) + radius * math.sin(angle)
            out[(int(item["line_idx"]), str(item["endpoint_key"]))] = (px, py)
    return out


def _select_arrow_segment(xy: List[Tuple[float, float]]) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if len(xy) < 2:
        return None
    total = 0.0
    seg_lengths: List[float] = []
    for idx in range(len(xy) - 1):
        length = _distance_xy(xy[idx], xy[idx + 1])
        seg_lengths.append(length)
        total += length
    if total <= 1e-6:
        return None
    target = total * 0.6
    accum = 0.0
    for idx, length in enumerate(seg_lengths):
        next_accum = accum + length
        if length > 1.0 and target <= next_accum:
            return xy[idx], xy[idx + 1]
        accum = next_accum
    for idx in range(len(seg_lengths) - 1, -1, -1):
        if seg_lengths[idx] > 1.0:
            return xy[idx], xy[idx + 1]
    return None


def _draw_direction_arrow(draw: ImageDraw.ImageDraw, xy: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
    segment = _select_arrow_segment(xy)
    if segment is None:
        return
    start_xy, end_xy = segment
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        return
    ux = dx / seg_len
    uy = dy / seg_len
    arrow_tip = (
        float(start_xy[0]) + 0.72 * dx,
        float(start_xy[1]) + 0.72 * dy,
    )
    arrow_base = (
        float(arrow_tip[0]) - 10.0 * ux,
        float(arrow_tip[1]) - 10.0 * uy,
    )
    perp = (-uy, ux)
    wing1 = (
        float(arrow_base[0]) + 4.0 * perp[0],
        float(arrow_base[1]) + 4.0 * perp[1],
    )
    wing2 = (
        float(arrow_base[0]) - 4.0 * perp[0],
        float(arrow_base[1]) - 4.0 * perp[1],
    )
    draw.line((arrow_base, arrow_tip), fill=color, width=3)
    draw.polygon([arrow_tip, wing1, wing2], fill=color)


def _clip_points_to_canvas(points: Sequence[Sequence[float]], width: int, height: int) -> List[List[float]]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        return []
    rect = (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
    clipped = clip_polyline_to_rect(arr, rect)
    if not clipped:
        return []
    merged: List[List[float]] = []
    for piece in clipped:
        for idx, point in enumerate(np.asarray(piece, dtype=np.float32)):
            if merged and idx == 0 and _distance_xy(tuple(merged[-1]), (float(point[0]), float(point[1]))) <= 1e-3:
                continue
            merged.append([float(point[0]), float(point[1])])
    return merged


def _build_visual_lines(lines: List[Dict], width: int, height: int) -> List[Dict]:
    out: List[Dict] = []
    for line in lines:
        geometry_type = str(line.get("geometry_type", "line"))
        points = line.get("points", [])
        if geometry_type == "polygon":
            clean_points = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                clean_points.append([float(point[0]), float(point[1])])
            if len(clean_points) >= 3:
                out.append(
                    {
                        "category": str(line.get("category", "intersection_polygon")),
                        "geometry_type": "polygon",
                        "start_type": "closed",
                        "end_type": "closed",
                        "points": clean_points,
                    }
                )
            continue
        clipped_points = _clip_points_to_canvas(points, width=width, height=height)
        if len(clipped_points) < 2:
            continue
        out.append(
            {
                "category": str(line.get("category", "lane_line")),
                "geometry_type": "line",
                "start_type": str(line.get("start_type", "start")),
                "end_type": str(line.get("end_type", "end")),
                "points": clipped_points,
            }
        )
    return out


def _draw_target_box(draw: ImageDraw.ImageDraw, target_box, font: ImageFont.ImageFont) -> None:
    if target_box is None:
        return
    x0, y0, x1, y1 = [float(v) for v in target_box]
    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=2)
    draw.text((x0 + 4, max(0.0, y0 - 12.0)), "target_box", fill=(255, 255, 255), font=font)


def _draw_overlay_image(
    raw_img: Image.Image,
    visual_lines: List[Dict],
    line_color: Tuple[int, int, int],
    title: str,
    target_box=None,
) -> Image.Image:
    image = raw_img.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    endpoint_positions = _spread_endpoint_positions(visual_lines)

    _draw_target_box(draw, target_box=target_box, font=font)
    draw.rectangle((0, 0, width - 1, 18), fill=(0, 0, 0))
    draw.text((6, 3), title, fill=(255, 255, 255), font=font)

    for line_idx, row in enumerate(visual_lines):
        points = row.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            continue
        geometry_type = str(row.get("geometry_type", "line"))
        xy = [(float(pt[0]), float(pt[1])) for pt in points]
        if geometry_type == "polygon" and len(xy) >= 3:
            draw.polygon(xy, outline=line_color, width=3)
            continue

        draw.line(xy, fill=line_color, width=4)
        _draw_direction_arrow(draw, xy, line_color)

        start_type = str(row.get("start_type", "start")).strip() or "start"
        end_type = str(row.get("end_type", "end")).strip() or "end"
        start_xy = xy[0]
        end_xy = xy[-1]
        start_display = endpoint_positions.get((int(line_idx), "start"), start_xy)
        end_display = endpoint_positions.get((int(line_idx), "end"), end_xy)
        if _distance_xy(start_display, start_xy) > 1.0:
            draw.line((start_xy, start_display), fill=_endpoint_color(start_type), width=2)
        if _distance_xy(end_display, end_xy) > 1.0:
            draw.line((end_xy, end_display), fill=_endpoint_color(end_type), width=2)
        radius = 5
        draw.ellipse(
            (
                float(start_display[0]) - radius,
                float(start_display[1]) - radius,
                float(start_display[0]) + radius,
                float(start_display[1]) + radius,
            ),
            fill=_endpoint_color(start_type),
            outline=(0, 0, 0),
        )
        draw.ellipse(
            (
                float(end_display[0]) - radius,
                float(end_display[1]) - radius,
                float(end_display[0]) + radius,
                float(end_display[1]) + radius,
            ),
            fill=_endpoint_color(end_type),
            outline=(0, 0, 0),
        )
        draw.text(
            (float(start_display[0]) + 6.0, float(start_display[1]) - 10.0),
            start_type,
            fill=_endpoint_color(start_type),
            font=font,
        )
        draw.text(
            (float(end_display[0]) + 6.0, float(end_display[1]) - 10.0),
            end_type,
            fill=_endpoint_color(end_type),
            font=font,
        )
    return image


def process_and_draw(jsonl_file, output_dir="result", save_single=True):
    """
    处理逻辑：分图绘制+拼接
    Args:
        jsonl_file: 输入jsonl文件路径
        output_dir: 输出目录
        save_single: 是否保存单独的预测/真值图（True/False）
    """
    jsonl_path = Path(jsonl_file).resolve()
    output_root = Path(output_dir).resolve()
    _ensure_dir(output_root)
    if save_single:
        _ensure_dir(output_root / "pred")
        _ensure_dir(output_root / "gt")

    summary: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            print(f"\n===== 处理第 {index} 行 =====")
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ 第{index}行 JSON 解析错误: {e}")
                continue

            try:
                image_ref = _extract_image_ref(data)
                resolved_image_path = _resolve_image_path(jsonl_path=jsonl_path, image_ref=image_ref)
                if not resolved_image_path:
                    print(f"警告: 无法解析原始图像路径 → {image_ref}，跳过该帧")
                    continue

                try:
                    raw_img = Image.open(resolved_image_path).convert("RGB")
                except Exception as e:
                    print(f"错误: 打开图像失败 → {e}，跳过该帧")
                    continue

                img_width, img_height = raw_img.size
                prompt_text = _extract_user_prompt(data)
                target_box = _extract_target_box(prompt_text)
                pred_lines = _extract_pred_lines(data)
                gt_lines = _extract_gt_lines(data)
                pred_visual_lines = _build_visual_lines(pred_lines, width=img_width, height=img_height)
                gt_visual_lines = _build_visual_lines(gt_lines, width=img_width, height=img_height)

                print(f"成功打开图像: {resolved_image_path} (尺寸: {img_width}x{img_height})")
                print(f"预测值: 共解析到 {len(pred_lines)} 条线，可视化有效 {len(pred_visual_lines)} 条")
                print(f"真值: 共解析到 {len(gt_lines)} 条线，可视化有效 {len(gt_visual_lines)} 条")

                img_pred = _draw_overlay_image(
                    raw_img=raw_img,
                    visual_lines=pred_visual_lines,
                    line_color=(255, 0, 0),
                    title="Prediction",
                    target_box=target_box,
                )
                img_gt = _draw_overlay_image(
                    raw_img=raw_img,
                    visual_lines=gt_visual_lines,
                    line_color=(0, 90, 255),
                    title="GroundTruth",
                    target_box=target_box,
                )

                base_name = Path(resolved_image_path).stem
                row_id = str(data.get("id", "")).strip()
                name_prefix = f"{index:04d}_{row_id or base_name}"
                ext = ".png"

                if save_single:
                    pred_save_path = output_root / "pred" / f"{name_prefix}_pred{ext}"
                    gt_save_path = output_root / "gt" / f"{name_prefix}_gt{ext}"
                    img_pred.save(pred_save_path)
                    img_gt.save(gt_save_path)
                    print(f"单独预测图保存: {pred_save_path}")
                    print(f"单独真值图保存: {gt_save_path}")

                combined_img = merge_two_images(img_pred, img_gt, direction="horizontal")
                combined_save_path = output_root / f"{name_prefix}_combined{ext}"
                combined_img.save(combined_save_path)
                print(f"✅ 拼接图保存完成: {combined_save_path}")

                summary.append(
                    {
                        "row_index": int(index),
                        "id": row_id,
                        "image_ref": image_ref,
                        "resolved_image_path": resolved_image_path,
                        "prompt_text": prompt_text,
                        "target_box": list(target_box) if target_box is not None else [],
                        "num_pred_lines": int(len(pred_lines)),
                        "num_gt_lines": int(len(gt_lines)),
                        "num_pred_drawn": int(len(pred_visual_lines)),
                        "num_gt_drawn": int(len(gt_visual_lines)),
                        "combined_image": str(combined_save_path),
                    }
                )
            except Exception as e:
                print(f"❌ 第{index}行 处理失败: {e}")

    summary_path = output_root / "visualize_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n===== 所有帧处理完成，summary: {summary_path} =====")


if __name__ == "__main__":
    # ========== 配置项（修改为你的实际路径） ==========
    INPUT_JSONL = ""
    SAVE_SINGLE_IMAGES = True
    OUTPUT_DIR = "result"
    # ================================================
    if not str(INPUT_JSONL).strip():
        raise SystemExit("请先在脚本底部填写 INPUT_JSONL")
    process_and_draw(INPUT_JSONL, output_dir=OUTPUT_DIR, save_single=SAVE_SINGLE_IMAGES)
