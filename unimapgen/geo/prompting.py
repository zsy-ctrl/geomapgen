from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .io import RasterMeta
from .schema import TaskSchema


_LANE_EXAMPLE = (
    'Example Lane.uv.geojson: '
    '{"type":"FeatureCollection","name":"Lane",'
    '"features":[{"type":"Feature","properties":{"LaneType":14},"geometry":{"type":"LineString","coordinates":[[20,40,0],[180,40,0]]}},'
    '{"type":"Feature","properties":{"LaneType":9},"geometry":{"type":"LineString","coordinates":[[36,84,0],[172,156,0]]}},'
    '{"type":"Feature","properties":{"LaneType":7},"geometry":{"type":"LineString","coordinates":[[60,24,0],[170,32,0],[220,92,0]]}}]}.'
)

_INTERSECTION_EXAMPLE = (
    'Example Intersection.uv.geojson: '
    '{"type":"FeatureCollection","name":"Intersection",'
    '"features":[{"type":"Feature","properties":{"IntersectionType":1},"geometry":{"type":"Polygon","coordinates":[[[24,24,0],[88,24,0],[88,88,0],[24,88,0],[24,24,0]]]}},'
    '{"type":"Feature","properties":{"IntersectionType":1},"geometry":{"type":"Polygon","coordinates":[[[144,80,0],[208,80,0],[208,144,0],[144,144,0],[144,80,0]]]}},'
    '{"type":"Feature","properties":{"IntersectionType":2},"geometry":{"type":"Polygon","coordinates":[[[168,168,0],[224,168,0],[232,220,0],[176,232,0],[168,168,0]]]}}]}.'
)


def _topology_constraint_text(task_key: str) -> str:
    key = str(task_key).strip().lower()
    if key == "lane":
        return (
            "Lane endpoints that meet an intersection must terminate exactly on the visible "
            "intersection boundary. Do not overshoot past the boundary and do not leave a gap."
        )
    if key == "intersection":
        return (
            "Intersection boundaries must exactly meet connected lane endpoints visible in the "
            "patch. Do not overshoot and do not leave gaps at touching locations."
        )
    return ""


def _cut_constraint_text(task_key: str) -> str:
    key = str(task_key).strip().lower()
    if key == "lane":
        return (
            "If a lane is cut by the patch boundary, the final coordinate at that side must lie exactly on the boundary cut point. "
            "Use CutIn and CutOut to mark the boundary side of the truncated endpoints, and terminate the lane at those cut points."
        )
    if key == "intersection":
        return (
            "If an intersection polygon is cut by the patch boundary, keep the polygon clipped exactly at the boundary. "
            "Use CutPoints and CutSides to mark the boundary cut points where the polygon is truncated."
        )
    return ""


def _companion_reference_text(task_key: str, companion_task_name: str, companion_geojson_text: str) -> str:
    companion_name = str(companion_task_name).strip()
    geojson_text = str(companion_geojson_text).strip()
    if not companion_name or not geojson_text:
        return ""
    key = str(task_key).strip().lower()
    if key == "lane":
        return (
            f"Reference {companion_name}.uv.geojson for the same patch: {geojson_text} "
            "Use it to keep lane endpoints exactly attached to visible intersection boundaries."
        )
    if key == "intersection":
        return (
            f"Reference {companion_name}.uv.geojson for the same patch: {geojson_text} "
            "Use it to keep intersection boundaries exactly attached to visible connected lane endpoints."
        )
    return f"Reference {companion_name}.uv.geojson for the same patch: {geojson_text}"


def _fmt_float(value: float, precision: int) -> str:
    return f"{float(value):.{max(0, int(precision))}f}"


def build_geotiff_context_text(
    raster_meta: RasterMeta | dict | None,
    crop_bbox: Optional[Sequence[int]],
    precision: int = 3,
) -> str:
    if raster_meta is None:
        return ""
    meta = raster_meta if isinstance(raster_meta, RasterMeta) else RasterMeta.from_dict(raster_meta)
    if crop_bbox is None:
        x0, y0, x1, y1 = 0, 0, int(meta.width), int(meta.height)
    else:
        x0, y0, x1, y1 = [int(v) for v in crop_bbox]
    return (
        "PatchMeta "
        f"patch_px={x0},{y0},{x1},{y1} "
        f"image_size_hint={int(max(1, x1 - x0))}x{int(max(1, y1 - y0))} "
        "uv_origin=0,0."
    )


def build_task_prompt_text(
    *,
    task_name: str = "",
    base_prompt: str,
    has_state: bool,
    with_state_suffix: str,
    without_state_suffix: str,
    raster_meta: RasterMeta | dict | None,
    crop_bbox: Optional[Sequence[int]],
    include_geospatial_context: bool = True,
    geospatial_precision: int = 3,
    companion_task_name: str = "",
    companion_geojson_text: str = "",
) -> str:
    parts = []
    if bool(include_geospatial_context):
        geo_text = build_geotiff_context_text(
            raster_meta=raster_meta,
            crop_bbox=crop_bbox,
            precision=int(geospatial_precision),
        ).strip()
        if geo_text:
            parts.append(geo_text)
    task_key = str(task_name).strip().lower()
    if task_key == "lane":
        parts.append(_LANE_EXAMPLE)
    elif task_key == "intersection":
        parts.append(_INTERSECTION_EXAMPLE)
    parts.append(str(base_prompt).strip())
    topology_text = _topology_constraint_text(task_key)
    if topology_text:
        parts.append(topology_text)
    cut_text = _cut_constraint_text(task_key)
    if cut_text:
        parts.append(cut_text)
    companion_text = _companion_reference_text(task_key, companion_task_name, companion_geojson_text)
    if companion_text:
        parts.append(companion_text)
    parts.append(
        "Output only the final GeoJSON FeatureCollection text. "
        "Do not output commentary or markdown. "
        "Do not generate CRS, Id, or RoadId fields; they are assigned after decoding."
    )
    suffix = str(with_state_suffix if has_state else without_state_suffix).strip()
    if suffix:
        parts.append(suffix)
    return " ".join(part for part in parts if part).strip()


def build_state_text(
    *,
    task_schema: TaskSchema,
    state_items: Sequence[dict],
    geojson_text: str,
) -> str:
    items = list(state_items or [])
    lines = [
        f"StateAnchorMeta task={task_schema.collection_name} anchor_count={len(items)}."
    ]
    if not items:
        lines.append("StateAnchor none.")
    else:
        for idx, item in enumerate(items):
            points = np.asarray(item.get("points_uv", []), dtype=np.float32)
            lines.append(
                f"StateAnchor idx={idx:03d} side={str(item.get('side', 'none'))} "
                f"geometry={task_schema.geometry_type} points={int(points.shape[0])}."
            )
    lines.append("StateGeoJSON:")
    lines.append(str(geojson_text or "").strip())
    return "\n".join(lines).strip()


def build_target_text(
    *,
    task_schema: TaskSchema,
    target_items: Sequence[dict],
    geojson_text: str,
) -> str:
    items = list(target_items or [])
    cut_lines = []
    for idx, item in enumerate(items):
        cut_in = str(item.get("cut_in", "none"))
        cut_out = str(item.get("cut_out", "none"))
        source = str(item.get("source", "local"))
        has_cut = cut_in != "none" or cut_out != "none" or source == "state"
        if not has_cut:
            continue
        points = np.asarray(item.get("points_uv", []), dtype=np.float32)
        rings = item.get("rings_uv") or []
        if task_schema.geometry_type == "polygon":
            cut_lines.append(
                f"CutFeature idx={idx:03d} source={source} cut_in={cut_in} cut_out={cut_out} "
                f"rings={len(rings)} outer_points={int(points.shape[0])}."
            )
        else:
            cut_lines.append(
                f"CutFeature idx={idx:03d} source={source} cut_in={cut_in} cut_out={cut_out} "
                f"points={int(points.shape[0])}."
            )
    lines = [
        f"PatchTargetMeta task={task_schema.collection_name} feature_count={len(items)} "
        f"cut_feature_count={len(cut_lines)}."
    ]
    if cut_lines:
        lines.extend(cut_lines)
    else:
        lines.append("CutFeature none.")
    lines.append("GeoJSON:")
    lines.append(str(geojson_text or "").strip())
    return "\n".join(lines).strip()
