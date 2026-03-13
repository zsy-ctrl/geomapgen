from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .io import RasterMeta, pixel_to_world
from .schema import TaskSchema


_LANE_EXAMPLE = (
    'Example Lane.geojson: '
    '{"type":"FeatureCollection","name":"Lane","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},'
    '"features":[{"type":"Feature","properties":{"Id":"L1","LaneType":14},"geometry":{"type":"LineString","coordinates":[[116.1001,39.9001,0],[116.1006,39.9001,0]]}},'
    '{"type":"Feature","properties":{"Id":"L2","LaneType":9},"geometry":{"type":"LineString","coordinates":[[116.1002,39.9003,0],[116.1007,39.9006,0]]}},'
    '{"type":"Feature","properties":{"Id":"L3","LaneType":7},"geometry":{"type":"LineString","coordinates":[[116.1003,39.8998,0],[116.1009,39.8999,0],[116.1012,39.9002,0]]}}]}.'
)

_INTERSECTION_EXAMPLE = (
    'Example Intersection.geojson: '
    '{"type":"FeatureCollection","name":"Intersection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},'
    '"features":[{"type":"Feature","properties":{"Id":"I1","IntersectionType":1},"geometry":{"type":"Polygon","coordinates":[[[116.2001,39.8001,0],[116.2005,39.8001,0],[116.2005,39.8005,0],[116.2001,39.8005,0],[116.2001,39.8001,0]]]}},'
    '{"type":"Feature","properties":{"Id":"I2","IntersectionType":1},"geometry":{"type":"Polygon","coordinates":[[[116.2010,39.8010,0],[116.2014,39.8010,0],[116.2014,39.8014,0],[116.2010,39.8014,0],[116.2010,39.8010,0]]]}},'
    '{"type":"Feature","properties":{"Id":"I3","IntersectionType":2},"geometry":{"type":"Polygon","coordinates":[[[116.2020,39.8020,0],[116.2024,39.8020,0],[116.2025,39.8024,0],[116.2021,39.8025,0],[116.2020,39.8020,0]]]}}]}.'
)


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
    corners_px = np.asarray(
        [
            [float(x0), float(y0)],
            [float(x1), float(y0)],
            [float(x0), float(y1)],
            [float(x1), float(y1)],
        ],
        dtype=np.float32,
    )
    corners_world = pixel_to_world(points_px=corners_px, raster_meta=meta)
    xmin = float(np.min(corners_world[:, 0]))
    xmax = float(np.max(corners_world[:, 0]))
    ymin = float(np.min(corners_world[:, 1]))
    ymax = float(np.max(corners_world[:, 1]))
    crs_text = str(meta.crs or "unknown").replace(" ", "_")
    origin_x = float(meta.transform[2])
    origin_y = float(meta.transform[5])
    return (
        "GeoMeta "
        f"crs={crs_text} "
        f"origin={_fmt_float(origin_x, precision)},{_fmt_float(origin_y, precision)} "
        f"pixel_size={_fmt_float(meta.pixel_size_x, precision)},{_fmt_float(meta.pixel_size_y, precision)} "
        f"patch_px={x0},{y0},{x1},{y1} "
        f"patch_world={_fmt_float(xmin, precision)},{_fmt_float(ymin, precision)},{_fmt_float(xmax, precision)},{_fmt_float(ymax, precision)}."
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
    parts.append(
        "Output only the final GeoJSON FeatureCollection text. "
        "Do not output StateAnchorMeta, PatchTargetMeta, CutFeature, commentary, or markdown."
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
