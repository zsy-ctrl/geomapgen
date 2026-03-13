from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .io import RasterMeta, pixel_to_world


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
    parts.append(str(base_prompt).strip())
    suffix = str(with_state_suffix if has_state else without_state_suffix).strip()
    if suffix:
        parts.append(suffix)
    return " ".join(part for part in parts if part).strip()
