from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from pyproj import CRS, Transformer
from rasterio import open as rasterio_open
from rasterio.transform import Affine
from rasterio.windows import Window

from .errors import wrap_geo_error
from .schema import TaskSchema
from .coord_sequence import points_abs_to_uv, points_uv_to_abs, rings_abs_to_uv, rings_uv_to_abs
from .geometry import ResizeContext


DEFAULT_GEOJSON_CRS = "urn:ogc:def:crs:OGC:1.3:CRS84"
NON_TRAINING_PROPERTY_KEYS = {"Id", "RoadId"}
CUT_METADATA_PROPERTY_KEYS = {"CutIn", "CutOut", "CutSides", "CutPoints"}
DEFAULT_TASK_ID_ORDER = ("lane", "intersection")


@dataclass
class RasterMeta:
    path: str
    width: int
    height: int
    crs: str
    transform: List[float]
    band_count: int
    dtype: str

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)

    @property
    def pixel_size_x(self) -> float:
        return float(self.transform[0])

    @property
    def pixel_size_y(self) -> float:
        return float(self.transform[4])

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "width": int(self.width),
            "height": int(self.height),
            "crs": self.crs,
            "transform": [float(x) for x in self.transform],
            "band_count": int(self.band_count),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, raw: Dict) -> "RasterMeta":
        return cls(
            path=str(raw["path"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            crs=str(raw["crs"]),
            transform=[float(x) for x in raw["transform"]],
            band_count=int(raw["band_count"]),
            dtype=str(raw["dtype"]),
        )


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1200",
            message=f"failed to read text file: {path}",
            exc=exc,
        )


def read_rgb_geotiff(
    path: str,
    band_indices: Sequence[int],
    crop_bbox: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, RasterMeta]:


    try:
        with rasterio_open(path) as ds:
            bands = [int(x) for x in band_indices]
            if crop_bbox is not None:

                x0, y0, x1, y1 = [int(v) for v in crop_bbox]
                window = Window(
                    col_off=int(x0),
                    row_off=int(y0),
                    width=int(max(1, x1 - x0)),
                    height=int(max(1, y1 - y0)),
                )
                arr = ds.read(indexes=bands, window=window)
            else:
                arr = ds.read(indexes=bands)
            image = np.transpose(arr, (1, 2, 0)).astype(np.float32)
            meta = _dataset_to_raster_meta(ds=ds, path=path)
        return image, meta
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1201",
            message=f"failed to read RGB GeoTIFF: {path}",
            exc=exc,
        )


def read_binary_mask(path: str, threshold: int = 127) -> np.ndarray:
    try:
        with rasterio_open(path) as ds:
            mask = ds.read(1)
        return (mask > int(threshold)).astype(np.uint8)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1202",
            message=f"failed to read binary review mask: {path}",
            exc=exc,
        )


def read_raster_meta(path: str) -> RasterMeta:
    try:
        with rasterio_open(path) as ds:
            return _dataset_to_raster_meta(ds=ds, path=path)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1203",
            message=f"failed to read raster metadata: {path}",
            exc=exc,
        )


def load_geojson(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1204",
            message=f"failed to load GeoJSON: {path}",
            exc=exc,
        )


def save_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text))
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1208",
            message=f"failed to save text output: {path}",
            exc=exc,
        )


def _dataset_to_raster_meta(ds, path: str) -> RasterMeta:
    return RasterMeta(
        path=str(path),
        width=int(ds.width),
        height=int(ds.height),
        crs=str(ds.crs),
        transform=[float(x) for x in tuple(ds.transform)[:6]],
        band_count=int(ds.count),
        dtype=str(ds.dtypes[0]) if ds.dtypes else "unknown",
    )


def detect_geojson_crs(geojson_dict: Dict) -> str:
    crs = geojson_dict.get("crs", {})
    props = crs.get("properties", {}) if isinstance(crs, dict) else {}
    name = props.get("name")
    if isinstance(name, str) and name.strip():
        return str(name).strip()
    return DEFAULT_GEOJSON_CRS


def _build_transformer(src_crs: str, dst_crs: str) -> Transformer:
    try:
        return Transformer.from_crs(
            CRS.from_user_input(src_crs),
            CRS.from_user_input(dst_crs),
            always_xy=True,
        )
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1205",
            message=f"failed to build CRS transformer: src={src_crs} dst={dst_crs}",
            exc=exc,
        )


def _project_coords(points_lonlat: Sequence[Sequence[float]], transformer: Transformer) -> np.ndarray:
    xy = []
    try:
        for coord in points_lonlat:
            if len(coord) < 2:
                continue
            x, y = transformer.transform(float(coord[0]), float(coord[1]))
            xy.append((x, y))
        return np.asarray(xy, dtype=np.float32)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1206",
            message="failed to project GeoJSON coordinates into raster CRS",
            exc=exc,
        )


def _world_to_pixel(points_world: np.ndarray, affine: Affine) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    inv = ~affine
    cols = []
    rows = []
    for x, y in points_world:
        col, row = inv * (float(x), float(y))
        cols.append(float(col))
        rows.append(float(row))
    return np.stack([cols, rows], axis=-1).astype(np.float32)


def pixel_to_world(points_px: np.ndarray, raster_meta: RasterMeta) -> np.ndarray:
    if points_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    affine = raster_meta.affine
    xs = []
    ys = []
    for col, row in np.asarray(points_px, dtype=np.float32):
        x, y = affine * (float(col), float(row))
        xs.append(float(x))
        ys.append(float(y))
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def geojson_to_pixel_features(
    geojson_dict: Dict,
    task_schema: TaskSchema,
    raster_meta: RasterMeta,
) -> List[Dict]:
    try:
        src_crs = detect_geojson_crs(geojson_dict)
        transformer = _build_transformer(src_crs=src_crs, dst_crs=raster_meta.crs)
        features = []
        for feature in geojson_dict.get("features", []):
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry", {})
            geometry_type = str(geometry.get("type", "")).strip().lower()
            if task_schema.geometry_type == "linestring" and geometry_type != "linestring":
                continue
            if task_schema.geometry_type == "polygon" and geometry_type != "polygon":
                continue
            coords = geometry.get("coordinates", [])
            pixel_geom = _extract_pixel_geometry(
                geometry_type=task_schema.geometry_type,
                coordinates=coords,
                transformer=transformer,
                raster_meta=raster_meta,
            )
            pixel_points = np.asarray(pixel_geom.get("points", []), dtype=np.float32)
            if pixel_points.shape[0] < task_schema.min_points_per_feature:
                continue
            record = {
                "properties": dict(feature.get("properties", {})),
                "points": pixel_points,
            }
            if task_schema.geometry_type == "polygon" and pixel_geom.get("rings"):
                record["rings"] = [np.asarray(ring, dtype=np.float32) for ring in pixel_geom.get("rings", [])]
            features.append(record)
        return features
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1207",
            message=f"failed to convert GeoJSON to pixel features for task={task_schema.name}",
            exc=exc,
        )


def _extract_pixel_geometry(
    geometry_type: str,
    coordinates,
    transformer: Transformer,
    raster_meta: RasterMeta,
) -> Dict:
    if geometry_type == "linestring":
        world = _project_coords(coordinates, transformer=transformer)
        return {"points": _world_to_pixel(world, affine=raster_meta.affine)}
    if geometry_type == "polygon":
        if not coordinates:
            return {"points": np.zeros((0, 2), dtype=np.float32), "rings": []}
        normalized_rings = _normalize_polygon_coordinate_rings(coordinates)
        rings_px: List[np.ndarray] = []
        for ring in normalized_rings:
            world = _project_coords(ring, transformer=transformer)
            if world.shape[0] >= 2 and np.allclose(world[0], world[-1]):
                world = world[:-1]
            pixel_ring = _world_to_pixel(world, affine=raster_meta.affine)
            if pixel_ring.ndim == 2 and pixel_ring.shape[0] >= 3:
                rings_px.append(pixel_ring.astype(np.float32))
        if not rings_px:
            return {"points": np.zeros((0, 2), dtype=np.float32), "rings": []}
        return {"points": rings_px[0], "rings": rings_px}
    return {"points": np.zeros((0, 2), dtype=np.float32)}


def _normalize_polygon_coordinate_rings(coordinates) -> List[Sequence[Sequence[float]]]:
    if not isinstance(coordinates, list) or not coordinates:
        return []
    first = coordinates[0]
    if _looks_like_coordinate(first):
        return [coordinates]
    out = []
    for ring in coordinates:
        if isinstance(ring, list) and ring and _looks_like_coordinate(ring[0]):
            out.append(ring)
    return out


def _looks_like_coordinate(value) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _resize_ctx_content_bounds(resize_ctx: ResizeContext) -> tuple[float, float, float, float]:
    left = float(resize_ctx.pad_x)
    top = float(resize_ctx.pad_y)
    right = float(resize_ctx.pad_x + max(1, int(resize_ctx.resized_width)) - 1)
    bottom = float(resize_ctx.pad_y + max(1, int(resize_ctx.resized_height)) - 1)
    return left, top, right, bottom


def _boundary_side_for_point_in_bounds(
    point_xy: Sequence[float],
    bounds: tuple[float, float, float, float],
    tol_px: float = 1.5,
) -> str:
    x = float(point_xy[0])
    y = float(point_xy[1])
    left, top, right, bottom = bounds
    distances = {
        "left": abs(x - left),
        "top": abs(y - top),
        "right": abs(x - right),
        "bottom": abs(y - bottom),
    }
    side = min(distances.items(), key=lambda kv: kv[1])[0]
    return side if distances[side] <= float(tol_px) else "none"


def _collect_boundary_points_in_bounds(
    points_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    tol_px: float = 1.5,
) -> tuple[np.ndarray, list[str]]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), []
    cut_points: List[np.ndarray] = []
    cut_sides: List[str] = []
    for point in pts:
        side = _boundary_side_for_point_in_bounds(point_xy=point, bounds=bounds, tol_px=tol_px)
        if side == "none":
            continue
        if not any(np.allclose(point, existing, atol=1e-3) for existing in cut_points):
            cut_points.append(np.asarray(point, dtype=np.float32))
        if side not in cut_sides:
            cut_sides.append(side)
    if not cut_points:
        return np.zeros((0, 2), dtype=np.float32), []
    return np.stack(cut_points, axis=0).astype(np.float32), list(cut_sides)


def _line_cut_metadata_for_resize_ctx(points_uv: np.ndarray, feature: Dict, resize_ctx: ResizeContext) -> Dict:
    pts = np.asarray(points_uv, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {"CutIn": "none", "CutOut": "none", "CutSides": [], "CutPoints": []}
    bounds = _resize_ctx_content_bounds(resize_ctx)
    start_side = _boundary_side_for_point_in_bounds(point_xy=pts[0], bounds=bounds, tol_px=1.5)
    end_side = _boundary_side_for_point_in_bounds(point_xy=pts[-1], bounds=bounds, tol_px=1.5)
    cut_in = start_side if start_side != "none" else ("internal" if bool(feature.get("cut_start", False)) else "none")
    cut_out = end_side if end_side != "none" else ("internal" if bool(feature.get("cut_end", False)) else "none")
    cut_points: List[List[float]] = []
    cut_sides: List[str] = []
    if cut_in != "none":
        cut_points.append([float(pts[0, 0]), float(pts[0, 1])])
        if cut_in not in {"internal"} and cut_in not in cut_sides:
            cut_sides.append(cut_in)
    if cut_out != "none":
        candidate = [float(pts[-1, 0]), float(pts[-1, 1])]
        if candidate not in cut_points:
            cut_points.append(candidate)
        if cut_out not in {"internal"} and cut_out not in cut_sides:
            cut_sides.append(cut_out)
    if not cut_points:
        boundary_points, boundary_sides = _collect_boundary_points_in_bounds(pts, bounds=bounds, tol_px=1.5)
        cut_points = [[float(point[0]), float(point[1])] for point in boundary_points]
        cut_sides = list(boundary_sides)
    return {
        "CutIn": str(cut_in),
        "CutOut": str(cut_out),
        "CutSides": cut_sides,
        "CutPoints": cut_points,
    }


def _polygon_cut_metadata_for_resize_ctx(rings_uv: Sequence[np.ndarray], feature: Dict, resize_ctx: ResizeContext) -> Dict:
    bounds = _resize_ctx_content_bounds(resize_ctx)
    cut_points: List[np.ndarray] = []
    cut_sides: List[str] = []
    for ring in rings_uv:
        ring_np = np.asarray(ring, dtype=np.float32)
        if ring_np.ndim != 2 or ring_np.shape[0] == 0:
            continue
        ring_points, ring_sides = _collect_boundary_points_in_bounds(ring_np, bounds=bounds, tol_px=1.5)
        for point in ring_points:
            if not any(np.allclose(point, existing, atol=1e-3) for existing in cut_points):
                cut_points.append(np.asarray(point, dtype=np.float32))
        for side in ring_sides:
            if side not in cut_sides:
                cut_sides.append(side)
    return {
        "CutSides": list(cut_sides),
        "CutPoints": [[float(point[0]), float(point[1])] for point in cut_points],
    }


def _append_cut_metadata_to_props(
    props: Dict,
    task_schema: TaskSchema,
    feature: Dict,
    resize_ctx: ResizeContext,
    points_uv: np.ndarray,
    rings_uv: Sequence[np.ndarray] | None,
) -> Dict:
    out = dict(props)
    if task_schema.geometry_type == "linestring":
        cut_props = _line_cut_metadata_for_resize_ctx(points_uv=points_uv, feature=feature, resize_ctx=resize_ctx)
        if cut_props["CutIn"] != "none":
            out["CutIn"] = cut_props["CutIn"]
        if cut_props["CutOut"] != "none":
            out["CutOut"] = cut_props["CutOut"]
        if cut_props["CutSides"]:
            out["CutSides"] = list(cut_props["CutSides"])
        if cut_props["CutPoints"]:
            out["CutPoints"] = [[float(pt[0]), float(pt[1]), 0.0] for pt in cut_props["CutPoints"]]
        return out
    cut_props = _polygon_cut_metadata_for_resize_ctx(
        rings_uv=list(rings_uv or []),
        feature=feature,
        resize_ctx=resize_ctx,
    )
    if cut_props["CutSides"]:
        out["CutSides"] = list(cut_props["CutSides"])
    if cut_props["CutPoints"]:
        out["CutPoints"] = [[float(pt[0]), float(pt[1]), 0.0] for pt in cut_props["CutPoints"]]
    return out


def pixel_features_to_geojson(
    task_schema: TaskSchema,
    feature_records: Sequence[Dict],
    raster_meta: RasterMeta,
    output_crs: str = DEFAULT_GEOJSON_CRS,
    include_z: bool = True,
) -> Dict:
    try:
        transformer = _build_transformer(src_crs=raster_meta.crs, dst_crs=output_crs)
        out_features = []
        for feature in feature_records:
            props = dict(feature.get("properties", {}))
            points_px = np.asarray(feature.get("points", []), dtype=np.float32)
            points_world = pixel_to_world(points_px, raster_meta=raster_meta)
            rings_world = None
            if task_schema.geometry_type == "polygon":
                raw_rings = feature.get("rings")
                if raw_rings:
                    rings_px = [np.asarray(ring, dtype=np.float32) for ring in raw_rings]
                    rings_world = [pixel_to_world(ring, raster_meta=raster_meta) for ring in rings_px]
                    rings_world = [ring for ring in rings_world if ring.shape[0] >= task_schema.min_points_per_feature]
                elif points_world.shape[0] >= task_schema.min_points_per_feature:

                    rings_world = [points_world]
                if not rings_world:
                    continue
            else:
                if points_world.shape[0] < task_schema.min_points_per_feature:
                    continue
            if task_schema.geometry_type != "polygon" and points_world.shape[0] < task_schema.min_points_per_feature:
                continue
            if task_schema.geometry_type == "polygon":
                polygon_coords = []
                for ring_world in rings_world:
                    coords = []
                    for x, y in ring_world:
                        lon, lat = transformer.transform(float(x), float(y))
                        if include_z:
                            coords.append([float(lon), float(lat), 0.0])
                        else:
                            coords.append([float(lon), float(lat)])
                    if coords and coords[0] != coords[-1]:
                        coords.append(list(coords[0]))
                    if coords:
                        polygon_coords.append(coords)
                if not polygon_coords:
                    continue
                geometry = {
                    "type": "Polygon",
                    "coordinates": polygon_coords,
                }
            else:
                coords = []
                for x, y in points_world:
                    lon, lat = transformer.transform(float(x), float(y))
                    if include_z:
                        coords.append([float(lon), float(lat), 0.0])
                    else:
                        coords.append([float(lon), float(lat)])
                geometry = {"type": "LineString", "coordinates": coords}
            out_features.append({"type": "Feature", "properties": props, "geometry": geometry})
        return {
            "type": "FeatureCollection",
            "name": task_schema.collection_name,
            "crs": {
                "type": "name",
                "properties": {"name": output_crs},
            },
            "features": out_features,
        }
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1209",
            message=f"failed to convert pixel features to GeoJSON for task={task_schema.name}",
            exc=exc,
        )


def pixel_features_to_uv_geojson(
    task_schema: TaskSchema,
    feature_records: Sequence[Dict],
    resize_ctx: ResizeContext,
    include_z: bool = True,
) -> Dict:
    try:
        out_features = []
        for feature in feature_records:
            props = dict(feature.get("properties", {}))
            points_abs = np.asarray(feature.get("points", []), dtype=np.float32)
            points_uv = points_abs_to_uv(points_abs, resize_ctx=resize_ctx)
            if task_schema.geometry_type == "polygon":
                raw_rings = feature.get("rings")
                rings_uv = []
                if raw_rings:
                    rings_uv = rings_abs_to_uv(raw_rings, resize_ctx=resize_ctx)
                elif points_uv.shape[0] >= task_schema.min_points_per_feature:
                    rings_uv = [points_uv]
                if not rings_uv:
                    continue
                props = _append_cut_metadata_to_props(
                    props=props,
                    task_schema=task_schema,
                    feature=feature,
                    resize_ctx=resize_ctx,
                    points_uv=points_uv,
                    rings_uv=rings_uv,
                )
                polygon_coords = []
                for ring_uv in rings_uv:
                    coords = []
                    for u, v in ring_uv:
                        coords.append([float(u), float(v), 0.0] if include_z else [float(u), float(v)])
                    if coords and coords[0] != coords[-1]:
                        coords.append(list(coords[0]))
                    if coords:
                        polygon_coords.append(coords)
                geometry = {"type": "Polygon", "coordinates": polygon_coords}
            else:
                if points_uv.shape[0] < task_schema.min_points_per_feature:
                    continue
                props = _append_cut_metadata_to_props(
                    props=props,
                    task_schema=task_schema,
                    feature=feature,
                    resize_ctx=resize_ctx,
                    points_uv=points_uv,
                    rings_uv=None,
                )
                coords = []
                for u, v in points_uv:
                    coords.append([float(u), float(v), 0.0] if include_z else [float(u), float(v)])
                geometry = {"type": "LineString", "coordinates": coords}
            out_features.append({"type": "Feature", "properties": props, "geometry": geometry})
        return {
            "type": "FeatureCollection",
            "name": task_schema.collection_name,
            "features": out_features,
        }
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1214",
            message=f"failed to convert pixel features to UV GeoJSON for task={task_schema.name}",
            exc=exc,
        )


def uv_geojson_to_pixel_features(
    geojson_dict: Dict,
    task_schema: TaskSchema,
    resize_ctx: ResizeContext,
) -> List[Dict]:
    try:
        features = []
        for feature in geojson_dict.get("features", []):
            if not isinstance(feature, dict):
                continue
            geometry = feature.get("geometry", {})
            geometry_type = str(geometry.get("type", "")).strip().lower()
            if task_schema.geometry_type == "linestring" and geometry_type != "linestring":
                continue
            if task_schema.geometry_type == "polygon" and geometry_type != "polygon":
                continue
            coords = geometry.get("coordinates", [])
            if task_schema.geometry_type == "polygon":
                rings_uv = _normalize_polygon_coordinate_rings(coords)
                rings_abs = []
                for ring in rings_uv:
                    ring_uv_np = np.asarray([[float(pt[0]), float(pt[1])] for pt in ring if len(pt) >= 2], dtype=np.float32)
                    if ring_uv_np.shape[0] >= 2 and np.allclose(ring_uv_np[0], ring_uv_np[-1]):
                        ring_uv_np = ring_uv_np[:-1]
                    if ring_uv_np.ndim == 2 and ring_uv_np.shape[0] >= 3:
                        rings_abs.append(points_uv_to_abs(ring_uv_np, resize_ctx=resize_ctx))
                if not rings_abs:
                    continue
                record = {
                    "properties": dict(feature.get("properties", {})),
                    "points": rings_abs[0].astype(np.float32),
                    "rings": [ring.astype(np.float32) for ring in rings_abs],
                }
            else:
                points_uv = np.asarray([[float(pt[0]), float(pt[1])] for pt in coords if len(pt) >= 2], dtype=np.float32)
                points_abs = points_uv_to_abs(points_uv, resize_ctx=resize_ctx)
                if points_abs.shape[0] < task_schema.min_points_per_feature:
                    continue
                record = {
                    "properties": dict(feature.get("properties", {})),
                    "points": points_abs.astype(np.float32),
                }
            cut_sides = feature.get("properties", {}).get("CutSides")
            cut_points = feature.get("properties", {}).get("CutPoints")
            if isinstance(cut_sides, list):
                record["cut_sides"] = [str(side) for side in cut_sides]
            if isinstance(cut_points, list):
                cut_points_uv = np.asarray(
                    [[float(pt[0]), float(pt[1])] for pt in cut_points if isinstance(pt, (list, tuple)) and len(pt) >= 2],
                    dtype=np.float32,
                )
                if cut_points_uv.ndim == 2 and cut_points_uv.shape[0] > 0:
                    record["cut_points"] = points_uv_to_abs(points_uv=cut_points_uv, resize_ctx=resize_ctx).astype(np.float32)
            cut_in = feature.get("properties", {}).get("CutIn")
            cut_out = feature.get("properties", {}).get("CutOut")
            if cut_in is not None:
                record["cut_in"] = str(cut_in)
            if cut_out is not None:
                record["cut_out"] = str(cut_out)
            features.append(record)
        return features
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1215",
            message=f"failed to convert UV GeoJSON to pixel features for task={task_schema.name}",
            exc=exc,
        )


def geojson_dumps(obj: Dict) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1210",
            message="failed to serialize GeoJSON dict to text",
            exc=exc,
        )


def geojson_dumps_compact(obj: Dict) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:
        wrap_geo_error(
            code="GEO-1213",
            message="failed to serialize compact GeoJSON text",
            exc=exc,
        )


def strip_non_training_fields_from_properties(properties: Dict) -> Dict:
    if not isinstance(properties, dict):
        return {}
    return {
        str(key): value
        for key, value in properties.items()
        if str(key) not in NON_TRAINING_PROPERTY_KEYS
    }


def strip_non_training_fields_from_feature_collection(obj: Dict, task_schema: Optional[TaskSchema] = None) -> Dict:
    collection_name = str(getattr(task_schema, "collection_name", "") or obj.get("name") or "")
    out_features = []
    for feature in obj.get("features", []) if isinstance(obj, dict) else []:
        if not isinstance(feature, dict):
            continue
        out_features.append(
            {
                "type": "Feature",
                "properties": strip_non_training_fields_from_properties(feature.get("properties", {})),
                "geometry": feature.get("geometry", {}),
            }
        )
    return {
        "type": "FeatureCollection",
        "name": collection_name,
        "features": out_features,
    }


def assign_incremental_feature_ids(task_to_geojson: Dict[str, Dict], task_order: Sequence[str] = DEFAULT_TASK_ID_ORDER) -> Dict[str, Dict]:
    ordered_task_names = []
    seen = set()
    for name in list(task_order) + sorted(task_to_geojson.keys()):
        key = str(name)
        if key in task_to_geojson and key not in seen:
            ordered_task_names.append(key)
            seen.add(key)

    next_global_id = 1
    next_lane_road_id = 1
    out: Dict[str, Dict] = {}
    for task_name in ordered_task_names:
        geojson_dict = task_to_geojson[task_name]
        collection = coerce_feature_collection(
            task_schema=TaskSchema(
                name=str(task_name),
                collection_name=str(geojson_dict.get("name") or task_name),
                geometry_type="linestring" if str(task_name).lower() == "lane" else "polygon",
                prompt_template="",
                max_features=0,
                min_points_per_feature=2 if str(task_name).lower() == "lane" else 3,
            ),
            obj=geojson_dict,
        ) or dict(geojson_dict)
        features = []
        is_lane = str(task_name).strip().lower() == "lane"
        for feature in collection.get("features", []):
            if not isinstance(feature, dict):
                continue
            props = strip_non_training_fields_from_properties(feature.get("properties", {}))
            props["Id"] = str(next_global_id)
            next_global_id += 1
            if is_lane:
                props["RoadId"] = str(next_lane_road_id)
                next_lane_road_id += 1
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": feature.get("geometry", {}),
                }
            )
        out[task_name] = {
            "type": "FeatureCollection",
            "name": str(collection.get("name") or geojson_dict.get("name") or task_name),
            "crs": {
                "type": "name",
                "properties": {"name": DEFAULT_GEOJSON_CRS},
            },
            "features": features,
        }
    return out


def extract_first_json_object(text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    raw = str(text or "")
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def coerce_feature_collection(task_schema: TaskSchema, obj: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(obj, dict):
        return None
    if str(obj.get("type", "")).strip() == "FeatureCollection" and isinstance(obj.get("features"), list):
        out = dict(obj)
        out.setdefault("name", str(task_schema.collection_name))
        out.setdefault(
            "crs",
            {
                "type": "name",
                "properties": {"name": DEFAULT_GEOJSON_CRS},
            },
        )
        return out
    if str(obj.get("type", "")).strip() == "Feature":
        return {
            "type": "FeatureCollection",
            "name": str(task_schema.collection_name),
            "crs": {
                "type": "name",
                "properties": {"name": DEFAULT_GEOJSON_CRS},
            },
            "features": [obj],
        }
    if isinstance(obj.get("features"), list):
        return {
            "type": "FeatureCollection",
            "name": str(obj.get("name") or task_schema.collection_name),
            "crs": obj.get(
                "crs",
                {
                    "type": "name",
                    "properties": {"name": DEFAULT_GEOJSON_CRS},
                },
            ),
            "features": list(obj.get("features", [])),
        }
    return None
