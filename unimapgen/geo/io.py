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
