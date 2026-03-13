from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .errors import raise_geo_error


_ALLOWED_GEOMETRY_TYPES = {"linestring", "polygon"}


@dataclass(frozen=True)
class TaskSchema:
    name: str
    collection_name: str
    geometry_type: str
    prompt_template: str
    max_features: int
    min_points_per_feature: int


def _default_task_specs() -> Dict[str, Dict]:
    return {
        "lane": {
            "collection_name": "Lane",
            "geometry_type": "linestring",
            "prompt_template": (
                "You are given a GeoTIFF image embedding. "
                "Generate the reviewed Lane.geojson content for the current patch as valid GeoJSON."
            ),
            "max_features": 0,
            "min_points_per_feature": 2,
        },
        "intersection": {
            "collection_name": "Intersection",
            "geometry_type": "polygon",
            "prompt_template": (
                "You are given a GeoTIFF image embedding. "
                "Generate the reviewed Intersection.geojson content for the current patch as valid GeoJSON."
            ),
            "max_features": 0,
            "min_points_per_feature": 3,
        },
    }


def load_task_schemas(serialization_cfg: Dict) -> Dict[str, TaskSchema]:
    task_cfg = serialization_cfg.get("tasks")
    if not isinstance(task_cfg, dict) or not task_cfg:
        task_cfg = _default_task_specs()

    out: Dict[str, TaskSchema] = {}
    for task_name, raw in task_cfg.items():
        name = str(task_name).strip().lower()
        geometry_type = str(raw["geometry_type"]).strip().lower()
        if geometry_type not in _ALLOWED_GEOMETRY_TYPES:
            raise_geo_error("GEO-1302", f"unsupported geometry type for task {name}: {geometry_type}")
        raw_max_features = raw.get("max_features", 0)
        max_features = int(raw_max_features) if raw_max_features is not None else 0
        min_points = int(raw.get("min_points_per_feature", 2 if geometry_type == "linestring" else 3))
        out[name] = TaskSchema(
            name=name,
            collection_name=str(raw.get("collection_name", task_name)).strip(),
            geometry_type=geometry_type,
            prompt_template=str(raw["prompt_template"]).strip(),
            max_features=max_features,
            min_points_per_feature=max(2, min_points),
        )
    return out


def get_task_schema(task_schemas: Dict[str, TaskSchema], task_name: str) -> TaskSchema:
    key = str(task_name).strip().lower()
    if key not in task_schemas:
        raise_geo_error("GEO-1304", f"unknown task schema: {task_name}")
    return task_schemas[key]
