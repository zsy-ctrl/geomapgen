from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc

from unimapgen.models.hf_utils import resolve_hf_snapshot_path

from .coord_sequence import relaxed_parse_props_json
from .errors import raise_geo_error
from .schema import TaskSchema


class GeoMapGrammarHelper:
    def __init__(self, tokenizer: "GeoCoordTokenizer", task_schema: TaskSchema, max_prop_tokens: int = 128) -> None:
        self.tokenizer = tokenizer
        self.task_schema = task_schema
        self.max_prop_tokens = int(max_prop_tokens)
        self.category_qwen_ids = [
            int(tokenizer.obj_token_id),
            int(tokenizer.obj_end_token_id),
            int(tokenizer.pt_token_id),
            int(tokenizer.ring_token_id),
            int(tokenizer.prop_token_id),
            int(tokenizer.prop_end_token_id),
            int(tokenizer.line_token_id),
            int(tokenizer.poly_token_id),
        ]

    def valid_next_qwen_map_ids(
        self,
        generated_qwen_ids: Sequence[int],
        min_points_per_line: int = 2,
        max_lines: int = None,
    ) -> List[int]:
        ids = [int(x) for x in generated_qwen_ids]
        tok = self.tokenizer
        geom_token_id = int(tok.line_token_id if self.task_schema.geometry_type == "linestring" else tok.poly_token_id)
        src_ids = [int(v) for v in tok.src_token_ids.values()]
        cut_in_ids = [int(v) for v in tok.cut_in_token_ids.values()]
        cut_out_ids = [int(v) for v in tok.cut_out_token_ids.values()]
        coord_ids = [int(x) for x in tok.coord_token_ids]
        base_text_ids = [int(x) for x in range(int(tok.base_vocab_size))]
        min_points = max(int(self.task_schema.min_points_per_feature), int(min_points_per_line))
        max_objects = None if max_lines is None else max(1, int(max_lines))

        if not ids:
            return [int(tok.map_bos_token_id)]
        if ids[0] != int(tok.map_bos_token_id):
            return [int(tok.map_bos_token_id)]

        state = "after_map_bos"
        ring_point_count = 0
        object_count = 0
        prop_token_count = 0
        for token_id in ids[1:]:
            token_id = int(token_id)
            if state == "after_map_bos":
                if token_id == int(tok.obj_token_id):
                    object_count += 1
                    state = "after_obj"
                else:
                    return self._after_map_bos_allowed(max_objects=max_objects, object_count=object_count)
            elif state == "after_obj":
                if token_id == geom_token_id:
                    state = "after_geometry"
                else:
                    return [geom_token_id]
            elif state == "after_geometry":
                if token_id in src_ids:
                    state = "after_src"
                else:
                    return src_ids
            elif state == "after_src":
                if token_id in cut_in_ids:
                    state = "after_cut_in"
                else:
                    return cut_in_ids
            elif state == "after_cut_in":
                if token_id in cut_out_ids:
                    state = "after_cut_out"
                else:
                    return cut_out_ids
            elif state == "after_cut_out":
                if token_id == int(tok.pt_token_id):
                    state = "after_pt_token"
                else:
                    return [int(tok.pt_token_id)]
            elif state == "after_pt_token":
                if token_id in coord_ids:
                    state = "after_pt_x"
                else:
                    return coord_ids
            elif state == "after_pt_x":
                if token_id in coord_ids:
                    ring_point_count += 1
                    state = "after_point"
                else:
                    return coord_ids
            elif state == "after_point":
                allowed = [int(tok.pt_token_id)]
                if ring_point_count >= int(min_points):
                    if self.task_schema.geometry_type == "polygon":
                        allowed.append(int(tok.ring_token_id))
                    allowed.append(int(tok.prop_token_id))
                if token_id == int(tok.pt_token_id):
                    state = "after_pt_token"
                elif token_id == int(tok.ring_token_id) and self.task_schema.geometry_type == "polygon" and ring_point_count >= int(min_points):
                    state = "after_ring"
                    ring_point_count = 0
                elif token_id == int(tok.prop_token_id) and ring_point_count >= int(min_points):
                    state = "in_prop"
                    prop_token_count = 0
                else:
                    return allowed
            elif state == "after_ring":
                if token_id == int(tok.pt_token_id):
                    state = "after_pt_token"
                else:
                    return [int(tok.pt_token_id)]
            elif state == "in_prop":
                if token_id == int(tok.prop_end_token_id):
                    state = "after_prop_end"
                    prop_token_count = 0
                elif token_id < int(tok.base_vocab_size):
                    prop_token_count += 1
                    continue
                else:
                    return self._prop_allowed(base_text_ids=base_text_ids, prop_token_count=prop_token_count)
            elif state == "after_prop_end":
                if token_id == int(tok.obj_end_token_id):
                    state = "after_map_bos"
                    ring_point_count = 0
                else:
                    return [int(tok.obj_end_token_id)]

        if state == "after_map_bos":
            return self._after_map_bos_allowed(max_objects=max_objects, object_count=object_count)
        if state == "after_obj":
            return [geom_token_id]
        if state == "after_geometry":
            return src_ids
        if state == "after_src":
            return cut_in_ids
        if state == "after_cut_in":
            return cut_out_ids
        if state == "after_cut_out":
            return [int(tok.pt_token_id)]
        if state == "in_prop":
            return self._prop_allowed(base_text_ids=base_text_ids, prop_token_count=prop_token_count)
        if state == "after_pt_token":
            return coord_ids
        if state == "after_pt_x":
            return coord_ids
        if state == "after_point":
            allowed = [int(tok.pt_token_id)]
            if ring_point_count >= int(min_points):
                if self.task_schema.geometry_type == "polygon":
                    allowed.append(int(tok.ring_token_id))
                allowed.append(int(tok.prop_token_id))
            return allowed
        if state == "after_ring":
            return [int(tok.pt_token_id)]
        if state == "after_prop_end":
            return [int(tok.obj_end_token_id)]
        return [int(tok.eos_token_id)]

    def _after_map_bos_allowed(self, max_objects: int | None, object_count: int) -> List[int]:
        allowed = [int(self.tokenizer.eos_token_id)]
        if max_objects is None or int(object_count) < int(max_objects):
            allowed.append(int(self.tokenizer.obj_token_id))
        return allowed

    def _prop_allowed(self, base_text_ids: Sequence[int], prop_token_count: int) -> List[int]:
        if int(self.max_prop_tokens) <= 0 or int(prop_token_count) >= int(self.max_prop_tokens):
            return [int(self.tokenizer.prop_end_token_id)]
        return list(base_text_ids) + [int(self.tokenizer.prop_end_token_id)]


class GeoCoordTokenizer:
    _BASE_SPECIAL_TOKENS = [
        "<map_bos>",
        "<state_bos>",
        "<obj>",
        "<obj_end>",
        "<anchor>",
        "<anchor_end>",
        "<line>",
        "<poly>",
        "<prop>",
        "<prop_end>",
        "<pt>",
        "<ring>",
        "<src_local>",
        "<src_state>",
        "<cut_in_none>",
        "<cut_in_left>",
        "<cut_in_top>",
        "<cut_in_right>",
        "<cut_in_bottom>",
        "<cut_in_internal>",
        "<cut_out_none>",
        "<cut_out_left>",
        "<cut_out_top>",
        "<cut_out_right>",
        "<cut_out_bottom>",
        "<cut_out_internal>",
        "<side_none>",
        "<side_left>",
        "<side_top>",
        "<side_right>",
        "<side_bottom>",
    ]

    def __init__(
        self,
        qwen_model_path: str,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        coord_bins: int = 1024,
    ) -> None:
        if AutoTokenizer is None:
            raise_geo_error(
                "GEO-1307",
                "failed to import transformers.AutoTokenizer for GeoCoordTokenizer",
                cause=_TRANSFORMERS_IMPORT_ERROR,
            )
        self.qwen_model_path = resolve_hf_snapshot_path(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.base_vocab_size = int(len(self.tokenizer))
        self.coord_bins = max(16, int(coord_bins))
        coord_tokens = [f"<coord_{idx:04d}>" for idx in range(self.coord_bins)]
        added_tokens = self._BASE_SPECIAL_TOKENS + coord_tokens
        self.tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})

        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.special_token_ids = {
            token: int(self.tokenizer.convert_tokens_to_ids(token))
            for token in added_tokens
        }
        self.coord_token_ids = [self.special_token_ids[f"<coord_{idx:04d}>"] for idx in range(self.coord_bins)]
        self.coord_id_to_bin = {tok_id: idx for idx, tok_id in enumerate(self.coord_token_ids)}
        self.coord_bin_to_id = {idx: tok_id for idx, tok_id in enumerate(self.coord_token_ids)}
        self.control_token_ids = {
            name: tok_id for name, tok_id in self.special_token_ids.items() if not name.startswith("<coord_")
        }

        vocab_size = int(len(self.tokenizer))
        if self.pad_token_id == self.eos_token_id:
            self.allowed_output_token_ids = list(range(vocab_size))
        else:
            self.allowed_output_token_ids = [idx for idx in range(vocab_size) if idx != self.pad_token_id]

        self.map_bos_token_id = int(self.control_token_ids["<map_bos>"])
        self.state_bos_token_id = int(self.control_token_ids["<state_bos>"])
        self.obj_token_id = int(self.control_token_ids["<obj>"])
        self.obj_end_token_id = int(self.control_token_ids["<obj_end>"])
        self.anchor_token_id = int(self.control_token_ids["<anchor>"])
        self.anchor_end_token_id = int(self.control_token_ids["<anchor_end>"])
        self.line_token_id = int(self.control_token_ids["<line>"])
        self.poly_token_id = int(self.control_token_ids["<poly>"])
        self.prop_token_id = int(self.control_token_ids["<prop>"])
        self.prop_end_token_id = int(self.control_token_ids["<prop_end>"])
        self.pt_token_id = int(self.control_token_ids["<pt>"])
        self.ring_token_id = int(self.control_token_ids["<ring>"])
        self.src_token_ids = {
            "local": int(self.control_token_ids["<src_local>"]),
            "state": int(self.control_token_ids["<src_state>"]),
        }
        self.cut_in_token_ids = {
            "none": int(self.control_token_ids["<cut_in_none>"]),
            "left": int(self.control_token_ids["<cut_in_left>"]),
            "top": int(self.control_token_ids["<cut_in_top>"]),
            "right": int(self.control_token_ids["<cut_in_right>"]),
            "bottom": int(self.control_token_ids["<cut_in_bottom>"]),
            "internal": int(self.control_token_ids["<cut_in_internal>"]),
        }
        self.cut_out_token_ids = {
            "none": int(self.control_token_ids["<cut_out_none>"]),
            "left": int(self.control_token_ids["<cut_out_left>"]),
            "top": int(self.control_token_ids["<cut_out_top>"]),
            "right": int(self.control_token_ids["<cut_out_right>"]),
            "bottom": int(self.control_token_ids["<cut_out_bottom>"]),
            "internal": int(self.control_token_ids["<cut_out_internal>"]),
        }
        self.side_token_ids = {
            "none": int(self.control_token_ids["<side_none>"]),
            "left": int(self.control_token_ids["<side_left>"]),
            "top": int(self.control_token_ids["<side_top>"]),
            "right": int(self.control_token_ids["<side_right>"]),
            "bottom": int(self.control_token_ids["<side_bottom>"]),
        }
        self.src_id_to_name = {tok_id: name for name, tok_id in self.src_token_ids.items()}
        self.cut_in_id_to_name = {tok_id: name for name, tok_id in self.cut_in_token_ids.items()}
        self.cut_out_id_to_name = {tok_id: name for name, tok_id in self.cut_out_token_ids.items()}
        self.side_id_to_name = {tok_id: name for name, tok_id in self.side_token_ids.items()}

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode_prompt(self, text: str, max_length: int | None = None) -> List[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if max_length is not None and int(max_length) > 0:
            ids = ids[: int(max_length)]
        return [int(x) for x in ids]

    def encode_text(self, text: str, max_length: int | None = None, append_eos: bool = True) -> List[int]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if append_eos and self.eos_token_id >= 0:
            ids.append(int(self.eos_token_id))
        if max_length is not None and int(max_length) > 0:
            ids = ids[: int(max_length)]
        return [int(x) for x in ids]

    def decode_text(self, token_ids: Sequence[int]) -> str:
        ids = [int(x) for x in token_ids]
        return str(self.tokenizer.decode(ids, skip_special_tokens=True))

    def build_map_grammar_helper(self, task_schema: TaskSchema, max_prop_tokens: int = 128) -> GeoMapGrammarHelper:
        return None

    def encode_state_items(
        self,
        state_items: Sequence[dict],
        image_size: int,
        max_length: int | None = None,
        append_eos: bool = True,
    ) -> List[int]:
        ids: List[int] = [int(self.state_bos_token_id)]
        for item in state_items:
            geometry_type = str(item.get("geometry_type", "linestring")).strip().lower()
            ids.append(int(self.anchor_token_id))
            ids.append(int(self.line_token_id if geometry_type == "linestring" else self.poly_token_id))
            ids.append(int(self.side_token_ids.get(str(item.get("side", "none")), self.side_token_ids["none"])))
            ids.extend(self._encode_points(points_uv=item.get("points_uv", []), image_size=image_size))
            ids.append(int(self.anchor_end_token_id))
        return self._finalize_ids(ids=ids, max_length=max_length, append_eos=append_eos)

    def encode_map_items(
        self,
        map_items: Sequence[dict],
        image_size: int,
        max_length: int | None = None,
        append_eos: bool = True,
    ) -> List[int]:
        ids: List[int] = [int(self.map_bos_token_id)]
        for item in map_items:
            geometry_type = str(item.get("geometry_type", "linestring")).strip().lower()
            ids.append(int(self.obj_token_id))
            ids.append(int(self.line_token_id if geometry_type == "linestring" else self.poly_token_id))
            ids.append(int(self.src_token_ids.get(str(item.get("source", "local")), self.src_token_ids["local"])))
            ids.append(int(self.cut_in_token_ids.get(str(item.get("cut_in", "none")), self.cut_in_token_ids["none"])))
            ids.append(int(self.cut_out_token_ids.get(str(item.get("cut_out", "none")), self.cut_out_token_ids["none"])))
            if geometry_type == "polygon" and item.get("rings_uv"):
                ids.extend(self._encode_rings(rings_uv=item.get("rings_uv", []), image_size=image_size))
            else:
                ids.extend(self._encode_points(points_uv=item.get("points_uv", []), image_size=image_size))
            ids.append(int(self.prop_token_id))
            prop_ids = self.tokenizer.encode(str(item.get("props_json", "{}")), add_special_tokens=False)
            ids.extend(int(x) for x in prop_ids)
            ids.append(int(self.prop_end_token_id))
            ids.append(int(self.obj_end_token_id))
        return self._finalize_ids(ids=ids, max_length=max_length, append_eos=append_eos)

    def decode_map_items(
        self,
        token_ids: Sequence[int],
        task_schema: TaskSchema,
        image_size: int,
    ) -> Tuple[List[dict], dict]:
        ids = self.strip_padding(token_ids)
        if self.eos_token_id in ids:
            ids = ids[: ids.index(self.eos_token_id)]
        items: List[dict] = []
        valid_objects = 0
        saw_object = False
        i = 0
        while i < len(ids):
            if int(ids[i]) != int(self.obj_token_id):
                i += 1
                continue
            saw_object = True
            i += 1
            if i >= len(ids):
                break
            geometry_token = int(ids[i])
            if geometry_token == int(self.line_token_id):
                geometry_type = "linestring"
            elif geometry_token == int(self.poly_token_id):
                geometry_type = "polygon"
            else:
                geometry_type = str(task_schema.geometry_type)
            i += 1

            source = "local"
            if i < len(ids) and int(ids[i]) in self.src_id_to_name:
                source = self.src_id_to_name[int(ids[i])]
                i += 1
            cut_in = "none"
            if i < len(ids) and int(ids[i]) in self.cut_in_id_to_name:
                cut_in = self.cut_in_id_to_name[int(ids[i])]
                i += 1
            cut_out = "none"
            if i < len(ids) and int(ids[i]) in self.cut_out_id_to_name:
                cut_out = self.cut_out_id_to_name[int(ids[i])]
                i += 1

            body_ids: List[int] = []
            while i < len(ids) and int(ids[i]) != int(self.obj_end_token_id):
                body_ids.append(int(ids[i]))
                i += 1
            if i < len(ids) and int(ids[i]) == int(self.obj_end_token_id):
                i += 1

            prop_ids: List[int] = []
            if int(self.prop_token_id) in body_ids:
                prop_start = body_ids.index(int(self.prop_token_id)) + 1
                prop_end = len(body_ids)
                for j in range(prop_start, len(body_ids)):
                    if int(body_ids[j]) == int(self.prop_end_token_id):
                        prop_end = j
                        break
                prop_ids = [int(x) for x in body_ids[prop_start:prop_end]]

            point_buffer = []
            rings_uv = []
            j = 0
            while j < len(body_ids):
                if int(body_ids[j]) == int(self.prop_token_id):
                    break
                if int(body_ids[j]) == int(self.ring_token_id):
                    ring_np = np.asarray(point_buffer, dtype=np.float32)
                    if ring_np.ndim == 2 and ring_np.shape[0] > 0:
                        rings_uv.append(ring_np)
                    point_buffer = []
                    j += 1
                    continue
                if int(body_ids[j]) == int(self.pt_token_id) and j + 2 < len(body_ids):
                    x_id = int(body_ids[j + 1])
                    y_id = int(body_ids[j + 2])
                    if x_id in self.coord_id_to_bin and y_id in self.coord_id_to_bin:
                        point_buffer.append(
                            self._dequantize_pair(
                                x_bin=int(self.coord_id_to_bin[x_id]),
                                y_bin=int(self.coord_id_to_bin[y_id]),
                                image_size=image_size,
                            )
                        )
                        j += 3
                        continue
                j += 1
            point_buffer_np = np.asarray(point_buffer, dtype=np.float32)
            if point_buffer_np.ndim == 2 and point_buffer_np.shape[0] > 0:
                rings_uv.append(point_buffer_np)
            points_uv_np = rings_uv[0] if geometry_type == "polygon" and rings_uv else np.asarray(point_buffer, dtype=np.float32)
            if geometry_type != str(task_schema.geometry_type):
                continue
            raw_props_text = self.tokenizer.decode(prop_ids, skip_special_tokens=True).strip()
            parsed_props = relaxed_parse_props_json(raw_props_text)
            props_parse_ok = bool(not raw_props_text or raw_props_text == "{}" or "_raw_text" not in parsed_props)
            if points_uv_np.ndim == 2 and points_uv_np.shape[0] >= int(task_schema.min_points_per_feature):
                items.append(
                    {
                        "geometry_type": geometry_type,
                        "props_json": raw_props_text or "{}",
                        "props": parsed_props,
                        "props_raw_text": raw_props_text,
                        "props_parse_ok": bool(props_parse_ok),
                        "points_uv": points_uv_np,
                        "rings_uv": [np.asarray(ring, dtype=np.float32) for ring in rings_uv] if geometry_type == "polygon" and rings_uv else None,
                        "source": source,
                        "cut_in": cut_in,
                        "cut_out": cut_out,
                    }
                )
                valid_objects += 1
        return items, {"valid_objects": int(valid_objects), "saw_object": bool(saw_object)}

    def strip_padding(self, token_ids: Iterable[int]) -> List[int]:
        return [int(x) for x in token_ids if int(x) != self.pad_token_id]

    def _finalize_ids(self, ids: List[int], max_length: int | None, append_eos: bool) -> List[int]:
        out = [int(x) for x in ids]
        if append_eos and self.eos_token_id >= 0:
            out.append(int(self.eos_token_id))
        if max_length is not None and int(max_length) > 0:
            out = out[: int(max_length)]
        return out

    def _encode_points(self, points_uv: Sequence[Sequence[float]], image_size: int) -> List[int]:
        out: List[int] = []
        for x_bin, y_bin in self._quantize_points(points_uv=points_uv, image_size=image_size):
            out.extend([int(self.pt_token_id), int(self.coord_bin_to_id[x_bin]), int(self.coord_bin_to_id[y_bin])])
        return out

    def _encode_rings(self, rings_uv: Sequence[Sequence[Sequence[float]]], image_size: int) -> List[int]:
        out: List[int] = []
        for ring_index, ring in enumerate(rings_uv):
            ring_ids = self._encode_points(points_uv=ring, image_size=image_size)
            if not ring_ids:
                continue
            if ring_index > 0:
                out.append(int(self.ring_token_id))
            out.extend(ring_ids)
        return out

    def _quantize_points(self, points_uv: Sequence[Sequence[float]], image_size: int) -> List[Tuple[int, int]]:
        pts = np.asarray(points_uv, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return []
        denom = float(max(1, int(image_size) - 1))
        pts = np.clip(pts, 0.0, denom)
        u = np.round((pts[:, 0] / denom) * float(self.coord_bins - 1)).astype(np.int64)
        v = np.round((pts[:, 1] / denom) * float(self.coord_bins - 1)).astype(np.int64)
        u = np.clip(u, 0, self.coord_bins - 1)
        v = np.clip(v, 0, self.coord_bins - 1)
        return [(int(x), int(y)) for x, y in zip(u.tolist(), v.tolist())]

    def _dequantize_pair(self, x_bin: int, y_bin: int, image_size: int) -> List[float]:
        denom = float(max(1, self.coord_bins - 1))
        size = float(max(1, int(image_size) - 1))
        x = float(x_bin) / denom * size
        y = float(y_bin) / denom * size
        return [x, y]
