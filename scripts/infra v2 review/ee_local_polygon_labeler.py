#!/usr/bin/env python3
"""Local polygon labeler with optional Earth Engine imagery context.

This serves a local Leaflet UI for reviewing polygon GeoJSON against a point CSV
with a WKT geometry column. Edits are held in local session files, not written
back into the source inputs. Every labeling action writes:

- a checkpoint JSON file, for resuming the session
- an updated GeoJSON FeatureCollection, for easy inspection/export

Earth Engine is used only for optional Sentinel-1 and Sentinel-2 tile layers and
availability counts for the selected polygon buffer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import numbers
import os
import sys
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from shapely import wkt
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

try:
    import ee
except Exception:  # pragma: no cover - optional runtime dependency
    ee = None

LABELS = ("Wind", "Aquaculture", "Oil", "Unknown", "ignore")
CHECKPOINT_VERSION = 1
DEFAULT_POLYGONS_PATH = "/Users/jonathanraphael/Downloads/organized-infra.geojson"
DEFAULT_POINTS_PATH = (
    "/Users/jonathanraphael/Downloads/bq-results-20260511-193347-1778528054944.csv"
)
DEFAULT_WKT_COLUMN = "final_center"
DEFAULT_POLYGON_ID_PROP = "polygon_id"
DEFAULT_LABEL_PROP = "class_label"
DEFAULT_POINT_ID_PROP = "structure_id"
DEFAULT_EE_AUTH_SCOPES = ("https://www.googleapis.com/auth/earthengine",)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def label_is_present(value: Any) -> bool:
    # "Unknown" is intentionally counted as classified.
    return not is_blank(value)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def geojson_geometry(geom: Any) -> dict[str, Any]:
    return json_safe(mapping(geom))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    os.replace(tmp_path, path)


@dataclass
class PolygonRecord:
    uid: str
    polygon_id: str
    source_index: int
    feature_id: Any
    properties: dict[str, Any]
    geometry: Any
    original_label: Any


@dataclass
class PointRecord:
    uid: str
    structure_id: Any
    properties: dict[str, Any]
    geometry: Any


class LabelerApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.polygons_path = Path(args.polygons).expanduser().resolve()
        self.points_path = Path(args.points).expanduser().resolve()
        self.checkpoint_path = self._defaulted_path(
            args.checkpoint,
            f"{self.polygons_path.stem}.labeler_checkpoint.json",
        )
        self.output_geojson_path = self._defaulted_path(
            args.output_geojson,
            f"{self.polygons_path.stem}.labeled.geojson",
        )

        self.polygons: list[PolygonRecord] = []
        self.points: list[PointRecord] = []
        self.records_by_uid: dict[str, PolygonRecord] = {}
        self.all_order: list[str] = []
        self.order: list[str] = []
        self.duplicate_polygon_ids: list[str] = []
        self.generated_polygon_ids = 0
        self.missing_polygon_geometry_count = 0
        self.missing_point_wkt_rows = 0
        self.bad_point_wkt_rows = 0
        self.missing_structure_id_rows = 0

        self.edits: dict[str, dict[str, Any]] = {}
        self.skips: dict[str, dict[str, Any]] = {}
        self.undo_stack: list[dict[str, Any]] = []
        self.current_uid: str | None = None

        self.point_tree: STRtree | None = None
        self.point_tree_geoms: list[Any] = []
        self.point_geom_index: dict[int, int] = {}
        self.polygon_tree: STRtree | None = None
        self.polygon_tree_geoms: list[Any] = []
        self.polygon_geom_index: dict[int, int] = {}
        self.polygon_tree_uids: list[str] = []

        self.ee_ready = False
        self.ee_error: str | None = None

        self._load_polygons()
        self._load_points()
        self._build_spatial_indexes()
        self._load_checkpoint()
        if self.current_uid not in self.order:
            self.current_uid = None
        self.current_uid = (
            self.current_uid
            or self.first_unlabeled_uid()
            or (self.order[0] if self.order else None)
        )
        self._initialize_ee()
        self._persist(write_geojson=False)

    def _defaulted_path(self, value: str | None, default_name: str) -> Path:
        if value:
            return Path(value).expanduser().resolve()
        return self.polygons_path.with_name(default_name)

    def _load_polygons(self) -> None:
        with self.polygons_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        if payload.get("type") == "FeatureCollection":
            features = payload.get("features") or []
        elif payload.get("type") == "Feature":
            features = [payload]
        else:
            raise ValueError(
                f"{self.polygons_path} must be a GeoJSON FeatureCollection or Feature"
            )

        seen: dict[str, int] = {}
        for index, feature in enumerate(features):
            if feature.get("type") != "Feature":
                raise ValueError(f"GeoJSON item at index {index} is not a Feature")

            properties = dict(feature.get("properties") or {})
            raw_polygon_id = properties.get(self.args.polygon_id_prop)
            if is_blank(raw_polygon_id):
                raw_polygon_id = f"poly_{index}"
                self.generated_polygon_ids += 1

            polygon_id = str(raw_polygon_id)
            properties[self.args.polygon_id_prop] = polygon_id
            original_label = properties.get(self.args.label_prop)
            properties.setdefault(self.args.label_prop, None)
            duplicate_count = seen.get(polygon_id, 0)
            seen[polygon_id] = duplicate_count + 1
            uid = polygon_id if duplicate_count == 0 else f"{polygon_id}__dup_{index}"

            geometry = shape(feature["geometry"]) if feature.get("geometry") else None
            if geometry is None:
                self.missing_polygon_geometry_count += 1
            record = PolygonRecord(
                uid=uid,
                polygon_id=polygon_id,
                source_index=index,
                feature_id=feature.get("id"),
                properties=properties,
                geometry=geometry,
                original_label=original_label,
            )
            self.polygons.append(record)
            self.records_by_uid[uid] = record
            self.all_order.append(uid)
            if geometry is not None:
                self.order.append(uid)

        self.duplicate_polygon_ids = sorted(
            polygon_id for polygon_id, count in seen.items() if count > 1
        )

    def _load_points(self) -> None:
        with self.points_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            if not reader.fieldnames:
                raise ValueError(f"{self.points_path} has no CSV header")
            wkt_column = self._resolve_wkt_column(reader.fieldnames)
            transformer = self._point_transformer()

            for index, row in enumerate(reader):
                wkt_text = row.get(wkt_column)
                if is_blank(wkt_text):
                    self.missing_point_wkt_rows += 1
                    continue
                try:
                    geometry = wkt.loads(wkt_text)
                except Exception:
                    self.bad_point_wkt_rows += 1
                    continue
                if transformer:
                    geometry = shapely_transform(transformer.transform, geometry)

                properties = {
                    key: value for key, value in row.items() if key != wkt_column
                }
                structure_id = properties.get(self.args.point_id_prop)
                if is_blank(structure_id):
                    self.missing_structure_id_rows += 1

                uid_seed = (
                    structure_id if not is_blank(structure_id) else f"point_{index}"
                )
                self.points.append(
                    PointRecord(
                        uid=str(uid_seed),
                        structure_id=structure_id,
                        properties=properties,
                        geometry=geometry,
                    )
                )

    def _resolve_wkt_column(self, fieldnames: list[str]) -> str:
        if self.args.wkt_column in fieldnames:
            return self.args.wkt_column
        lowered = {name.lower(): name for name in fieldnames}
        for candidate in ("wkt", "geometry", "geom"):
            if candidate in lowered:
                return lowered[candidate]
        raise ValueError(
            f"WKT column '{self.args.wkt_column}' was not found in {self.points_path}. "
            f"Available columns: {', '.join(fieldnames)}"
        )

    def _point_transformer(self) -> Any | None:
        if self.args.points_crs.upper() in ("EPSG:4326", "4326", "WGS84", "WGS 84"):
            return None
        try:
            from pyproj import Transformer
        except Exception as exc:
            raise RuntimeError(
                "--points-crs requires pyproj when it is not EPSG:4326"
            ) from exc
        return Transformer.from_crs(self.args.points_crs, "EPSG:4326", always_xy=True)

    def _build_spatial_indexes(self) -> None:
        self.point_tree_geoms = [point.geometry for point in self.points]
        self.point_geom_index = {
            id(geometry): index for index, geometry in enumerate(self.point_tree_geoms)
        }
        self.point_tree = (
            STRtree(self.point_tree_geoms) if self.point_tree_geoms else None
        )

        self.polygon_tree_uids = [
            record.uid for record in self.polygons if record.geometry is not None
        ]
        self.polygon_tree_geoms = [
            self.records_by_uid[uid].geometry for uid in self.polygon_tree_uids
        ]
        self.polygon_geom_index = {
            id(geometry): index
            for index, geometry in enumerate(self.polygon_tree_geoms)
        }
        self.polygon_tree = (
            STRtree(self.polygon_tree_geoms) if self.polygon_tree_geoms else None
        )

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        with self.checkpoint_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version in {self.checkpoint_path}: "
                f"{payload.get('version')}"
            )

        known_uids = set(self.records_by_uid)
        self.edits = {
            uid: edit
            for uid, edit in (payload.get("edits") or {}).items()
            if uid in known_uids
        }
        self.skips = {
            uid: skip
            for uid, skip in (payload.get("skips") or {}).items()
            if uid in known_uids
        }
        # Checkpoints intentionally keep only the latest persisted state per
        # polygon. Undo history is session-local and is not restored.
        self.undo_stack = []
        checkpoint_uid = payload.get("current_uid")
        if checkpoint_uid in known_uids:
            self.current_uid = checkpoint_uid

    def _initialize_ee(self) -> None:
        if self.args.no_ee:
            self.ee_error = "Earth Engine disabled with --no-ee"
            return
        if ee is None:
            self.ee_error = (
                "earthengine-api is not importable. Install it with: "
                "python3 -m pip install earthengine-api"
            )
            return
        try:
            credentials = self._ee_service_account_credentials()
            if credentials is not None:
                ee.Initialize(credentials, project=self.args.ee_project)
            else:
                if self.args.ee_authenticate:
                    ee.Authenticate(
                        auth_mode=self.args.ee_auth_mode,
                        scopes=self.args.ee_auth_scopes.split(","),
                    )
                if self.args.ee_project:
                    ee.Initialize(project=self.args.ee_project)
                else:
                    ee.Initialize()
            self.ee_ready = True
        except Exception as exc:
            self.ee_error = str(exc)

    def _ee_service_account_credentials(self) -> Any | None:
        key_path = (
            self.args.ee_private_key
            or os.environ.get("EE_PRIVATE_KEY")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        service_account = self.args.ee_service_account or os.environ.get(
            "EE_SERVICE_ACCOUNT"
        )
        if not key_path:
            return None

        key_path_obj = Path(key_path).expanduser()
        if not service_account:
            with key_path_obj.open("r", encoding="utf-8") as file_obj:
                service_account = json.load(file_obj).get("client_email")
        if not service_account:
            raise ValueError(
                "Service account email was not provided and could not be read "
                "from the private key JSON."
            )
        return ee.ServiceAccountCredentials(service_account, str(key_path_obj))

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "saved_at": utc_now(),
            "source": {
                "polygons": str(self.polygons_path),
                "points": str(self.points_path),
            },
            "config": {
                "polygon_id_prop": self.args.polygon_id_prop,
                "label_prop": self.args.label_prop,
                "point_id_prop": self.args.point_id_prop,
                "wkt_column": self.args.wkt_column,
                "points_crs": self.args.points_crs,
            },
            "current_uid": self.current_uid,
            "edits": self.edits,
            "skips": self.skips,
        }

    def _persist(self, *, write_geojson: bool = True) -> None:
        atomic_write_json(self.checkpoint_path, self._checkpoint_payload())
        if write_geojson:
            self.write_updated_geojson()

    def write_updated_geojson(self, output_path: Path | None = None) -> Path:
        target = output_path or self.output_geojson_path
        payload = {
            "type": "FeatureCollection",
            "features": [
                self.current_feature(uid, include_internal=False)
                for uid in self.all_order
            ],
        }
        atomic_write_json(target, payload)
        return target

    def _valid_for_query(self, geometry: Any) -> Any:
        if geometry is None:
            return None
        if geometry.is_empty:
            return geometry
        if geometry.is_valid:
            return geometry
        fixed = geometry.buffer(0)
        return fixed if not fixed.is_empty else geometry

    def _tree_result_indices(
        self,
        result: Any,
        geom_index: dict[int, int],
    ) -> list[int]:
        indices: list[int] = []
        for item in result:
            if isinstance(item, numbers.Integral):
                indices.append(int(item))
            else:
                index = geom_index.get(id(item))
                if index is not None:
                    indices.append(index)
        return indices

    def points_inside(self, geometry: Any) -> list[PointRecord]:
        query_geometry = self._valid_for_query(geometry)
        if query_geometry is None:
            return []
        if query_geometry.is_empty:
            return []
        if self.point_tree is None:
            candidates = range(len(self.points))
        else:
            candidates = self._tree_result_indices(
                self.point_tree.query(query_geometry),
                self.point_geom_index,
            )
        inside: list[PointRecord] = []
        for index in candidates:
            point = self.points[index]
            if point.geometry.intersects(query_geometry):
                inside.append(point)
        inside.sort(
            key=lambda item: "" if item.structure_id is None else str(item.structure_id)
        )
        return inside

    def buffered_geometry(self, geometry: Any, meters: float) -> Any | None:
        if geometry is None:
            return None
        if geometry.is_empty:
            return geometry
        try:
            from pyproj import CRS, Transformer
        except Exception as exc:
            raise RuntimeError("pyproj is required for meter-based buffers") from exc

        centroid = geometry.centroid
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} "
            "+datum=WGS84 +units=m +no_defs"
        )
        to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
        local_geometry = shapely_transform(to_local.transform, geometry)
        return shapely_transform(to_wgs84.transform, local_geometry.buffer(meters))

    def points_in_buffer(self, geometry: Any, meters: float) -> list[PointRecord]:
        return self.points_inside(self.buffered_geometry(geometry, meters))

    def structure_ids_inside(self, geometry: Any) -> tuple[list[str], int]:
        points = self.points_inside(geometry)
        ids = sorted(
            {
                str(point.structure_id)
                for point in points
                if not is_blank(point.structure_id)
            }
        )
        return ids, len(points)

    def current_geometry(self, uid: str) -> Any:
        edit = self.edits.get(uid)
        if edit and edit.get("geometry_replaced") and edit.get("geometry"):
            return shape(edit["geometry"])
        return self.records_by_uid[uid].geometry

    def current_label(self, uid: str) -> Any:
        edit = self.edits.get(uid)
        if edit:
            return edit.get("label")
        return self.records_by_uid[uid].original_label

    def current_status(self, uid: str) -> str:
        if uid in self.edits:
            return "edited"
        if uid in self.skips:
            return "needs_review"
        if self.records_by_uid[uid].geometry is None:
            return "missing_geometry"
        if label_is_present(self.records_by_uid[uid].original_label):
            return "original_labeled"
        return "unclassified"

    def current_feature(self, uid: str, *, include_internal: bool) -> dict[str, Any]:
        record = self.records_by_uid[uid]
        properties = copy.deepcopy(record.properties)
        geometry = record.geometry
        status = self.current_status(uid)

        edit = self.edits.get(uid)
        if edit:
            if edit.get("geometry_replaced") and edit.get("geometry"):
                geometry = shape(edit["geometry"])
            properties[self.args.label_prop] = edit.get("label")
            properties["structure_ids"] = edit.get("structure_ids", [])
            properties["structure_ids_csv"] = edit.get("structure_ids_csv", "")
            properties["structure_id_count"] = edit.get("structure_id_count", 0)
            properties["point_count"] = edit.get("point_count", 0)
            properties["geometry_replaced"] = bool(edit.get("geometry_replaced"))
            properties["annotated_at"] = edit.get("annotated_at")
            properties["annotation_source"] = "ee_local_polygon_labeler"
            properties["annotation_status"] = "labeled"
        elif uid in self.skips:
            properties["annotation_status"] = "needs_review"
            properties["review_flagged_at"] = self.skips[uid].get("flagged_at")
            properties["review_note"] = self.skips[uid].get("note", "")
        elif record.geometry is None:
            properties["annotation_status"] = "missing_geometry"

        if include_internal:
            properties["_labeler_uid"] = uid
            properties["_labeler_status"] = status
            properties["_labeler_current_label"] = self.current_label(uid)
            properties["_labeler_original_label"] = record.original_label
            properties["_labeler_geometry_replaced"] = bool(
                edit and edit.get("geometry_replaced")
            )

        feature: dict[str, Any] = {
            "type": "Feature",
            "properties": json_safe(properties),
            "geometry": geojson_geometry(geometry) if geometry is not None else None,
        }
        if record.feature_id is not None:
            feature["id"] = record.feature_id
        return feature

    def selected_payload(self, uid: str) -> dict[str, Any]:
        self._require_uid(uid)
        record = self.records_by_uid[uid]
        geometry = self.current_geometry(uid)
        points = self.points_inside(geometry)
        structure_ids = sorted(
            {
                str(point.structure_id)
                for point in points
                if not is_blank(point.structure_id)
            }
        )
        inside_missing_ids = sum(1 for point in points if is_blank(point.structure_id))
        edit = self.edits.get(uid)
        warnings = []
        if is_blank(record.polygon_id):
            warnings.append("Polygon has no polygon ID.")
        if record.polygon_id in self.duplicate_polygon_ids:
            warnings.append("Polygon ID is duplicated in the source GeoJSON.")
        if geometry is None:
            warnings.append("Source feature has no geometry and is not labelable.")
        elif not geometry.is_valid:
            warnings.append("Current geometry is invalid.")
        if geometry is not None and geometry.is_empty:
            warnings.append("Current geometry is empty.")
        if inside_missing_ids:
            warnings.append(f"{inside_missing_ids} inside point(s) lack structure_id.")
        if not points:
            warnings.append("No points intersect the current polygon geometry.")

        return {
            "uid": uid,
            "polygon_id": record.polygon_id,
            "source_index": record.source_index,
            "original_label": record.original_label,
            "current_label": self.current_label(uid),
            "status": self.current_status(uid),
            "geometry_replaced": bool(edit and edit.get("geometry_replaced")),
            "point_count": len(points),
            "structure_id_count": len(structure_ids),
            "structure_ids": structure_ids[: self.args.point_preview_limit],
            "structure_ids_total": len(structure_ids),
            "inside_points_missing_structure_id": inside_missing_ids,
            "warnings": warnings,
            "feature": self.current_feature(uid, include_internal=True),
            "points_geojson": self.points_geojson(points),
        }

    def points_geojson(self, points: list[PointRecord]) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": point.uid,
                    "properties": json_safe(point.properties),
                    "geometry": geojson_geometry(point.geometry),
                }
                for point in points
            ],
        }

    def all_points_geojson(self) -> dict[str, Any]:
        if len(self.points) > self.args.max_points_layer:
            return {"type": "FeatureCollection", "features": []}
        return self.points_geojson(self.points)

    def buffer_points_geojson(self, uid: str) -> dict[str, Any]:
        self._require_uid(uid)
        points = self.points_in_buffer(
            self.current_geometry(uid),
            float(self.args.buffer_meters),
        )
        return self.points_geojson(points)

    def all_polygons_geojson(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [
                self.current_feature(uid, include_internal=True) for uid in self.order
            ],
        }

    def _require_uid(self, uid: str) -> None:
        if uid not in self.records_by_uid:
            raise KeyError(f"Unknown polygon uid: {uid}")

    def _push_undo(self, uid: str) -> None:
        self.undo_stack.append(
            {
                "uid": uid,
                "edit": copy.deepcopy(self.edits.get(uid)),
                "skip": copy.deepcopy(self.skips.get(uid)),
                "current_uid": self.current_uid,
                "at": utc_now(),
            }
        )
        self.undo_stack = self.undo_stack[-self.args.undo_limit :]

    def label_polygon(
        self,
        uid: str,
        label: str,
        replacement_geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_uid(uid)
        if label not in LABELS:
            raise ValueError(
                f"Unsupported label '{label}'. Valid labels: {', '.join(LABELS)}"
            )
        self._push_undo(uid)

        geometry = (
            shape(replacement_geometry)
            if replacement_geometry
            else self.current_geometry(uid)
        )
        if geometry is None:
            raise ValueError("Current source feature has no geometry")
        if geometry.is_empty:
            raise ValueError("Replacement geometry is empty")
        structure_ids, point_count = self.structure_ids_inside(geometry)
        self.edits[uid] = {
            "label": label,
            "geometry": geojson_geometry(geometry)
            if replacement_geometry
            else (self.edits.get(uid, {}).get("geometry")),
            "geometry_replaced": bool(
                replacement_geometry
                or self.edits.get(uid, {}).get("geometry_replaced", False)
            ),
            "structure_ids": structure_ids,
            "structure_ids_csv": ",".join(structure_ids),
            "structure_id_count": len(structure_ids),
            "point_count": point_count,
            "annotated_at": utc_now(),
        }
        self.skips.pop(uid, None)
        self.current_uid = uid
        self._persist(write_geojson=True)
        return self.selected_payload(uid)

    def skip_polygon(self, uid: str, note: str = "") -> dict[str, Any]:
        self._require_uid(uid)
        self._push_undo(uid)
        self.skips[uid] = {"flagged_at": utc_now(), "note": note}
        self.edits.pop(uid, None)
        self.current_uid = uid
        self._persist(write_geojson=True)
        return self.selected_payload(uid)

    def revert_polygon(self, uid: str) -> dict[str, Any]:
        self._require_uid(uid)
        self._push_undo(uid)
        self.edits.pop(uid, None)
        self.skips.pop(uid, None)
        self.current_uid = uid
        self._persist(write_geojson=True)
        return self.selected_payload(uid)

    def undo(self) -> dict[str, Any]:
        if not self.undo_stack:
            raise ValueError("Undo stack is empty")
        snapshot = self.undo_stack.pop()
        uid = snapshot["uid"]
        if snapshot.get("edit") is None:
            self.edits.pop(uid, None)
        else:
            self.edits[uid] = snapshot["edit"]
        if snapshot.get("skip") is None:
            self.skips.pop(uid, None)
        else:
            self.skips[uid] = snapshot["skip"]
        if snapshot.get("current_uid") in self.records_by_uid:
            self.current_uid = snapshot["current_uid"]
        else:
            self.current_uid = uid
        self._persist(write_geojson=True)
        return self.selected_payload(self.current_uid)

    def _needs_label(self, uid: str, *, include_skipped: bool) -> bool:
        if uid in self.edits:
            return False
        if uid in self.skips and not include_skipped:
            return False
        return not label_is_present(self.records_by_uid[uid].original_label)

    def first_unlabeled_uid(self) -> str | None:
        for uid in self.order:
            if self._needs_label(uid, include_skipped=False):
                return uid
        return None

    def next_uid(
        self,
        uid: str | None,
        *,
        direction: int = 1,
        only_unlabeled: bool = True,
        include_skipped: bool = False,
    ) -> str | None:
        if not self.order:
            return None
        if uid not in self.order:
            start = -1 if direction > 0 else 0
        else:
            start = self.order.index(uid)

        count = len(self.order)
        for offset in range(1, count + 1):
            index = (start + (offset * direction)) % count
            candidate = self.order[index]
            if not only_unlabeled or self._needs_label(
                candidate,
                include_skipped=include_skipped,
            ):
                return candidate
        return None

    def select_at(self, lon: float, lat: float) -> str | None:
        click = Point(lon, lat)
        candidates: list[str] = []
        if self.polygon_tree is not None:
            indices = self._tree_result_indices(
                self.polygon_tree.query(click),
                self.polygon_geom_index,
            )
            candidates.extend(self.polygon_tree_uids[index] for index in indices)

        # Replacement geometries are not in the static tree, so check them too.
        candidates.extend(
            uid for uid, edit in self.edits.items() if edit.get("geometry_replaced")
        )
        seen: set[str] = set()
        for uid in candidates:
            if uid in seen:
                continue
            seen.add(uid)
            geometry = self.current_geometry(uid)
            if geometry is not None and geometry.intersects(click):
                return uid
        return None

    def summary(self) -> dict[str, Any]:
        progress = self.progress()
        return {
            "config": {
                "labels": LABELS,
                "polygon_id_prop": self.args.polygon_id_prop,
                "label_prop": self.args.label_prop,
                "point_id_prop": self.args.point_id_prop,
                "buffer_meters": self.args.buffer_meters,
                "s1_lookback_days": self.args.s1_lookback_days,
                "s2_lookback_days": self.args.s2_lookback_days,
                "checkpoint_path": str(self.checkpoint_path),
                "output_geojson_path": str(self.output_geojson_path),
                "all_points_layer_enabled": len(self.points)
                <= self.args.max_points_layer,
            },
            "paths": {
                "polygons": str(self.polygons_path),
                "points": str(self.points_path),
            },
            "current_uid": self.current_uid,
            "progress": progress,
            "ee": {"ready": self.ee_ready, "error": self.ee_error},
            "polygons": [
                {
                    "uid": record.uid,
                    "polygon_id": record.polygon_id,
                    "source_index": record.source_index,
                    "original_label": record.original_label,
                    "current_label": self.current_label(record.uid),
                    "status": self.current_status(record.uid),
                    "geometry_replaced": bool(
                        self.edits.get(record.uid, {}).get("geometry_replaced")
                    ),
                }
                for uid in self.order
                for record in [self.records_by_uid[uid]]
            ],
        }

    def progress(self) -> dict[str, int]:
        original_labeled = sum(
            1
            for uid in self.order
            if label_is_present(self.records_by_uid[uid].original_label)
        )
        edited = len(self.edits)
        skipped = len(self.skips)
        remaining = sum(
            1 for uid in self.order if self._needs_label(uid, include_skipped=False)
        )
        return {
            "total_polygons": len(self.all_order),
            "labelable_polygons": len(self.order),
            "total_points": len(self.points),
            "original_labeled": original_labeled,
            "edited": edited,
            "skipped": skipped,
            "remaining_unclassified": remaining,
            "generated_polygon_ids": self.generated_polygon_ids,
            "duplicate_polygon_ids": len(self.duplicate_polygon_ids),
            "missing_polygon_geometry": self.missing_polygon_geometry_count,
            "missing_structure_id_rows": self.missing_structure_id_rows,
            "bad_point_wkt_rows": self.bad_point_wkt_rows,
            "missing_point_wkt_rows": self.missing_point_wkt_rows,
        }

    def validation_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.generated_polygon_ids:
            warnings.append(
                f"{self.generated_polygon_ids} polygon(s) had generated "
                f"{self.args.polygon_id_prop} values."
            )
        if self.duplicate_polygon_ids:
            preview = ", ".join(self.duplicate_polygon_ids[:10])
            warnings.append(
                f"{len(self.duplicate_polygon_ids)} duplicate polygon ID value(s): {preview}"
            )
        if self.missing_polygon_geometry_count:
            warnings.append(
                f"{self.missing_polygon_geometry_count} source feature(s) have no "
                "geometry and are excluded from map labeling."
            )
        if self.missing_point_wkt_rows:
            warnings.append(f"{self.missing_point_wkt_rows} point row(s) lack WKT.")
        if self.bad_point_wkt_rows:
            warnings.append(f"{self.bad_point_wkt_rows} point row(s) had invalid WKT.")
        if self.missing_structure_id_rows:
            warnings.append(
                f"{self.missing_structure_id_rows} point row(s) lack "
                f"{self.args.point_id_prop}."
            )

        unlabeled = [
            uid
            for uid in self.order
            if not label_is_present(self.current_label(uid)) and uid not in self.skips
        ]
        if unlabeled:
            warnings.append(f"{len(unlabeled)} polygon(s) remain unclassified.")

        skipped = len(self.skips)
        if skipped:
            warnings.append(f"{skipped} polygon(s) are flagged needs_review.")

        zero_point_edits = [
            uid
            for uid, edit in self.edits.items()
            if int(edit.get("point_count", 0)) == 0
        ]
        if zero_point_edits:
            warnings.append(
                f"{len(zero_point_edits)} edited polygon(s) contain zero points."
            )

        invalid_edits = []
        for uid, edit in self.edits.items():
            if edit.get("geometry_replaced") and edit.get("geometry"):
                geometry = shape(edit["geometry"])
                if geometry.is_empty or not geometry.is_valid:
                    invalid_edits.append(uid)
        if invalid_edits:
            warnings.append(
                f"{len(invalid_edits)} replacement geometry/geometries are empty or invalid."
            )

        return warnings

    def ee_layer_info(self, uid: str) -> dict[str, Any]:
        self._require_uid(uid)
        geometry = self.current_geometry(uid)
        if geometry is None:
            return {
                "ready": False,
                "error": "Selected source feature has no geometry",
            }
        if not self.ee_ready or ee is None:
            return {
                "ready": False,
                "error": self.ee_error or "Earth Engine unavailable",
            }

        roi_geometry = geojson_geometry(geometry)
        roi = ee.Geometry(roi_geometry).buffer(float(self.args.buffer_meters))
        today = dt.date.today()
        end = today.isoformat()
        s1_start = (today - dt.timedelta(days=self.args.s1_lookback_days)).isoformat()
        s1_fallback_start = (
            today - dt.timedelta(days=self.args.s1_fallback_days)
        ).isoformat()

        payload: dict[str, Any] = {
            "ready": True,
            "buffer_meters": self.args.buffer_meters,
            "s1": {
                "lookback_days": self.args.s1_lookback_days,
                "fallback_days": self.args.s1_fallback_days,
                "count": None,
            },
            "s2": {"lazy": True, "loaded": False},
            "layers": {},
            "warnings": [],
        }

        try:
            s1_info = self._s1_vv_layer(roi, s1_start, end)
            if (
                not s1_info["count"]
                and self.args.s1_fallback_days > self.args.s1_lookback_days
            ):
                fallback_info = self._s1_vv_layer(roi, s1_fallback_start, end)
                if fallback_info["count"]:
                    s1_info = fallback_info
                    s1_info["used_fallback"] = True

            payload["s1"].update(s1_info)
            payload["s1"]["date_end"] = end
            if s1_info.get("url"):
                payload["layers"]["s1"] = {
                    "name": s1_info["mode"],
                    "url": s1_info["url"],
                }
        except Exception as exc:
            payload["warnings"].append(f"S1 unavailable: {exc}")

        return payload

    def ee_s2_layer_info(self, uid: str) -> dict[str, Any]:
        self._require_uid(uid)
        geometry = self.current_geometry(uid)
        if geometry is None:
            return {
                "ready": False,
                "error": "Selected source feature has no geometry",
            }
        if not self.ee_ready or ee is None:
            return {
                "ready": False,
                "error": self.ee_error or "Earth Engine unavailable",
            }

        roi_geometry = geojson_geometry(geometry)
        roi = ee.Geometry(roi_geometry).buffer(float(self.args.buffer_meters))
        today = dt.date.today()
        end = today.isoformat()
        s2_start = (today - dt.timedelta(days=self.args.s2_lookback_days)).isoformat()
        payload: dict[str, Any] = {
            "ready": True,
            "buffer_meters": self.args.buffer_meters,
            "s2": {"lookback_days": self.args.s2_lookback_days, "count": None},
            "layers": {},
            "warnings": [],
        }

        try:
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(s2_start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            )
            s2_count = int(s2.size().getInfo())
            payload["s2"]["count"] = s2_count
            payload["s2"]["date_start"] = s2_start
            payload["s2"]["date_end"] = end
            payload["s2"]["clear_threshold"] = self.args.s2_clear_threshold
            if s2_count:
                newest = s2.aggregate_max("system:time_start").getInfo()
                payload["s2"]["newest"] = self._millis_to_date(newest)
                masked = self._cloud_masked_s2(s2)
                composite = masked.median().clip(roi)
                payload["s2"]["clear_pixel_count"] = self._s2_clear_pixel_count(
                    composite,
                    roi,
                )
                payload["layers"]["s2_true"] = {
                    "name": "S2 true color",
                    "url": self._tile_url(
                        composite,
                        {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
                    ),
                }
                payload["layers"]["s2_false"] = {
                    "name": "S2 false color",
                    "url": self._tile_url(
                        composite,
                        {"bands": ["B11", "B8", "B4"], "min": 0, "max": 4000},
                    ),
                }
                mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
                payload["layers"]["s2_mndwi"] = {
                    "name": "S2 MNDWI",
                    "url": self._tile_url(
                        mndwi,
                        {
                            "min": -0.5,
                            "max": 0.7,
                            "palette": ["8c510a", "f6e8c3", "5ab4ac", "01665e"],
                        },
                    ),
                }
        except Exception as exc:
            payload["warnings"].append(f"S2 unavailable: {exc}")

        return payload

    def _s1_vv_layer(self, roi: Any, start: str, end: str) -> dict[str, Any]:
        raw = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
        )
        vv = raw.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        counts = {
            "raw_count": int(raw.size().getInfo()),
            "vv_count": int(vv.size().getInfo()),
        }
        info: dict[str, Any] = {
            **counts,
            "count": 0,
            "date_start": start,
            "mode": "none",
            "tile_url_returned": False,
            "used_fallback": False,
        }

        if counts["vv_count"]:
            self._populate_s1_vv_band(info, vv, roi)

        return info

    def _populate_s1_vv_band(
        self, info: dict[str, Any], collection: Any, roi: Any
    ) -> None:
        newest = collection.aggregate_max("system:time_start").getInfo()
        image = collection.select("VV").median().clip(roi)
        url = self._tile_url(
            image,
            {
                "min": -25,
                "max": -3,
                # "palette": ["000000", "0044ff", "00ffff", "ffff00", "ffffff"],
            },
        )
        info.update(
            {
                "count": int(collection.size().getInfo()),
                "mode": "VV high-contrast grayscale",
                "newest": self._millis_to_date(newest),
                "url": url,
                "tile_url_returned": bool(url),
            }
        )

    def _cloud_masked_s2(self, s2: Any) -> Any:
        if hasattr(s2, "linkCollection"):
            cloud_score = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            linked = s2.linkCollection(cloud_score, ["cs_cdf"])

            def mask_image(image: Any) -> Any:
                return image.updateMask(
                    image.select("cs_cdf").gte(float(self.args.s2_clear_threshold))
                )

            return linked.map(mask_image)
        return s2

    def _s2_clear_pixel_count(self, composite: Any, roi: Any) -> Any:
        try:
            return (
                composite.select("B4")
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=roi,
                    scale=60,
                    bestEffort=True,
                    maxPixels=1e7,
                )
                .get("B4")
                .getInfo()
            )
        except Exception as exc:
            return f"unavailable: {exc}"

    def _tile_url(self, image: Any, vis_params: dict[str, Any]) -> str:
        map_id = image.getMapId(vis_params)
        tile_fetcher = map_id.get("tile_fetcher")
        if tile_fetcher is not None:
            return tile_fetcher.url_format
        return map_id["tile_url"]

    def _millis_to_date(self, millis: Any) -> str | None:
        if millis is None:
            return None
        return (
            dt.datetime.fromtimestamp(int(millis) / 1000, tz=dt.timezone.utc)
            .date()
            .isoformat()
        )


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EE Local Polygon Labeler</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
  <style>
    html, body { height: 100%; margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }
    #app { display: grid; grid-template-columns: 360px 1fr; height: 100%; }
    #panel { overflow: auto; padding: 14px; border-right: 1px solid #cbd5df; background: #f7f9fb; }
    #map { height: 100%; width: 100%; }
    h1 { margin: 0 0 10px; font-size: 18px; line-height: 1.2; }
    h2 { margin: 18px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0; color: #52616f; }
    .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
    .row > * { flex: 1; }
    .meta { font-size: 12px; line-height: 1.4; color: #34495e; word-break: break-word; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    button, input, select { font: inherit; }
    button { border: 1px solid #9fb0c0; background: white; color: #17212b; border-radius: 5px; padding: 7px 9px; cursor: pointer; }
    button:hover { background: #edf3f8; }
    button.primary { background: #1769aa; color: white; border-color: #1769aa; }
    button.warning { background: #fff7e6; border-color: #d89b3a; }
    button.danger { background: #fff1f0; border-color: #d36b6b; }
    input[type="text"] { border: 1px solid #9fb0c0; border-radius: 5px; padding: 7px 8px; min-width: 0; }
    label.toggle { display: flex; gap: 8px; align-items: center; margin: 7px 0; font-size: 13px; }
    #status { margin: 8px 0 0; padding: 8px; background: #e9f2fb; border: 1px solid #c1d8ed; border-radius: 5px; font-size: 12px; }
    #warnings div { margin: 5px 0; padding: 6px 8px; background: #fff7e6; border-left: 3px solid #d89b3a; font-size: 12px; }
    #pointList { max-height: 110px; overflow: auto; background: white; border: 1px solid #d7e0e7; border-radius: 5px; padding: 7px; }
    #pointList div { font-size: 12px; line-height: 1.35; }
    #classFilter, #polygonList { width: 100%; }
    .small { font-size: 12px; }
    @media (max-width: 820px) {
      #app { grid-template-columns: 1fr; grid-template-rows: 45% 55%; }
      #panel { order: 2; border-right: 0; border-top: 1px solid #cbd5df; }
      #map { order: 1; }
    }
  </style>
</head>
<body>
<div id="app">
  <aside id="panel">
    <h1>EE Local Polygon Labeler</h1>
    <div id="progress" class="meta"></div>
    <div id="status">Loading...</div>

    <h2>Current Polygon</h2>
    <div id="currentMeta" class="meta"></div>
    <div id="warnings"></div>
    <div id="pointList"></div>

    <h2>Class</h2>
    <div class="row">
      <button class="primary" data-label="Wind">1 Wind</button>
      <button data-label="Aquaculture">2 Aquaculture</button>
    </div>
    <div class="row">
      <button data-label="Oil">3 Oil</button>
      <button data-label="Unknown">4 Unknown</button>
    </div>
    <div class="row">
      <button data-label="ignore">5 ignore</button>
    </div>

    <h2>Geometry And Review</h2>
    <div class="row">
      <button id="redrawWind">R Redraw + Wind</button>
      <button id="skipBtn" class="warning">S Needs Review</button>
    </div>
    <div class="row">
      <button id="undoBtn">U Undo</button>
      <button id="revertBtn" class="danger">Revert Current</button>
    </div>

    <h2>Navigation</h2>
    <div class="row">
      <button id="prevBtn">Previous</button>
      <button id="nextBtn" class="primary">Enter Next</button>
    </div>
    <div class="row">
      <input id="jumpInput" type="text" placeholder="polygon_id or uid">
      <button id="jumpBtn">Jump</button>
    </div>
    <select id="classFilter">
      <option value="all">All labelable polygons</option>
      <option value="unclassified">Unclassified</option>
      <option value="needs_review">Needs review</option>
      <option value="Wind">Wind</option>
      <option value="Aquaculture">Aquaculture</option>
      <option value="Oil">Oil</option>
      <option value="Unknown">Unknown</option>
      <option value="ignore">ignore</option>
    </select>
    <select id="polygonList" size="8"></select>

    <h2>Layers</h2>
    <label class="toggle"><input id="toggleSelectedPolygon" type="checkbox" checked> Selected polygon</label>
    <label class="toggle"><input id="togglePolygons" type="checkbox"> Polygons</label>
    <label class="toggle"><input id="toggleAllPoints" type="checkbox"> All points in 20 km buffer</label>
    <label class="toggle"><input id="toggleSelectedPoints" type="checkbox" checked> Selected polygon points</label>
    <label class="toggle"><input id="toggleS1" type="checkbox" checked> S1 recent mosaic</label>
    <label class="toggle"><input id="toggleS2True" type="checkbox"> S2 true color</label>
    <label class="toggle"><input id="toggleS2False" type="checkbox"> S2 false color</label>
    <label class="toggle"><input id="toggleMndwi" type="checkbox"> S2 MNDWI</label>
    <div id="eeMeta" class="meta"></div>

    <h2>Save And Export</h2>
    <div class="row">
      <button id="checkpointBtn">Save Checkpoint</button>
      <button id="exportBtn" class="primary">Export GeoJSON</button>
    </div>
    <button id="validateBtn">Print Validation Warnings</button>
    <div id="paths" class="meta"></div>
  </aside>
  <main id="map"></main>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script>
let summary = null;
let current = null;
let polygonGeoJson = null;
let currentUid = null;
let polygonsLayer = null;
let allPointsLayer = null;
let selectedLayer = null;
let selectedPointsLayer = null;
let eeTileLayers = {};
let navigating = false;
let s2Payload = null;
let s2Loading = false;
let activeFilter = "all";

const map = L.map("map", { preferCanvas: true }).setView([0, 0], 2);
L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { attribution: "Imagery: Esri and contributors", maxZoom: 19 }
).addTo(map);

function qs(id) { return document.getElementById(id); }
function setStatus(text) { qs("status").textContent = text; }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[ch]));
}
async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function postJson(url, payload = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function statusColor(status, label) {
  if (status === "needs_review") return "#8c6d31";
  if (status === "edited") {
    if (label === "Wind") return "#1f78b4";
    if (label === "Aquaculture") return "#33a02c";
    if (label === "Oil") return "#e31a1c";
    if (label === "Unknown") return "#6a3d9a";
    if (label === "ignore") return "#111827";
  }
  if (status === "original_labeled") return "#4b5563";
  return "#f59e0b";
}

function itemMatchesFilter(item) {
  if (activeFilter === "all") return true;
  if (activeFilter === "unclassified") return item.status === "unclassified";
  if (activeFilter === "needs_review") return item.status === "needs_review";
  return item.current_label === activeFilter;
}

function featureMatchesFilter(feature) {
  const props = feature.properties || {};
  return itemMatchesFilter({
    status: props._labeler_status,
    current_label: props._labeler_current_label
  });
}

function filteredPolygonItems() {
  return (summary?.polygons || []).filter(itemMatchesFilter);
}

function polygonStyle(feature) {
  const props = feature.properties || {};
  const selected = props._labeler_uid === currentUid;
  const color = selected ? "#00ffff" : statusColor(props._labeler_status, props._labeler_current_label);
  return {
    color,
    weight: selected ? 4 : 2,
    opacity: selected ? 1 : 0.85,
    fillColor: color,
    fillOpacity: selected ? 0.12 : 0.04
  };
}

function pointStyle(color, radius) {
  return { radius, color, weight: 2, opacity: 0.95, fillColor: color, fillOpacity: 0 };
}

async function init() {
  summary = await getJson("/api/summary");
  updateSummary(summary);
  await refreshPolygons();
  rebuildPolygonSelect();
  const first = summary.current_uid || (summary.polygons[0] && summary.polygons[0].uid);
  if (first) await selectPolygon(first, true);
  bindControls();
}

function updateSummary(data) {
  summary = data;
  const p = data.progress;
  qs("progress").innerHTML =
    `<div><b>${p.edited}</b> edited, <b>${p.remaining_unclassified}</b> remaining, ` +
    `<b>${p.skipped}</b> needs review, <b>${p.total_polygons}</b> total polygons</div>` +
    `<div><b>${p.total_points}</b> points loaded</div>`;
  qs("paths").innerHTML =
    `<div>Checkpoint: <span class="mono">${escapeHtml(data.config.checkpoint_path)}</span></div>` +
    `<div>Updated GeoJSON: <span class="mono">${escapeHtml(data.config.output_geojson_path)}</span></div>`;
}

async function refreshPolygons() {
  polygonGeoJson = await getJson("/api/polygons.geojson");
  renderPolygonsLayer();
}

function renderPolygonsLayer() {
  if (polygonsLayer) map.removeLayer(polygonsLayer);
  const filteredGeoJson = {
    type: "FeatureCollection",
    features: (polygonGeoJson?.features || []).filter(featureMatchesFilter)
  };
  polygonsLayer = L.geoJSON(filteredGeoJson, {
    style: polygonStyle,
    onEachFeature: (feature, layer) => {
      layer.on("click", (event) => {
        L.DomEvent.stopPropagation(event);
        selectPolygon(feature.properties._labeler_uid, false);
      });
    }
  });
  if (qs("togglePolygons").checked) polygonsLayer.addTo(map);
}

async function refreshAllPoints() {
  if (!currentUid) return;
  const data = await getJson(`/api/buffer-points/${encodeURIComponent(currentUid)}.geojson`);
  if (allPointsLayer) map.removeLayer(allPointsLayer);
  allPointsLayer = L.geoJSON(data, {
    pointToLayer: (_, latlng) => L.circleMarker(latlng, pointStyle("#d7191c", 4))
  });
  if (qs("toggleAllPoints").checked) allPointsLayer.addTo(map);
}

function rebuildPolygonSelect() {
  const select = qs("polygonList");
  select.innerHTML = "";
  for (const item of filteredPolygonItems()) {
    const option = document.createElement("option");
    option.value = item.uid;
    option.textContent = `${item.polygon_id} | ${item.status} | ${item.current_label ?? ""}`;
    select.appendChild(option);
  }
}

async function selectPolygon(uid, recenter) {
  currentUid = uid;
  current = await getJson(`/api/polygon/${encodeURIComponent(uid)}`);
  updateCurrentPanel();
  updateSelectedLayers(recenter);
  await refreshAllPoints();
  updateSelectedLayers(false);
  if (polygonsLayer) polygonsLayer.setStyle(polygonStyle);
  qs("polygonList").value = uid;
  refreshEe(uid);
}

function updateCurrentPanel() {
  qs("currentMeta").innerHTML =
    `<div>polygon_id: <span class="mono">${escapeHtml(current.polygon_id)}</span></div>` +
    `<div>uid: <span class="mono">${escapeHtml(current.uid)}</span></div>` +
    `<div>original label: <b>${escapeHtml(current.original_label ?? "")}</b></div>` +
    `<div>session label: <b>${escapeHtml(current.current_label ?? "")}</b></div>` +
    `<div>status: <b>${escapeHtml(current.status)}</b></div>` +
    `<div>geometry: <b>${current.geometry_replaced ? "redrawn" : "original"}</b></div>` +
    `<div>points inside: <b>${current.point_count}</b>; unique structure IDs: <b>${current.structure_id_count}</b></div>`;
  qs("warnings").innerHTML = (current.warnings || []).map(w => `<div>${escapeHtml(w)}</div>`).join("");
  qs("pointList").innerHTML =
    `<div><b>First ${current.structure_ids.length} structure_ids</b></div>` +
    current.structure_ids.map(id => `<div class="mono">${escapeHtml(id)}</div>`).join("");
}

function updateSelectedLayers(recenter) {
  if (selectedLayer) map.removeLayer(selectedLayer);
  selectedLayer = L.geoJSON(current.feature, { style: () => ({
    color: "#00ffff", weight: 4, fillColor: "#00ffff", fillOpacity: 0.12
  }) });
  if (qs("toggleSelectedPolygon").checked) selectedLayer.addTo(map);

  if (selectedPointsLayer) map.removeLayer(selectedPointsLayer);
  selectedPointsLayer = L.geoJSON(current.points_geojson, {
    pointToLayer: (_, latlng) => L.circleMarker(latlng, pointStyle("#d7191c", 6))
  });
  if (qs("toggleSelectedPoints").checked) selectedPointsLayer.addTo(map);

  if (recenter) {
    const bounds = selectedLayer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds.pad(0.35));
  }
}

async function refreshEe(uid) {
  clearEeLayers();
  s2Payload = null;
  qs("eeMeta").textContent = "Loading S1 imagery counts...";
  try {
    const data = await getJson(`/api/ee/${encodeURIComponent(uid)}`);
    if (!data.ready) {
      qs("eeMeta").textContent = `EE unavailable: ${data.error}`;
      return;
    }
    qs("eeMeta").innerHTML =
      `<div>S1 layer: <b>${escapeHtml(data.s1.mode ?? "none")}</b></div>` +
      `<div>S1 raw: <b>${data.s1.raw_count ?? "n/a"}</b>; ` +
      `VV: <b>${data.s1.vv_count ?? "n/a"}</b></div>` +
      `<div>S1 tile URL: <b>${data.s1.tile_url_returned ? "yes" : "no"}</b>; ` +
      `fallback window: <b>${data.s1.used_fallback ? "yes" : "no"}</b> ` +
      `(${data.s1.date_start} to ${data.s1.date_end}) newest: ${data.s1.newest ?? "none"}</div>` +
      `<div id="s2Meta">S2: lazy loaded when an S2 layer is enabled.</div>` +
      (data.warnings || []).map(w => `<div>${escapeHtml(w)}</div>`).join("");
    addEeLayer("s1", data.layers.s1, qs("toggleS1").checked);
    await refreshS2IfNeeded(uid);
  } catch (err) {
    qs("eeMeta").textContent = `EE imagery failed: ${err.message}`;
  }
}

function anyS2ToggleChecked() {
  return qs("toggleS2True").checked || qs("toggleS2False").checked || qs("toggleMndwi").checked;
}

async function refreshS2IfNeeded(uid) {
  if (!anyS2ToggleChecked()) return;
  await loadS2(uid);
}

async function loadS2(uid) {
  if (!uid || s2Loading) return;
  if (!s2Payload || s2Payload.uid !== uid) {
    s2Loading = true;
    const s2Meta = qs("s2Meta");
    if (s2Meta) s2Meta.textContent = "S2: loading...";
    try {
      const data = await getJson(`/api/ee-s2/${encodeURIComponent(uid)}`);
      if (!data.ready) {
        if (s2Meta) s2Meta.textContent = `S2 unavailable: ${data.error}`;
        return;
      }
      s2Payload = { uid, data };
      if (s2Meta) {
        s2Meta.innerHTML =
          `<div>S2 raw count: <b>${data.s2.count ?? "n/a"}</b> ` +
          `(${data.s2.date_start} to ${data.s2.date_end}) newest: ${data.s2.newest ?? "none"}</div>` +
          `<div>S2 clear pixels: <b>${escapeHtml(data.s2.clear_pixel_count ?? "n/a")}</b></div>` +
          (data.warnings || []).map(w => `<div>${escapeHtml(w)}</div>`).join("");
      }
    } finally {
      s2Loading = false;
    }
  }
  const data = s2Payload.data;
  addEeLayer("s2_true", data.layers.s2_true, qs("toggleS2True").checked);
  addEeLayer("s2_false", data.layers.s2_false, qs("toggleS2False").checked);
  addEeLayer("s2_mndwi", data.layers.s2_mndwi, qs("toggleMndwi").checked);
}

function clearEeLayers() {
  for (const key of Object.keys(eeTileLayers)) {
    map.removeLayer(eeTileLayers[key]);
  }
  eeTileLayers = {};
}

function addEeLayer(key, layerInfo, shown) {
  if (!layerInfo || !layerInfo.url) return;
  if (eeTileLayers[key]) map.removeLayer(eeTileLayers[key]);
  const zIndex = key === "s1" ? 430 : 420;
  const layer = L.tileLayer(layerInfo.url, {
    opacity: key === "s1" ? 1.0 : 0.7,
    zIndex: zIndex
  });
  eeTileLayers[key] = layer;
  if (shown) layer.addTo(map);
}

async function labelCurrent(label, geometry = null) {
  if (!currentUid) return;
  const labeledUid = currentUid;
  setStatus(`Saving ${label}...`);
  const data = await postJson("/api/label", { uid: labeledUid, label, geometry });
  current = data.selected;
  updateSummary(data.summary);
  rebuildPolygonSelect();
  await refreshPolygons();
  await advanceAfterClassification(labeledUid, label);
}

async function advanceAfterClassification(labeledUid, label) {
  const data = await getJson(`/api/next?uid=${encodeURIComponent(labeledUid)}&direction=1&only_unlabeled=1`);
  if (data.uid) {
    await selectPolygon(data.uid, true);
    setStatus(`${label} saved. Advanced to next unclassified polygon.`);
  } else {
    await selectPolygon(labeledUid, false);
    setStatus(`${label} saved. No remaining unclassified polygon found.`);
  }
}

async function skipCurrent() {
  if (!currentUid) return;
  setStatus("Saving needs_review...");
  const data = await postJson("/api/skip", { uid: currentUid });
  current = data.selected;
  updateSummary(data.summary);
  rebuildPolygonSelect();
  await refreshPolygons();
  await selectPolygon(current.uid, false);
  setStatus("Needs review saved. Checkpoint and updated GeoJSON written.");
}

async function revertCurrent() {
  if (!currentUid) return;
  setStatus("Reverting current polygon...");
  const data = await postJson("/api/revert", { uid: currentUid });
  current = data.selected;
  updateSummary(data.summary);
  rebuildPolygonSelect();
  await refreshPolygons();
  await selectPolygon(current.uid, false);
  setStatus("Current polygon reverted. Checkpoint and updated GeoJSON written.");
}

async function undoLast() {
  setStatus("Undoing...");
  const data = await postJson("/api/undo", {});
  current = data.selected;
  updateSummary(data.summary);
  rebuildPolygonSelect();
  await refreshPolygons();
  await selectPolygon(current.uid, false);
  setStatus("Undo complete. Checkpoint and updated GeoJSON written.");
}

async function goNext(direction, onlyUnlabeled) {
  if (navigating) return;
  navigating = true;
  try {
    const filteredUid = nextFilteredUid(direction, onlyUnlabeled);
    if (filteredUid) {
      await selectPolygon(filteredUid, true);
      setStatus("Selected polygon.");
      return;
    }
    if (activeFilter === "all") {
      const data = await getJson(`/api/next?uid=${encodeURIComponent(currentUid || "")}&direction=${direction}&only_unlabeled=${onlyUnlabeled ? "1" : "0"}`);
      if (data.uid) {
        await selectPolygon(data.uid, true);
        setStatus("Selected polygon.");
      } else {
        setStatus("No matching polygon found.");
      }
      return;
    }
    setStatus("No matching polygon found for the active filter.");
  } finally {
    navigating = false;
  }
}

function nextFilteredUid(direction, onlyUnlabeled) {
  if (activeFilter === "all" && onlyUnlabeled) return null;
  const items = filteredPolygonItems();
  if (!items.length) return null;
  const currentIndex = items.findIndex(item => item.uid === currentUid);
  const start = currentIndex >= 0 ? currentIndex : (direction > 0 ? -1 : 0);
  const nextIndex = (start + direction + items.length) % items.length;
  return items[nextIndex].uid;
}

function startRedraw() {
  if (!currentUid) return;
  setStatus("Draw replacement polygon. Double-click or close the shape to finish.");
  const drawer = new L.Draw.Polygon(map, {
    allowIntersection: false,
    showArea: true,
    shapeOptions: { color: "#00ffff", weight: 3 }
  });
  map.once(L.Draw.Event.CREATED, async (event) => {
    const geometry = event.layer.toGeoJSON().geometry;
    await labelCurrent("Wind", geometry);
  });
  drawer.enable();
}

function bindControls() {
  document.querySelectorAll("button[data-label]").forEach(button => {
    button.addEventListener("click", () => {
      button.blur();
      labelCurrent(button.dataset.label).catch(showError);
    });
  });
  qs("redrawWind").addEventListener("click", startRedraw);
  qs("skipBtn").addEventListener("click", () => skipCurrent().catch(showError));
  qs("undoBtn").addEventListener("click", () => undoLast().catch(showError));
  qs("revertBtn").addEventListener("click", () => revertCurrent().catch(showError));
  qs("nextBtn").addEventListener("click", () => goNext(1, true).catch(showError));
  qs("prevBtn").addEventListener("click", () => goNext(-1, false).catch(showError));
  qs("jumpBtn").addEventListener("click", jumpToPolygon);
  qs("classFilter").addEventListener("change", () => {
    activeFilter = qs("classFilter").value;
    rebuildPolygonSelect();
    renderPolygonsLayer();
    const currentItem = (summary?.polygons || []).find(item => item.uid === currentUid);
    if (currentItem && !itemMatchesFilter(currentItem)) {
      const firstMatch = filteredPolygonItems()[0];
      if (firstMatch) selectPolygon(firstMatch.uid, true).catch(showError);
    }
  });
  qs("polygonList").addEventListener("change", () => selectPolygon(qs("polygonList").value, true).catch(showError));
  qs("checkpointBtn").addEventListener("click", () => postJson("/api/checkpoint", {}).then(() => setStatus("Checkpoint written.")).catch(showError));
  qs("exportBtn").addEventListener("click", () => postJson("/api/export", {}).then(data => setStatus(`Exported ${data.output_geojson_path}`)).catch(showError));
  qs("validateBtn").addEventListener("click", () => getJson("/api/validate").then(data => {
    console.table(data.warnings.map(w => ({ warning: w })));
    setStatus(`${data.warnings.length} validation warning(s) printed to the browser console.`);
  }).catch(showError));

  qs("togglePolygons").addEventListener("change", (event) => toggleLayer(polygonsLayer, event.target.checked));
  qs("toggleAllPoints").addEventListener("change", (event) => toggleLayer(allPointsLayer, event.target.checked));
  qs("toggleSelectedPolygon").addEventListener("change", (event) => toggleLayer(selectedLayer, event.target.checked));
  qs("toggleSelectedPoints").addEventListener("change", (event) => toggleLayer(selectedPointsLayer, event.target.checked));
  qs("toggleS1").addEventListener("change", (event) => toggleLayer(eeTileLayers.s1, event.target.checked));
  qs("toggleS2True").addEventListener("change", (event) => handleS2Toggle("s2_true", event.target.checked));
  qs("toggleS2False").addEventListener("change", (event) => handleS2Toggle("s2_false", event.target.checked));
  qs("toggleMndwi").addEventListener("change", (event) => handleS2Toggle("s2_mndwi", event.target.checked));

  map.on("click", async (event) => {
    try {
      const data = await postJson("/api/select-at", { lat: event.latlng.lat, lon: event.latlng.lng });
      if (data.uid) await selectPolygon(data.uid, false);
    } catch (err) {
      showError(err);
    }
  });

  document.addEventListener("keydown", (event) => {
    const tag = (event.target && event.target.tagName || "").toLowerCase();
    if (event.repeat) return;
    if (tag === "input" || tag === "select" || tag === "textarea" || tag === "button") return;
    if (event.key === "Enter") {
      event.preventDefault();
      event.stopPropagation();
      goNext(1, true).catch(showError);
      return;
    }
    if (event.key === "1") labelCurrent("Wind").catch(showError);
    if (event.key === "2") labelCurrent("Aquaculture").catch(showError);
    if (event.key === "3") labelCurrent("Oil").catch(showError);
    if (event.key === "4") labelCurrent("Unknown").catch(showError);
    if (event.key === "5") labelCurrent("ignore").catch(showError);
    if (event.key.toLowerCase() === "u") undoLast().catch(showError);
    if (event.key.toLowerCase() === "r") startRedraw();
    if (event.key.toLowerCase() === "s") skipCurrent().catch(showError);
  });
}

function handleS2Toggle(key, shown) {
  if (!shown) {
    toggleLayer(eeTileLayers[key], false);
    return;
  }
  loadS2(currentUid).catch(showError);
}

function toggleLayer(layer, shown) {
  if (!layer) return;
  if (shown) layer.addTo(map);
  else map.removeLayer(layer);
}

function jumpToPolygon() {
  const value = qs("jumpInput").value.trim();
  if (!value) return;
  const match = summary.polygons.find(item => item.uid === value || item.polygon_id === value);
  if (!match) {
    setStatus("No polygon matched that ID.");
    return;
  }
  selectPolygon(match.uid, true).catch(showError);
}

function showError(err) {
  console.error(err);
  setStatus(`Error: ${err.message || err}`);
}

init().catch(showError);
</script>
</body>
</html>
"""


class LabelerRequestHandler(BaseHTTPRequestHandler):
    app: LabelerApp

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), fmt % args)
        )

    def do_GET(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            if path == "/":
                self._html_response(HTML_PAGE)
            elif path == "/api/summary":
                self._json_response(self.app.summary())
            elif path == "/api/polygons.geojson":
                self._json_response(self.app.all_polygons_geojson())
            elif path == "/api/points.geojson":
                self._json_response(self.app.all_points_geojson())
            elif path.startswith("/api/buffer-points/"):
                uid = urllib.parse.unquote(path[len("/api/buffer-points/") :])
                if uid.endswith(".geojson"):
                    uid = uid[: -len(".geojson")]
                self._json_response(self.app.buffer_points_geojson(uid))
            elif path.startswith("/api/polygon/"):
                uid = urllib.parse.unquote(path[len("/api/polygon/") :])
                self.app.current_uid = uid
                self._json_response(self.app.selected_payload(uid))
            elif path.startswith("/api/ee/"):
                uid = urllib.parse.unquote(path[len("/api/ee/") :])
                self._json_response(self.app.ee_layer_info(uid))
            elif path.startswith("/api/ee-s2/"):
                uid = urllib.parse.unquote(path[len("/api/ee-s2/") :])
                self._json_response(self.app.ee_s2_layer_info(uid))
            elif path == "/api/next":
                uid = query.get("uid", [None])[0]
                direction = int(query.get("direction", ["1"])[0])
                only_unlabeled = query.get("only_unlabeled", ["1"])[0] == "1"
                next_uid = self.app.next_uid(
                    uid,
                    direction=direction,
                    only_unlabeled=only_unlabeled,
                )
                self._json_response({"uid": next_uid})
            elif path == "/api/validate":
                self._json_response({"warnings": self.app.validation_warnings()})
            else:
                self._json_response(
                    {"error": f"Not found: {path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
        except Exception as exc:
            self._json_response(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def do_POST(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            payload = self._read_json_body()
            if path == "/api/label":
                selected = self.app.label_polygon(
                    payload["uid"],
                    payload["label"],
                    payload.get("geometry"),
                )
                self._json_response(
                    {"selected": selected, "summary": self.app.summary()}
                )
            elif path == "/api/skip":
                selected = self.app.skip_polygon(
                    payload["uid"], payload.get("note", "")
                )
                self._json_response(
                    {"selected": selected, "summary": self.app.summary()}
                )
            elif path == "/api/revert":
                selected = self.app.revert_polygon(payload["uid"])
                self._json_response(
                    {"selected": selected, "summary": self.app.summary()}
                )
            elif path == "/api/undo":
                selected = self.app.undo()
                self._json_response(
                    {"selected": selected, "summary": self.app.summary()}
                )
            elif path == "/api/checkpoint":
                self.app._persist(write_geojson=False)
                self._json_response(
                    {
                        "checkpoint_path": str(self.app.checkpoint_path),
                        "summary": self.app.summary(),
                    }
                )
            elif path == "/api/export":
                output_path = payload.get("output_geojson_path")
                target = (
                    Path(output_path).expanduser().resolve() if output_path else None
                )
                written = self.app.write_updated_geojson(target)
                self._json_response(
                    {
                        "output_geojson_path": str(written),
                        "warnings": self.app.validation_warnings(),
                    }
                )
            elif path == "/api/select-at":
                uid = self.app.select_at(float(payload["lon"]), float(payload["lat"]))
                self._json_response({"uid": uid})
            else:
                self._json_response(
                    {"error": f"Not found: {path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
        except Exception as exc:
            self._json_response(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _html_response(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json_response(
        self,
        payload: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(json_safe(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local Earth Engine assisted polygon labeler.",
    )
    parser.add_argument(
        "--polygons",
        default=DEFAULT_POLYGONS_PATH,
        help="Input polygon GeoJSON path.",
    )
    parser.add_argument(
        "--points",
        default=DEFAULT_POINTS_PATH,
        help="Input point CSV path.",
    )
    parser.add_argument(
        "--wkt-column",
        default=DEFAULT_WKT_COLUMN,
        help="CSV column containing point WKT. Falls back to wkt/geometry/geom.",
    )
    parser.add_argument(
        "--points-crs",
        default="EPSG:4326",
        help="CRS for point WKT coordinates. Reprojected to EPSG:4326 if needed.",
    )
    parser.add_argument("--polygon-id-prop", default=DEFAULT_POLYGON_ID_PROP)
    parser.add_argument("--label-prop", default=DEFAULT_LABEL_PROP)
    parser.add_argument("--point-id-prop", default=DEFAULT_POINT_ID_PROP)
    parser.add_argument("--checkpoint", help="Checkpoint JSON output path.")
    parser.add_argument("--output-geojson", help="Updated GeoJSON output path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--buffer-meters", type=float, default=20000)
    parser.add_argument("--s1-lookback-days", type=int, default=90)
    parser.add_argument(
        "--s1-fallback-days",
        type=int,
        default=365,
        help="Fallback S1 lookback window if the primary S1 window has no imagery.",
    )
    parser.add_argument("--s2-lookback-days", type=int, default=180)
    parser.add_argument("--s2-clear-threshold", type=float, default=0.60)
    parser.add_argument("--point-preview-limit", type=int, default=20)
    parser.add_argument(
        "--max-points-layer",
        type=int,
        default=5000,
        help="Maximum point count for rendering the all-points overlay.",
    )
    parser.add_argument("--undo-limit", type=int, default=100)
    parser.add_argument("--ee-project", help="Google Cloud project for ee.Initialize.")
    parser.add_argument(
        "--ee-service-account",
        help=(
            "Service account email for Earth Engine. If omitted, the script "
            "will read client_email from the key JSON."
        ),
    )
    parser.add_argument(
        "--ee-private-key",
        help=(
            "Path to a service-account private key JSON. Also read from "
            "EE_PRIVATE_KEY or GOOGLE_APPLICATION_CREDENTIALS."
        ),
    )
    parser.add_argument(
        "--ee-authenticate",
        action="store_true",
        help="Run ee.Authenticate() before ee.Initialize().",
    )
    parser.add_argument(
        "--ee-auth-mode",
        default="localhost",
        help="Auth mode passed to ee.Authenticate when --ee-authenticate is set.",
    )
    parser.add_argument(
        "--ee-auth-scopes",
        default=",".join(DEFAULT_EE_AUTH_SCOPES),
        help=(
            "Comma-separated OAuth scopes for --ee-authenticate. The default "
            "omits Drive and Cloud Storage because this viewer only needs EE."
        ),
    )
    parser.add_argument("--no-ee", action="store_true", help="Disable EE imagery.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    app = LabelerApp(args)
    LabelerRequestHandler.app = app

    server = ThreadingHTTPServer((args.host, args.port), LabelerRequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving polygon labeler at {url}")
    print(f"Polygons: {app.polygons_path}")
    print(f"Points: {app.points_path}")
    print(f"Checkpoint: {app.checkpoint_path}")
    print(f"Updated GeoJSON: {app.output_geojson_path}")
    if app.ee_ready:
        print("Earth Engine: initialized")
    else:
        print(f"Earth Engine: unavailable ({app.ee_error})")
        if app.ee_error and "earthengine-api is not importable" in app.ee_error:
            print("Install dependency: python3 -m pip install earthengine-api")
        else:
            print(
                "If browser OAuth is blocked by Google or your Workspace admin, "
                "try scoped Earth Engine-only auth or use a service account key:"
            )
            print(
                "  earthengine authenticate --force --auth_mode=localhost "
                "--scopes=https://www.googleapis.com/auth/earthengine"
            )
            print(
                "  python3 scripts/ee_local_polygon_labeler.py "
                "--ee-project YOUR_PROJECT_ID "
                "--ee-private-key /path/to/service-account-key.json"
            )
    print("Press Ctrl-C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        app._persist(write_geojson=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
