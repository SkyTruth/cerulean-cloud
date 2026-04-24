"""Utilities for orchestrator-side AOI joins."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import geopandas as gpd
import google.auth
from google.cloud import storage
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


BoundsLike = Union[Sequence[float], BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]
GCS_READONLY_SCOPE = ("https://www.googleapis.com/auth/devstorage.read_only",)


@dataclass(frozen=True)
class AOIAccessConfig:
    """Configuration for accessing a single AOI dataset in Google Cloud Storage."""

    key: str
    geometry_source_uri: str
    ext_id_field: str
    name_field: Optional[str] = None
    pmtiles_uri: Optional[str] = None
    dataset_version: Optional[str] = None
    filter_toggle: Optional[bool] = None
    read_perm: Optional[int] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AOIAccessConfig":
        """
        Build an AOI access config from an `aoi_type`-style row.

        The authoritative field mapping lives in `aoi_type.properties`; direct
        keys are accepted for compatibility with existing DatabaseClient return
        values while that interface is being updated.
        """
        properties = row.get("properties") or {}
        if not isinstance(properties, Mapping):
            raise ValueError("AOI access config properties must be a mapping")

        key = row.get("key") or row.get("short_name")
        if not key:
            raise ValueError("AOI access config requires `key` or `short_name`")

        access_type = row.get("access_type", "GCS")
        if access_type != "GCS":
            raise NotImplementedError(
                "AOIJoiner currently supports only GCS-backed AOI types; "
                f"unsupported AOI type {key!r} has access_type={access_type!r}"
            )

        geometry_source_uri = (
            row.get("geometry_source_uri")
            or properties.get("geometry_source_uri")
            or properties.get("fgb_uri")
        )
        if not geometry_source_uri:
            raise ValueError(f"GCS AOI type {key!r} is missing properties['fgb_uri']")

        ext_id_field = (
            row.get("ext_id_field")
            or properties.get("ext_id_field")
            or properties.get("ext_id_col")
        )
        if not ext_id_field:
            raise ValueError(
                f"GCS AOI type {key!r} is missing properties['ext_id_field']"
            )

        name_field = (
            row.get("name_field")
            or properties.get("name_field")
            or properties.get("display_name_field")
            or properties.get("name_col")
        )

        return cls(
            key=str(key),
            geometry_source_uri=str(geometry_source_uri),
            ext_id_field=str(ext_id_field),
            name_field=str(name_field) if name_field else None,
            pmtiles_uri=row.get("pmtiles_uri") or properties.get("pmt_uri"),
            dataset_version=row.get("dataset_version")
            or properties.get("dataset_version"),
            filter_toggle=row.get("filter_toggle"),
            read_perm=row.get("read_perm"),
        )


class AOIJoiner:
    """
    Load scene-relevant AOI datasets and compute slick intersections.

    This class intentionally stops at the geospatial join boundary. It does not
    write AOIs or slick-to-AOI mappings to the database.
    """

    def __init__(
        self,
        scene_bounds: BoundsLike,
        aoi_access_configs: Iterable[Union[AOIAccessConfig, Mapping[str, Any]]],
    ) -> None:
        self.scene_bounds = self._normalize_bounds(scene_bounds)
        self.scene_bbox = tuple(self.scene_bounds.bounds)
        self.aoi_configs = tuple(
            access_config
            if isinstance(access_config, AOIAccessConfig)
            else AOIAccessConfig.from_mapping(access_config)
            for access_config in aoi_access_configs
        )
        if not self.aoi_configs:
            raise ValueError("AOIJoiner requires at least one AOI access configuration")
        self.cache_dir = Path(tempfile.gettempdir()) / "cerulean_aoi_cache"
        self.gcp_project: Optional[str] = None
        self.aoi_gdfs: Dict[str, gpd.GeoDataFrame] = self._load_aoi_gdfs()

    def _normalize_bounds(self, scene_bounds: BoundsLike) -> BaseGeometry:
        """Normalize supported bounds inputs into a shapely polygon."""
        if isinstance(scene_bounds, BaseGeometry):
            return scene_bounds
        if isinstance(scene_bounds, gpd.GeoDataFrame):
            return scene_bounds.union_all()
        if isinstance(scene_bounds, gpd.GeoSeries):
            return scene_bounds.union_all()
        if len(scene_bounds) != 4:
            raise ValueError(
                "scene_bounds must be a geometry or a 4-value bounds tuple"
            )
        minx, miny, maxx, maxy = scene_bounds
        return box(minx, miny, maxx, maxy)

    def _load_aoi_gdfs(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load AOI FlatGeobufs clipped to the current scene bounds."""
        return {
            access_config.key: self._read_aoi_dataset(access_config)
            for access_config in self.aoi_configs
        }

    def _get_gcs_credentials(self):
        """
        Resolve credentials for AOI downloads.

        Use application default credentials for both local development and
        Cloud Run.
        """
        credentials, project = google.auth.default(scopes=GCS_READONLY_SCOPE)
        self.gcp_project = project
        return credentials

    def _download_aoi_dataset(self, access_config: AOIAccessConfig) -> str:
        """Resolve `gs://` AOI dataset paths into local cached files for GeoPandas."""
        if not access_config.geometry_source_uri.startswith("gs://"):
            return access_config.geometry_source_uri

        bucket_and_path = access_config.geometry_source_uri[len("gs://") :]
        bucket_name, _, object_name = bucket_and_path.partition("/")
        if not bucket_name or not object_name:
            raise ValueError(
                f"Invalid gs:// AOI dataset URL: {access_config.geometry_source_uri}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_digest = hashlib.sha256(
            (
                f"{access_config.geometry_source_uri}|"
                f"{access_config.dataset_version or ''}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        local_name = f"{bucket_name}__{object_name.replace('/', '__')}__{cache_digest}"
        local_path = self.cache_dir / local_name
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        credentials = self._get_gcs_credentials()
        client = storage.Client(project=self.gcp_project, credentials=credentials)
        client.bucket(bucket_name).blob(object_name).download_to_filename(local_path)
        return str(local_path)

    def _read_aoi_dataset(self, access_config: AOIAccessConfig) -> gpd.GeoDataFrame:
        """
        Read an AOI FlatGeobuf for the scene bbox.

        GeoPandas can pass `bbox` through to the underlying vector driver, which
        keeps these reads bounded to the scene envelope instead of loading the
        full global layer.
        """
        gdf = gpd.read_file(
            self._download_aoi_dataset(access_config), bbox=self.scene_bbox
        )
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        rename_map = {access_config.ext_id_field: "ext_id"}
        if access_config.name_field and access_config.name_field in gdf.columns:
            rename_map[access_config.name_field] = "name"
        gdf = gdf.rename(columns=rename_map)

        if "ext_id" not in gdf.columns:
            raise ValueError(
                f"AOI dataset '{access_config.key}' is missing expected ext id field "
                f"'{access_config.ext_id_field}'"
            )

        keep_cols = [
            col for col in ("ext_id", "name", "geometry") if col in gdf.columns
        ]
        gdf = gdf[keep_cols].copy()
        gdf["ext_id"] = gdf["ext_id"].astype(str)
        if "name" not in gdf.columns:
            gdf["name"] = gdf["ext_id"]
        return gdf

    def compute_aoi_matches(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """
        Return AOI match metadata per slick.

        The result order matches `slick_gdf.reset_index(drop=True)`, and each item
        uses the configured AOI type keys, for example:

        `{"EEZ": [{"ext_id": "...", "name": "...", "geometry": ...}], ...}`
        """
        if slick_gdf.empty:
            return []

        slicks = slick_gdf.copy()
        if slicks.crs is None:
            slicks = slicks.set_crs("EPSG:4326")
        else:
            slicks = slicks.to_crs("EPSG:4326")
        slicks = slicks.reset_index(drop=True)

        results: List[Dict[str, List[Dict[str, Any]]]] = [
            {access_config.key: [] for access_config in self.aoi_configs}
            for _ in range(len(slicks))
        ]

        for access_config in self.aoi_configs:
            aoi_gdf = self.aoi_gdfs[access_config.key]
            if aoi_gdf.empty:
                continue

            joined = gpd.sjoin(
                slicks[["geometry"]],
                aoi_gdf[["ext_id", "name", "geometry"]],
                how="left",
                predicate="intersects",
            )

            for slick_idx, group in joined.groupby(level=0):
                matches: List[Dict[str, Any]] = []
                matched = group.dropna(subset=["ext_id", "index_right"])
                if matched.empty:
                    results[int(slick_idx)][access_config.key] = matches
                    continue

                for ext_id, ext_id_group in matched.groupby("ext_id", sort=True):
                    right_indices = ext_id_group["index_right"].dropna().tolist()
                    geometries = aoi_gdf.loc[right_indices, "geometry"].tolist()
                    names = ext_id_group["name"].dropna().tolist()
                    matches.append(
                        {
                            "ext_id": str(ext_id),
                            "name": str(names[0]) if names else str(ext_id),
                            "geometry": unary_union(geometries),
                        }
                    )

                results[int(slick_idx)][access_config.key] = matches

        return results

    def compute_aoi_intersect(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[str]]]:
        """
        Return AOI external IDs per slick, shaped like the legacy `aoi_ext_ids`.
        """
        return [
            {
                aoi_type: [match["ext_id"] for match in matches]
                for aoi_type, matches in slick_matches.items()
            }
            for slick_matches in self.compute_aoi_matches(slick_gdf)
        ]

    def compute_single_slick_intersect(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> Dict[str, List[str]]:
        """
        Convenience wrapper for the common single-slick case.

        Raises if more than one slick row is provided.
        """
        if len(slick_gdf) != 1:
            raise ValueError(
                "compute_single_slick_intersect expects exactly one slick row"
            )
        return self.compute_aoi_intersect(slick_gdf)[0]

    def compute_single_slick_matches(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convenience wrapper for single-slick AOI match metadata.
        """
        if len(slick_gdf) != 1:
            raise ValueError(
                "compute_single_slick_matches expects exactly one slick row"
            )
        return self.compute_aoi_matches(slick_gdf)[0]

    def get_aoi_gdf(self, aoi_type: str) -> gpd.GeoDataFrame:
        """Return the cached GeoDataFrame for a single AOI type."""
        return self.aoi_gdfs[aoi_type]

    def as_dict(self) -> Mapping[str, gpd.GeoDataFrame]:
        """Expose the cached AOI GeoDataFrames."""
        return self.aoi_gdfs
