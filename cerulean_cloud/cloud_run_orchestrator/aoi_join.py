"""Utilities for orchestrator-side AOI joins."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


BoundsLike = Union[Sequence[float], BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]


@dataclass(frozen=True)
class AOISourceConfig:
    """Configuration for a single AOI source."""

    key: str
    url: str
    ext_id_field: str
    name_field: Optional[str] = None


DEFAULT_AOI_SOURCES: Tuple[AOISourceConfig, ...] = (
    AOISourceConfig(
        key="eez",
        url="https://storage.googleapis.com/cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
        ext_id_field="MRGID",
        name_field="GEONAME",
    ),
    AOISourceConfig(
        key="iho",
        url="https://storage.googleapis.com/cerulean-cloud-aoi/iho-mr/World_Seas_IHO_v3.fgb",
        ext_id_field="MRGID",
        name_field="NAME",
    ),
    AOISourceConfig(
        key="mpa",
        url="https://storage.googleapis.com/cerulean-cloud-aoi/mpa-wdpa/marine_wdpa_0.001.fgb",
        ext_id_field="WDPAID",
        name_field="NAME",
    ),
)


class AOIJoiner:
    """
    Load scene-relevant AOI boundaries and compute slick intersections.

    This class intentionally stops at the geospatial join boundary. It does not
    write AOIs or slick-to-AOI mappings to the database.
    """

    def __init__(
        self,
        scene_bounds: BoundsLike,
        aoi_sources: Iterable[AOISourceConfig] = DEFAULT_AOI_SOURCES,
    ) -> None:
        self.scene_bounds = self._normalize_bounds(scene_bounds)
        self.scene_bbox = tuple(self.scene_bounds.bounds)
        self.aoi_sources = tuple(aoi_sources)
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
            raise ValueError("scene_bounds must be a geometry or a 4-value bounds tuple")
        minx, miny, maxx, maxy = scene_bounds
        return box(minx, miny, maxx, maxy)

    def _load_aoi_gdfs(self) -> Dict[str, gpd.GeoDataFrame]:
        """Load AOI FlatGeobufs clipped to the current scene bounds."""
        return {source.key: self._read_source(source) for source in self.aoi_sources}

    def _read_source(self, source: AOISourceConfig) -> gpd.GeoDataFrame:
        """
        Read a FlatGeobuf source for the scene bbox.

        GeoPandas can pass `bbox` through to the underlying vector driver, which
        keeps these reads bounded to the scene envelope instead of loading the
        full global layer.
        """
        gdf = gpd.read_file(source.url, bbox=self.scene_bbox)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        rename_map = {source.ext_id_field: "ext_id"}
        if source.name_field and source.name_field in gdf.columns:
            rename_map[source.name_field] = "name"
        gdf = gdf.rename(columns=rename_map)

        if "ext_id" not in gdf.columns:
            raise ValueError(
                f"AOI source '{source.key}' is missing expected ext id field "
                f"'{source.ext_id_field}'"
            )

        keep_cols = [col for col in ("ext_id", "name", "geometry") if col in gdf.columns]
        gdf = gdf[keep_cols].copy()
        gdf["ext_id"] = gdf["ext_id"].astype(str)
        return gdf

    def compute_aoi_intersect(self, slick_gdf: gpd.GeoDataFrame) -> List[Dict[str, List[str]]]:
        """
        Return AOI external IDs per slick, shaped like `aoi_ext_ids`.

        The result order matches `slick_gdf.reset_index(drop=True)`, and each item
        has the form:

        `{"eez": [...], "iho": [...], "mpa": [...]}`
        """
        if slick_gdf.empty:
            return []

        slicks = slick_gdf.copy()
        if slicks.crs is None:
            slicks = slicks.set_crs("EPSG:4326")
        else:
            slicks = slicks.to_crs("EPSG:4326")
        slicks = slicks.reset_index(drop=True)

        results: List[Dict[str, List[str]]] = [
            {source.key: [] for source in self.aoi_sources} for _ in range(len(slicks))
        ]

        for source in self.aoi_sources:
            aoi_gdf = self.aoi_gdfs[source.key]
            if aoi_gdf.empty:
                continue

            joined = gpd.sjoin(
                slicks[["geometry"]],
                aoi_gdf[["ext_id", "geometry"]],
                how="left",
                predicate="intersects",
            )

            for slick_idx, group in joined.groupby(level=0):
                ext_ids = sorted({ext_id for ext_id in group["ext_id"].dropna().tolist()})
                results[int(slick_idx)][source.key] = ext_ids

        return results

    def compute_single_slick_intersect(self, slick_gdf: gpd.GeoDataFrame) -> Dict[str, List[str]]:
        """
        Convenience wrapper for the common single-slick case.

        Raises if more than one slick row is provided.
        """
        if len(slick_gdf) != 1:
            raise ValueError("compute_single_slick_intersect expects exactly one slick row")
        return self.compute_aoi_intersect(slick_gdf)[0]

    def get_aoi_gdf(self, aoi_type: str) -> gpd.GeoDataFrame:
        """Return the cached GeoDataFrame for a single AOI type."""
        return self.aoi_gdfs[aoi_type]

    def as_dict(self) -> Mapping[str, gpd.GeoDataFrame]:
        """Expose the cached AOI GeoDataFrames."""
        return self.aoi_gdfs
