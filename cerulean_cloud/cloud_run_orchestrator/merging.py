"""merging inference from base and offset tiles"""
from typing import Optional

import geojson
import geopandas as gpd
import pandas as pd


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)


def concat_grids_adjust_conf(grid_base, grid_offset, offset_max_acceptable_distance):
    """concats the two grid inferences, divides machine_confidence by two if they are not
    intersecting or within offset_max_acceptable_distance meters from another grid's
    polygons."""
    sjoin_result_inner = grid_offset.sjoin_nearest(
        grid_base,
        how="inner",
        max_distance=offset_max_acceptable_distance,
        distance_col="join_distance",
    )
    grid_offset.loc[
        ~grid_offset.index.isin(sjoin_result_inner.index), "machine_confidence"
    ] /= 2
    grid_base.loc[
        ~grid_base.index.isin(sjoin_result_inner["index_right"]), "machine_confidence"
    ] /= 2

    return pd.concat([grid_offset, grid_base])


def merge_inferences(
    base_tile_fc: geojson.FeatureCollection,
    offset_tile_fc: geojson.FeatureCollection,
    offset_max_acceptable_distance: int = 70 * 8,
    buffer_distance: Optional[int] = None,  # 2 * 70
) -> geojson.FeatureCollection:
    """merge base and offset tile inference"""
    pd.options.mode.chained_assignment = None

    print(
        f"Params for determining base and offset inference match are offset_max_acceptable_distance: {offset_max_acceptable_distance}"
    )
    print(f"Using buffer_distance: {buffer_distance} for erosion and dilation")

    if base_tile_fc["features"] and offset_tile_fc["features"]:
        base_gdf = gpd.GeoDataFrame.from_features(base_tile_fc["features"], crs=4326)
        offset_gdf = gpd.GeoDataFrame.from_features(
            offset_tile_fc["features"], crs=4326
        )

        base_gdf = reproject_to_utm(base_gdf)
        offset_gdf = reproject_to_utm(offset_gdf).to_crs(base_gdf.crs)

        if offset_max_acceptable_distance:
            concat_gdf = concat_grids_adjust_conf(
                base_gdf, offset_gdf, offset_max_acceptable_distance
            )
        else:
            concat_gdf = pd.concat([base_gdf, offset_gdf])

        if buffer_distance:
            # Do some dilation
            concat_gdf = concat_gdf.buffer(buffer_distance)

        dissolved_gdf = concat_gdf.dissolve(
            aggfunc={"machine_confidence": "median", "inf_idx": "max"}
        )

        if buffer_distance:
            # Do some erosion
            dissolved_gdf = dissolved_gdf.buffer(-buffer_distance)

        result = dissolved_gdf.to_crs(crs=4326)

        return result.__geo_interface__
    else:
        return geojson.FeatureCollection(features=[])
