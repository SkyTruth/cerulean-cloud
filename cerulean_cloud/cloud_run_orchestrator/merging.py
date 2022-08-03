"""merging inference from base and offset tiles"""
from typing import Optional

import geojson
import geopandas as gpd
import libpysal
import pandas as pd


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)


def concat_grids_adjust_conf(grid_base, grid_offset, offset_max_acceptable_distance):
    """concats the two grid inferences, divides confidence by two if they are not
    intersecting or within offset_max_acceptable_distance meters from another grid's
    polygons."""
    sjoin_result_inner = grid_offset.sjoin_nearest(
        grid_base,
        how="inner",
        max_distance=offset_max_acceptable_distance,
        distance_col="join_distance",
    )

    grid_offset.loc[
        ~grid_offset.index.isin(sjoin_result_inner.index), "confidence"
    ] /= 2
    grid_base.loc[
        ~grid_base.index.isin(sjoin_result_inner["index_right"]), "confidence"
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

    grid_base = gpd.GeoDataFrame.from_features(base_tile_fc["features"], crs=4326)
    grid_offset = gpd.GeoDataFrame.from_features(offset_tile_fc["features"], crs=4326)

    grid_base = reproject_to_utm(grid_base)
    grid_offset = reproject_to_utm(grid_offset)

    all_grid_gdf = concat_grids_adjust_conf(
        grid_base, grid_offset, offset_max_acceptable_distance
    )

    # create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(all_grid_gdf)

    # get component labels
    components = W.component_labels

    all_grid_dissolved_class_dominance_median_conf = all_grid_gdf.dissolve(
        by=components, aggfunc={"confidence": "median", "classification": "max"}
    )

    if buffer_distance:
        # Do some erosion and dilation
        all_grid_dissolved_class_dominance_median_conf = (
            all_grid_dissolved_class_dominance_median_conf.buffer(
                buffer_distance
            ).buffer(-buffer_distance)
        )

    result = all_grid_dissolved_class_dominance_median_conf.to_crs(crs=4326)

    return result.__geo_interface__
