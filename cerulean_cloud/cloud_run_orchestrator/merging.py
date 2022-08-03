"""merging inference from base and offset tiles"""
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
