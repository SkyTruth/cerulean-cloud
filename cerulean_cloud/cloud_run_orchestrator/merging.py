"""merging inference from base and offset tiles"""
from typing import List

import geojson
import geopandas as gpd
import networkx as nx
import pandas as pd


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)


def merge_inferences(
    feature_collections: List[geojson.FeatureCollection],
    proximity_meters: int = 500,
    closing_meters: int = 0,
    opening_meters: int = 0,
) -> geojson.FeatureCollection:
    """
    Merge base and all offset tile inference.

    This function takes a list of geojson FeatureCollections and merges them together. During the merge, the
    geometries can be adjusted to incorporate neighboring features based on the proximity setting. The
    confidence of isolated features can also be adjusted.

    Parameters:
    - feature_collections: A list of FeatureCollecitons to be merged, a primary and any secondary FeatureCollections
    - proximity_meters: The distance to check for neighboring features and expand the geometries (default is 500m).
    - closing_meters: The distance to apply the morphological 'closing' operation (default is 0m).
    - opening_meters: The distance to apply the morphological 'opening' operation (default is 0m).

    Returns:
    A merged geojson FeatureCollection.
    """
    # We reproject to UTM for processing. This assumes that all offset images will either be in the same UTM zone as
    # the input image chip, or that the difference that arise from an offset crossing into a second UTM zone will
    # have little or no impact on comparison to the original image.
    gdfs_for_processing = [
        reproject_to_utm(
            gpd.GeoDataFrame.from_features(fc["features"], crs=4326).assign(fc_index=i)
        )
        for i, fc in enumerate(feature_collections)
        if fc["features"]
    ]

    if len(gdfs_for_processing) == 0:
        # No inferences found in any tiling
        return geojson.FeatureCollection(features=[])

    # Concat the GeoDataFrames
    concat_gdf = pd.concat(gdfs_for_processing, ignore_index=True)
    final_gdf = concat_gdf.copy()

    # If proximity is set, expand the geometry of each feature by the defined distance
    if proximity_meters is not None:
        concat_gdf["geometry"] = concat_gdf.buffer(proximity_meters)

    # Join the features that intersect with each other
    joined = gpd.sjoin(concat_gdf, concat_gdf, predicate="intersects").reset_index()

    # Create a graph where each node represents a feature and edges represent overlaps/intersections
    G = nx.from_pandas_edgelist(joined, "index", "index_right")

    # For each connected component in the graph, assign a group index and count its features
    group_mapping = {
        feature: group
        for group, component in enumerate(nx.connected_components(G))
        for feature in component
    }
    group_counts = {
        feature: len(component)
        for component in nx.connected_components(G)
        for feature in component
    }

    # Map the group indices and counts back to the GeoDataFrame
    final_gdf["group_index"] = final_gdf.index.map(group_mapping)
    final_gdf["group_count"] = final_gdf.index.map(group_counts)

    # Adjust the confidence value for features that are isolated (not part of a larger group)
    final_gdf["overlap_factor"] = final_gdf.groupby("group_index")[
        "fc_index"
    ].transform(lambda x: len(x.unique()) / len(feature_collections))

    final_gdf["machine_confidence"] *= final_gdf["overlap_factor"]

    # Dissolve overlapping features into one based on their group index and calculate the median confidence and maximum inference index
    dissolved_gdf = final_gdf.dissolve(
        by="group_index", aggfunc={"machine_confidence": "median", "inf_idx": "max"}
    )

    # If set, apply a morphological 'closing' operation to the geometries
    if closing_meters is not None:
        dissolved_gdf["geometry"] = dissolved_gdf.buffer(closing_meters).buffer(
            -closing_meters
        )

    # If set, apply a morphological 'opening' operation to the geometries
    if opening_meters is not None:
        dissolved_gdf["geometry"] = dissolved_gdf.buffer(-opening_meters).buffer(
            opening_meters
        )

    # Reproject the GeoDataFrame back to WGS 84 CRS
    result = dissolved_gdf.to_crs(crs=4326)

    # Clean up potentially memory heavy assets
    del dissolved_gdf
    del concat_gdf
    del final_gdf
    del joined

    return result.__geo_interface__
