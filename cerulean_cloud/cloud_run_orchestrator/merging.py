"""merging inference from base and offset tiles"""
import geojson
import geopandas as gpd
import networkx as nx
import pandas as pd
from typing import List


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)

def merge_inferences(
    feature_collections: List[geojson.FeatureCollection],  # XXXC >> USING THIS AS A TEST replacement for base_tile_fc and offset_tile_fc
    isolated_conf_multiplier: float = None,  # THIS IS FURTHER DEFINED ON LINES 39-40 as 1 / len(feature_collections)
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
    - isolated_conf_multiplier: A multiplier for the confidence of isolated features (default is 1 / len(feature_collections)).
    - proximity_meters: The distance to check for neighboring features and expand the geometries (default is 500m).
    - closing_meters: The distance to apply the morphological 'closing' operation (default is 0m).
    - opening_meters: The distance to apply the morphological 'opening' operation (default is 0m).

    Returns:
    A merged geojson FeatureCollection.
    """

    # Define the isolated_conf_multiplier
    if isolated_conf_multiplier is None:
        isolated_conf_multiplier = 1 / len(feature_collections)

    # Combined GeoDataFrames. Only appended inf all FeatureCollections have at least 1 feature.
    # gdfs_for_processing = []

    # Check that all FeatureCollections have features. This throws out any detection if it is not present in 
    # all tiles of the feature_collections.
    # if any(len(fc["features"]) == 0 for fc in feature_collections):
    #     shared_crs = feature_collections[0].crs
    #     for fc in feature_collections:
    #         # Convert the fc to a GeoDataFrame
    #         gdf = gpd.GeoDataFrame.from_features(fc["features"], crs=4326)
        
    #         # Reproject both GeoDataFrames to a UTM CRS (for accurate distance calculations)
    #         gdfs_for_processing.append(gdf)

    #     gdf_r = reproject_to_utm(gdf)
    
    # else:
    #     # If one of the FeatureCollections is empty, return an empty FeatureCollection
    #     return geojson.FeatureCollection(features=[])

    gdfs_for_processing = [gpd.GeoDataFrame.from_features(fc["features"], crs=4326) if fc["features"] else gpd.GeoDataFrame([], crs=4326) for fc in feature_collections]
    gdfs_for_processing = [gdf.to_crs(gdfs_for_processing[0].reproject_to_utm(gdfs_for_processing[0])) for gdf in gdfs_for_processing]

    # Concat the GeoDataFrames
    concat_gdf = pd.concat(gdfs_for_processing, ignore_index=True)
    final_gdf = concat_gdf.copy()

    # If proximity is set, expand the geometry of each feature by the defined distance
    if proximity_meters is not None:
        concat_gdf["geometry"] = concat_gdf.buffer(proximity_meters)

    # Join the features that intersect with each other >> XXXC THIS MAY NEED REWORK, keep an eye on it :)
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
    # XXXC >> how do we make this a little more dynamic, check against number of tile detections to
    # impact the conf_multiplier value
    final_gdf.loc[
        final_gdf["group_count"] == 1, "machine_confidence"
    ] *= isolated_conf_multiplier

    # Dissolve overlapping features into one based on their group index and calculate the median confidence and maximum inference index
    dissolved_gdf = final_gdf.dissolve(
        by="group_index", aggfunc={"machine_confidence": "median", "inf_idx": "max"}
    )

    # If set, apply a morphological 'closing' operation to the geometries
    if closing_meters is not None:
        dissolved_gdf["geometry"] = dissolved_gdf.buffer(closing_meters).buffer(-closing_meters)

    # If set, apply a morphological 'opening' operation to the geometries
    if opening_meters is not None:
        dissolved_gdf["geometry"] = dissolved_gdf.buffer(-opening_meters).buffer(opening_meters)

    # Reproject the GeoDataFrame back to WGS 84 CRS
    result = dissolved_gdf.to_crs(crs=4326)

    return result.__geo_interface__
