"""merging inference from base and offset tiles"""
import geojson
import geopandas as gpd
import networkx as nx
import pandas as pd


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)


def merge_inferences(
    base_tile_fc: geojson.FeatureCollection,
    offset_tile_fc: geojson.FeatureCollection,
    isolated_conf_multiplier: float = 1,
    proximity_meters: int = 500,
    closing_meters: int = 0,
    opening_meters: int = 0,
) -> geojson.FeatureCollection:
    """
    Merge base and offset tile inference.

    This function takes in two geojson FeatureCollections and merges them together.
    During the merge, the geometries can be adjusted to incorporate neighboring features
    based on the proximity setting. The confidence of isolated features can also be adjusted.

    Parameters:
    - base_tile_fc: The primary FeatureCollection to be merged.
    - offset_tile_fc: The secondary FeatureCollection to be merged with the primary.
    - isolated_conf_multiplier: A multiplier for the confidence of isolated features (default is 1).
    - proximity_meters: The distance to check for neighboring features and expand the geometries (default is 500m).
    - closing_meters: The distance to apply the morphological 'closing' operation (default is 0m).
    - opening_meters: The distance to apply the morphological 'opening' operation (default is 0m).

    Returns:
    A merged geojson FeatureCollection.
    """

    # Check if both FeatureCollections have features
    if base_tile_fc["features"] and offset_tile_fc["features"]:
        # Convert the FeatureCollections to GeoDataFrames
        base_gdf = gpd.GeoDataFrame.from_features(base_tile_fc["features"], crs=4326)
        offset_gdf = gpd.GeoDataFrame.from_features(
            offset_tile_fc["features"], crs=4326
        )

        # Reproject both GeoDataFrames to a UTM CRS (for accurate distance calculations)
        base_gdf = reproject_to_utm(base_gdf)
        offset_gdf = offset_gdf.to_crs(base_gdf.crs)

        # Combine both GeoDataFrames
        concat_gdf = pd.concat([base_gdf, offset_gdf], ignore_index=True)
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
        final_gdf.loc[
            final_gdf["group_count"] == 1, "machine_confidence"
        ] *= isolated_conf_multiplier

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

        return result.__geo_interface__
    else:
        # If one of the FeatureCollections is empty, return an empty FeatureCollection
        return geojson.FeatureCollection(features=[])
