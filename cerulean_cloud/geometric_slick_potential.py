import geopandas as gpd


def add_geom_columns(
    slick_gdf: gpd.GeoDataFrame,
    feature_columns: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Add geometry-derived feature columns for MultiPolygon geometries.

    This function assumes the geometry column contains MultiPolygon objects
    and computes several area-based features after projecting the data
    to an equal-area CRS.
    """
    slick_gdf = gpd.GeoDataFrame(slick_gdf)

    slick_gdf["geometry_count"] = slick_gdf["geometry"].apply(
        lambda geom: len(geom.geoms)
    )

    slick_gdf_newProj = slick_gdf.to_crs("EPSG:6933")

    slick_gdf_newProj["largest_area"] = slick_gdf_newProj["geometry"].apply(
        lambda geom: max(part.area for part in geom.geoms)
    )

    # Apply to all members of the dataframe
    def median_area(multipoly):
        num_geom = len(multipoly.geoms)
        middle = num_geom // 2
        areas = [part.area for part in multipoly.geoms]
        return sorted(areas)[middle]

    slick_gdf_newProj["median_area"] = slick_gdf_newProj["geometry"].apply(median_area)
    if feature_columns is not None:
        return slick_gdf_newProj[feature_columns]
    return slick_gdf_newProj
