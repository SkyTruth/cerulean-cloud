import geopandas as gpd
import shapely.wkt as wkt


def add_geom_columns(slick_gdf):
    slick_gdf["st_astext"] = slick_gdf["st_astext"].apply(wkt.loads)
    hitl_gdf = gpd.GeoDataFrame(
        slick_gdf,
        geometry="st_astext",
        crs="EPSG:4326",  # adjust if coordinates are not lon/lat
    ).drop_duplicates(subset=["id"], keep="first")

    hitl_gdf["geometry_count"] = hitl_gdf["st_astext"].apply(
        lambda geom: len(geom.geoms)
    )

    hitl_gdf_newProj = hitl_gdf.to_crs("EPSG:6933")

    hitl_gdf_newProj["largest_area"] = hitl_gdf_newProj["st_astext"].apply(
        lambda geom: max(part.area for part in geom.geoms)
    )

    # Apply to all members of the dataframe
    def median_area(multipoly):
        num_geom = len(multipoly.geoms)
        middle = num_geom // 2
        areas = [part.area for part in multipoly.geoms]
        return sorted(areas)[middle]

    hitl_gdf_newProj["median_area"] = hitl_gdf_newProj["st_astext"].apply(median_area)

    return hitl_gdf_newProj
