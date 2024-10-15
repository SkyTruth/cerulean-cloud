"""
This module facilitates the retrieval, processing, and visualization of infrastructure data from the Global Fishing Watch API. It downloads Mapbox Vector Tiles (MVT), transforms tile coordinates to geographic coordinates, processes geometric data, and generates visual plots based on label confidence levels.
"""

# %%

import os

import geopandas as gpd
import mapbox_vector_tile
import mercantile
from shapely.geometry import MultiPoint, Point, shape
from shapely.ops import transform

token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJTU1MiLCJ1c2VySWQiOjgwNTAsImFwcGxpY2F0aW9uTmFtZSI6IlNTUyIsImlkIjoxODgyLCJ0eXBlIjoidXNlci1hcHBsaWNhdGlvbiJ9LCJpYXQiOjE3MjgzNjM3NzgsImV4cCI6MjA0MzcyMzc3OCwiYXVkIjoiZ2Z3IiwiaXNzIjoiZ2Z3In0.cr93rUeqZ2jC2TYy96Tn7_bxk1pCdm5VAo4VukeIExFlkzzdl2oZSgDLmjocGiIK6WWDDeInMHCUuxuhZuiMsvXvkmrcNWRSWWiMafj65u2SWmg5VlVTuI88O3Nr0BNYUHTuJ1T9r7HGEwhz6dwPjlXh2SQyMZurHLcxVq6lsqKLp1BfLW8mti2b_3EXa6mO5m96-UH4Ng1t5iLyvArxYklzdDXyhlCfViZNJ0XEDG08ZyC_8ORTpY5eAbPrjyZZLArpYPDWXrIaKnHDh-td-puLR_BKCpGSQ9W8sMMtY4kYelg8oHqv8d7-MKbdzX8kqKppKL8AKGOeBQWGeqTsY2mIAOjhoWyVdoYT9ikbmhVfDw7r95Wlq0uLQRr8cC4ckuo2gS9XdGOIapvBLqmhs2FKPmCDY_y41QfQC7PMDb7b1_Dp4dFhOhlHHMY__sr3incwyFQWAA8YH-E5RxSgC_laZt-GJPn-bT4N7xjNeRWNB-MK-uweXUUnnm1YXcoE"
z = 0
x = 0
y = 0
output_path = "/Users/jonathanraphael/Downloads/sar_fixed_infrastructure.mvt"

command = f"""
    curl --location 'https://gateway.api.globalfishingwatch.org/v3/datasets/public-fixed-infrastructure-filtered:latest/context-layers/{z}/{x}/{y}' \
    -H "Authorization: Bearer {token}" \
    -o "{output_path}"
"""

os.system(command)
# %%


def view_mvt(file_path, z, x, y, extent=4096):
    """
    Reads an MVT file, decodes it, and plots the geometries in latitude and longitude.

    Parameters:
    - file_path: Path to the MVT file.
    - z: Zoom level of the tile.
    - x: X coordinate of the tile.
    - y: Y coordinate of the tile.
    - extent: The extent of the tile coordinate system (default is 4096).
    """
    # Step 1: Read the MVT file in binary mode
    with open(file_path, "rb") as f:
        mvt_data = f.read()

    # Step 2: Decode the MVT data
    decoded = mapbox_vector_tile.decode(mvt_data)

    # Step 3: Get the tile's bounding box in lat/lon
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    min_lon, min_lat, max_lon, max_lat = bounds

    # Step 4: Create a transformer from tile coordinates to lat/lon
    # Calculate scaling factors
    scale_x = (max_lon - min_lon) / extent
    scale_y = (max_lat - min_lat) / extent

    def tile_to_lonlat(x_coord, y_coord):
        lon = min_lon + (x_coord * scale_x)
        lat = min_lat + (y_coord * scale_y)
        return (lon, lat)

    # Redefine transform_coords to accept two arguments
    def transform_coords(x, y):
        return tile_to_lonlat(x, y)

    # Step 5: Extract and transform features
    features = []
    for layer_name, layer in decoded.items():
        for feature in layer.get("features", []):
            geom = shape(feature["geometry"])  # Shapely geometry in tile coordinates

            # Transform tile coordinates to lat/lon
            try:
                geom_lonlat = transform(transform_coords, geom)
            except Exception as e:
                print(f"Error transforming geometry: {e}")
                continue  # Skip this feature if transformation fails

            props = feature.get("properties", {})
            features.append({**props, "geometry": geom_lonlat, "geom": geom})

    if not features:
        print("No features found in the MVT file.")
        return

    # Step 4: Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(features)

    # Optional: Set CRS if known (e.g., Web Mercator)
    gdf.set_crs(epsg=3857, inplace=True)
    return gdf


def multipoint_to_point(geom):
    """
    Converts MultiPoint to Point by keeping only the first coordinate pair.
    If geometry is already a Point, returns it unchanged.
    If geometry is None or not a valid geometry, returns None.
    """
    if geom is None:
        return None
    if isinstance(geom, MultiPoint):
        if len(geom.geoms) > 0:
            return geom.geoms[0]
        else:
            return None
    elif isinstance(geom, Point):
        return geom
    else:
        # If there are other geometry types, handle accordingly or return None
        return None


gdf = view_mvt(output_path, z, x, y)

confidence_colors = {"low": "blue", "medium": "orange", "high": "red"}

# %%
gdf["color"] = gdf["label_confidence"].map(confidence_colors)
gdf["geometry"] = gdf["geometry"].apply(multipoint_to_point)

# %%
oil = gdf[gdf["label"] == "oil"]
ax = oil.plot(
    figsize=(10, 10),  # Figure size (width, height) in inches
    alpha=0.5,  # Transparency of the polygons
    edgecolor="k",  # Edge color of the polygons
    color=oil["color"],  # Fill color based on the 'color' column
)
ax.set_aspect(2)

# %%
