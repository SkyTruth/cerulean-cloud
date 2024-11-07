"""
This module facilitates the retrieval, processing, and visualization of infrastructure data from the Global Fishing Watch API. It downloads Mapbox Vector Tiles (MVT), transforms tile coordinates to geographic coordinates, processes geometric data, and generates visual plots based on label confidence levels.
"""

# %%

import os

import geopandas as gpd
import mapbox_vector_tile
from shapely.geometry import Point

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


def view_mvt(file_path):
    """
    Reads an MVT file, decodes it, and plots the geometries in latitude and longitude.

    Parameters:
    - file_path: Path to the MVT file.
    """
    with open(file_path, "rb") as f:
        mvt_data = f.read()
    decoded = mapbox_vector_tile.decode(mvt_data)

    features = []
    for detection in decoded["main"]["features"]:
        geom = Point(detection["properties"]["lon"], detection["properties"]["lat"])
        features.append({**detection["properties"], "geometry": geom})

    return gpd.GeoDataFrame(features, crs="epsg:4326")


gdf = view_mvt(output_path)

# %%
confidence_colors = {"low": "blue", "medium": "orange", "high": "red"}
gdf["color"] = gdf["label_confidence"].map(confidence_colors)

oil = gdf[gdf["label"] == "oil"]
ax = oil.plot(
    figsize=(10, 10),  # Figure size (width, height) in inches
    alpha=0.5,  # Transparency of the polygons
    edgecolor="k",  # Edge color of the polygons
    color=oil["color"],  # Fill color based on the 'color' column
)

# %%
