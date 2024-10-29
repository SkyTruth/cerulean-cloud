# %%
"""
Slick Plus Infrastructure Confidence Score Calculator

Processes GeoJSON files to compute confidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""
# %load_ext autoreload
# %autoreload 2

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cerulean_cloud.cloud_function_ais_analysis.utils.asa import (  # noqa: E402
    associate_infra_to_slick,
)


def download_geojson(id, download_path="/Users/jonathanraphael/Downloads"):
    """
    Downloads a GeoJSON file from the specified URL if it hasn't been downloaded already.

    Parameters:
    - id (int): The unique identifier for the GeoJSON item.
    - download_path (str): The directory path where the GeoJSON will be saved.

    Returns:
    - geojson_file_path (str): The file path to the downloaded GeoJSON.
    """
    url = f"https://api.cerulean.skytruth.org/collections/public.slick_plus/items?id={id}&f=geojson"
    geojson_file_path = os.path.join(download_path, f"{id}.geojson")

    if not os.path.exists(geojson_file_path):
        print(f"Downloading GeoJSON file for ID {id}...")
        os.system(f'curl "{url}" -o "{geojson_file_path}"')
        print(f"Downloaded GeoJSON to {geojson_file_path}")
    else:
        print(f"GeoJSON file already exists at {geojson_file_path}. Skipping download.")

    return geojson_file_path


def generate_infrastructure_points(
    combined_geometry, num_points, expansion_factor=0.2, crs="epsg:4326"
):
    """
    Generates random infrastructure points within an expanded bounding box of the combined geometry.

    Parameters:
    - combined_geometry (GeometryCollection | MultiPolygon | Polygon): The combined geometry.
    - num_points (int): Number of infrastructure points to generate.
    - expansion_factor (float): Fraction to expand the bounding box.

    Returns:
    - infra_gdf (GeoDataFrame): GeoDataFrame of infrastructure points.
    """
    minx, miny, maxx, maxy = combined_geometry.bounds
    width = maxx - minx
    height = maxy - miny
    infra_x = np.random.uniform(
        minx - expansion_factor * width, maxx + expansion_factor * width, num_points
    )
    infra_y = np.random.uniform(
        miny - expansion_factor * height, maxy + expansion_factor * height, num_points
    )
    infra_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(infra_x, infra_y), crs=crs)
    return infra_gdf


def plot_confidence(
    infra_gdf,
    confidence_scores,
    polygons,
    overall_centroid,
):
    """
    Plots a sample of infrastructure points with their confidence scores.

    Parameters:
    - infra_points (np.ndarray): Array of infrastructure points coordinates.
    - confidence_scores (np.ndarray): Array of confidence scores.
    - polygons (list of Polygon): List of Polygon objects.
    - overall_centroid (np.ndarray): Coordinates of the overall centroid (x, y).
    - sample_size (int): Number of points to plot.
    - id (int): Identifier for the plot title.
    """
    sample_size = len(infra_gdf)
    plt.figure(figsize=(10, 10))

    # Plot the polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.plot(x, y, "r", linewidth=3.0)

    # Plot infrastructure points with confidence scores
    plt.scatter(
        infra_gdf.geometry.x[:sample_size],
        infra_gdf.geometry.y[:sample_size],
        c=confidence_scores[:sample_size],
        cmap="Blues",
        s=10,
        vmin=0,
        vmax=1,
        alpha=0.6,
    )

    # Plot the centroid
    plt.plot(
        overall_centroid[0], overall_centroid[1], "k+", markersize=10, label="Centroid"
    )

    plt.colorbar(label="Confidence")
    plt.title(f"Slick ID {id}: Max Confidence {round(confidence_scores.max(), 2)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


# %%
# Usage example

# Sample parameters
id = 3342876
geojson_file_path = download_geojson(id)

# Plotting parameters
num_infra_points = 50000  # Number of infrastructure points
plot_sample = True

slick_gdf = gpd.read_file(geojson_file_path)
infra_gdf = generate_infrastructure_points(
    slick_gdf.geometry.values[0], num_infra_points
)

confidence_scores = associate_infra_to_slick(infra_gdf, slick_gdf)

plot_confidence(
    infra_gdf,
    confidence_scores,
    slick_gdf.geometry.iloc[0].geoms,
    slick_gdf.centroid.iloc[0].coords[0],
)
# %%
