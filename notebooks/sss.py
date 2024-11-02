# %%
"""
Slick Plus Infrastructure Coincidence Score Calculator

Processes GeoJSON files to compute coincidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""
# %load_ext autoreload
# %autoreload 2

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from cerulean_cloud.cloud_function_ais_analysis.utils.analyzer import (
    InfrastructureAnalyzer,
)

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
    slick_gdf, num_points, expansion_factor=0.2, crs="epsg:4326"
):
    """
    Generates random infrastructure points within an expanded bounding box of the combined geometry.

    Parameters:
    - slick_gdf (GeoDataFrame): GeoDataFrame containing slick polygons.
    - num_points (int): Number of infrastructure points to generate.
    - expansion_factor (float): Fraction to expand the bounding box.

    Returns:
    - infra_gdf (GeoDataFrame): GeoDataFrame of infrastructure points.
    """
    minx, miny, maxx, maxy = slick_gdf.total_bounds
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


def plot_coincidence(
    infra_gdf,
    slick_gdf,
    coincidence_scores,
    id,  # Added 'id' as a parameter
    black=True,
):
    """
    Plots a sample of infrastructure points with their coincidence scores.

    Parameters:
    - infra_gdf (GeoDataFrame): GeoDataFrame containing infrastructure points.
    - slick_gdf (GeoDataFrame): GeoDataFrame containing slick polygons.
    - coincidence_scores (np.ndarray): Array of coincidence scores.
    - id (int): Identifier for the plot title.
    """
    sample_size = len(infra_gdf)
    plt.figure(figsize=(10, 10))

    # Create an axes object
    ax = plt.gca()

    # First plot the infrastructure points
    scatter = ax.scatter(
        infra_gdf.geometry.x[:sample_size],
        infra_gdf.geometry.y[:sample_size],
        c=coincidence_scores[:sample_size],
        cmap="Blues",
        s=10,
        vmin=0,
        vmax=1,
        alpha=0.6,
        edgecolor="black" if black else None,  # Adds black borders
        # linewidth=0.5,  # Optional: adjust border thickness
        label="Infrastructure Points",
    )

    # Then plot the slick_gdf polygons on top
    slick_gdf.plot(
        edgecolor="red", linewidth=1, color="none", ax=ax, label="Slick Polygons"
    )

    # Optionally, plot the centroid on top
    centroid = slick_gdf.centroid.iloc[0]
    ax.plot(centroid.x, centroid.y, "k+", markersize=10, label="Centroid")

    # Set plot limits with padding
    min_x, min_y, max_x, max_y = slick_gdf.total_bounds
    padding_ratio = 0.2

    width = max_x - min_x
    height = max_y - min_y

    padding_x = width * padding_ratio
    padding_y = height * padding_ratio

    # Apply padding
    min_x_padded = min_x - padding_x
    max_x_padded = max_x + padding_x
    min_y_padded = min_y - padding_y
    max_y_padded = max_y + padding_y

    # Determine the larger dimension
    width_padded = max_x_padded - min_x_padded
    height_padded = max_y_padded - min_y_padded

    if width_padded > height_padded:
        # Width is the larger dimension
        extra_height = width_padded - height_padded
        min_y_final = min_y_padded - extra_height / 2
        max_y_final = max_y_padded + extra_height / 2
        min_x_final = min_x_padded
        max_x_final = max_x_padded
    else:
        # Height is the larger dimension
        extra_width = height_padded - width_padded
        min_x_final = min_x_padded - extra_width / 2
        max_x_final = max_x_padded + extra_width / 2
        min_y_final = min_y_padded
        max_y_final = max_y_padded

    ax.set_xlim(min_x_final, max_x_final)
    ax.set_ylim(min_y_final, max_y_final)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Coincidence")

    # Set titles and labels
    plt.title(f"Slick ID {id}: Max Coincidence {round(coincidence_scores.max(), 2)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Remove or adjust the aspect ratio
    # plt.axis("equal")  # Removed to prevent overriding limits

    # Add grid
    plt.grid(True)

    # Optionally, add a legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, labels=labels)

    # Show the plot
    plt.show()

    # Process and print the infrastructure GeoDataFrame with coincidence scores
    copy_infra_gdf = infra_gdf.copy()  # To avoid SettingWithCopyWarning
    copy_infra_gdf["coincidence_score"] = coincidence_scores
    copy_infra_gdf = copy_infra_gdf.sort_values(by="coincidence_score", ascending=False)
    copy_infra_gdf.reset_index(drop=True, inplace=True)
    print(copy_infra_gdf[copy_infra_gdf["coincidence_score"] > 0].head())


# %%
# Usage example

# Sample parameters
id = 3115674
geojson_file_path = download_geojson(id)
slick_gdf = gpd.read_file(geojson_file_path)
scene_id = "S1A_IW_GRDH_1SDV_20230612"

ia = InfrastructureAnalyzer(slick_gdf, scene_id)
res = ia.associate_sources_to_slicks()
plot_coincidence(ia.infra_gdf, ia.slick_gdf, ia.coincidence_scores, id)

# %%
infra_gdf = generate_infrastructure_points(slick_gdf, 50000)
coincidence_scores = associate_infra_to_slick(infra_gdf, slick_gdf)
plot_coincidence(infra_gdf, slick_gdf, coincidence_scores, id, False)

# %%
