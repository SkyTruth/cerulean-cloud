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
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from geoalchemy2 import WKTElement

load_dotenv(".env")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from cerulean_cloud.cloud_function_ais_analysis.utils.analyzer import (  # noqa: E402
    ASA_MAPPING,
    InfrastructureAnalyzer,
    SourceAnalyzer,
)


def download_geojson(id, download_path=os.getenv("ASA_DOWNLOAD_PATH")):
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


def get_s1_scene(scene_id, download_path=os.getenv("ASA_DOWNLOAD_PATH")):
    """
    Downloads a S1 scene GeoJSON file from the specified URL if it hasn't been downloaded already.
    """
    url = f"https://api.cerulean.skytruth.org/collections/public.sentinel1_grd/items?scene_id={scene_id}&f=geojson"
    geojson_file_path = os.path.join(download_path, f"{scene_id}.geojson")
    if not os.path.exists(geojson_file_path):
        print(f"Downloading GeoJSON file for Scene {scene_id}...")
        os.system(f'curl "{url}" -o "{geojson_file_path}"')
        print(f"Downloaded GeoJSON to {geojson_file_path}")
    else:
        print(f"GeoJSON file already exists at {geojson_file_path}. Skipping download.")
    s1_gdf = gpd.read_file(geojson_file_path)
    s1_scene = SimpleNamespace(
        scene_id=scene_id,
        scihub_ingestion_time=s1_gdf.scihub_ingestion_time.iloc[0],
        start_time=s1_gdf.start_time.iloc[0],
        end_time=s1_gdf.end_time.iloc[0],
        geometry=WKTElement(str(s1_gdf.geometry.iloc[0])),
    )
    return s1_scene


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
    df = pd.DataFrame(
        {
            "structure_start_date": [pd.Timestamp(0)] * num_points,
            "structure_end_date": [pd.Timestamp.now()] * num_points,
        }
    )
    infra_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(infra_x, infra_y), crs=crs
    )
    return infra_gdf


def plot_coincidence(
    analyzer,
    slick_id,
    black=True,
):
    """
    Plots a sample of infrastructure points with their coincidence scores.

    Parameters:
    - analyzer (SourceAnalyzer): Analyzer object containing infrastructure points and coincidence scores.
    - slick_id (int): Identifier for the plot title.
    - black (bool): Whether to use black borders for the infrastructure points.
    """

    sample_size = len(analyzer.infra_gdf)
    plt.figure(figsize=(10, 10))

    # Create an axes object
    ax = plt.gca()

    # First plot the infrastructure points
    scatter = ax.scatter(
        analyzer.infra_gdf.geometry.x[:sample_size],
        analyzer.infra_gdf.geometry.y[:sample_size],
        c=analyzer.coincidence_scores[:sample_size],
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
    analyzer.slick_gdf.plot(
        edgecolor="red", linewidth=1, color="none", ax=ax, label="Slick Polygons"
    )

    # Optionally, plot the centroid on top
    centroid = analyzer.slick_gdf.centroid.iloc[0]
    ax.plot(centroid.x, centroid.y, "k+", markersize=10, label="Centroid")

    # Set plot limits with padding
    min_x, min_y, max_x, max_y = analyzer.slick_gdf.total_bounds
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

    max_coincidence = (
        round(analyzer.coincidence_scores.max(), 2)
        if len(analyzer.coincidence_scores)
        else 0
    )

    # Set titles and labels
    plt.title(f"Slick ID {slick_id}: Max Coincidence {max_coincidence}")
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


analyzers: dict[int, SourceAnalyzer] = {}

# %%
slick_ids = [
    # 3476096,
    # 3216961,
    # 3049976,
    # 3045541
    # 3537529, # indonesia
    3045541,  # infra
]

accumulated_sources = []
for slick_id in slick_ids:
    geojson_file_path = download_geojson(slick_id)
    slick_gdf = gpd.read_file(geojson_file_path)
    s1_scene = get_s1_scene(slick_gdf.s1_scene_id.iloc[0])

    source_types = []
    # source_types += [1]  # ais
    source_types += [2]  # infra
    if not (  # If the last analyzer is for the same scene, reuse it
        analyzers
        and next(iter(analyzers.items()))[1].s1_scene.scene_id == s1_scene.scene_id
    ):
        analyzers = {
            s_type: ASA_MAPPING[s_type](
                s1_scene,
                gfw_infra_filepath="/Users/jonathanraphael/git/cerulean-cloud/cerulean_cloud/cloud_function_ais_analysis/SAR Fixed Infrastructure 202407 DENOISED UNIQUE.csv",
            )
            for s_type in source_types
        }

    ranked_sources = pd.DataFrame(columns=["type", "st_name", "collated_score"])
    for s_type, analyzer in analyzers.items():
        res = analyzer.compute_coincidence_scores(slick_gdf)
        ranked_sources = pd.concat([ranked_sources, res], ignore_index=True)

    print(f"{len(ranked_sources)} sources found for Slick ID: {slick_id}")
    if len(ranked_sources) > 0:
        ranked_sources = ranked_sources.sort_values(
            "collated_score", ascending=False
        ).reset_index(drop=True)
        accumulated_sources.append(
            [
                slick_id,
                ranked_sources["st_name"].iloc[0],
                float(ranked_sources["collated_score"].iloc[0]),
            ]
        )

    if 2 in analyzers:
        plot_coincidence(analyzers[2], slick_id)

    print(ranked_sources[["type", "st_name", "collated_score"]].head())

    print(ranked_sources.head())
# print(accumulated_sources)
# %%
fake_infra_gdf = generate_infrastructure_points(slick_gdf, 50000)
infra_analyzer = InfrastructureAnalyzer(s1_scene, infra_gdf=fake_infra_gdf)
coincidence_scores = infra_analyzer.compute_coincidence_scores(slick_gdf)
plot_coincidence(infra_analyzer, slick_id, False)
