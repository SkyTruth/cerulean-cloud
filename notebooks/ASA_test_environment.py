# %%
"""
Slick Plus Infrastructure Coincidence Score Calculator

Processes GeoJSON files to compute coincidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""

# %load_ext autoreload
# %autoreload 2

import json
import os
import sys
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from geoalchemy2 import WKTElement
from matplotlib.patches import Patch

load_dotenv(".env")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from cerulean_cloud.cloud_function_asa.utils.analyzer import (  # noqa: E402; NaturalAnalyzer
    ASA_MAPPING,
    DarkAnalyzer,
    InfrastructureAnalyzer,
    SourceAnalyzer,
)


def download_geojson(
    id, download_path=os.getenv("ASA_DOWNLOAD_PATH"), use_test_db=False
):
    """
    Downloads a GeoJSON file from the specified URL if it hasn't been downloaded already.

    Parameters:
    - id (int): The unique identifier for the GeoJSON item.
    - download_path (str): The directory path where the GeoJSON will be saved.

    Returns:
    - geojson_file_path (str): The file path to the downloaded GeoJSON.
    """
    url = f"https://api.cerulean.skytruth.org/collections/public.slick_plus/items?id={id}&f=geojson"
    if use_test_db:
        url = f"https://cerulean-cloud-test-cr-tipg-5qkjkyomta-ew.a.run.app/collections/public.slick_plus/items?id={id}&f=geojson"
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


def plot(analyzers, slick_id, black=True, num_ais=5):
    """
    Combines the plots of AIS buffered geometries and infrastructure coincidence scores,
    with a legend showing colors corresponding to each st_name.

    Parameters:
    - analyzers (dict): Dictionary containing Analyzer objects with keys 1 and/or 2.
        - Key 1: AIS Analyzer containing ais_buffered, results, and slick_gdf.
        - Key 2: Infrastructure Analyzer containing infra_gdf, coincidence_scores, and slick_gdf.
    - slick_id (int): Identifier for the plot title.
    - black (bool): Whether to use black borders for the infrastructure points.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize variables
    ais_analyzer = None
    infra_analyzer = None
    dark_analyzer = None
    slick_gdf = None

    # Assign analyzers based on keys
    if 1 in analyzers.keys():
        ais_analyzer = analyzers[1]
        slick_gdf = analyzers[1].slick_gdf
    if 2 in analyzers.keys():
        infra_analyzer = analyzers[2]
        slick_gdf = analyzers[
            2
        ].slick_gdf  # This will overwrite if both 1 and 2 are present
    if 3 in analyzers.keys():
        dark_analyzer = analyzers[3]
        slick_gdf = analyzers[
            3
        ].slick_gdf  # This will overwrite if both 1 and 2 and 3are present

    # Check if slick_gdf is assigned
    if slick_gdf is None:
        raise ValueError(
            "No slick_gdf found in analyzers. Ensure that analyzers[1] or analyzers[2] is provided."
        )

    # Plot the AIS buffered geometries if ais_analyzer is present
    if ais_analyzer is not None:
        # Ensure 'st_name' exists in both ais_buffered and results
        if (
            "st_name" in ais_analyzer.ais_buffered.columns
            and "st_name" in ais_analyzer.results.columns
        ):
            ranked_results = (
                ais_analyzer.results.sort_values("collated_score", ascending=False)
                .reset_index(drop=True)
                .iloc[:num_ais]
            )
            # Filter ais_buffered where st_name is in results.st_name
            filtered_ais_buffered = ais_analyzer.ais_buffered[
                ais_analyzer.ais_buffered["st_name"].isin(
                    ranked_results["st_name"].values
                )
            ]

            # Get unique st_name values
            unique_st_names = filtered_ais_buffered["st_name"].unique()
            num_st_names = len(unique_st_names)

            # Assign a unique color to each st_name using a colormap
            cmap = plt.get_cmap("rainbow", num_st_names)
            st_name_to_color = {
                st_name: cmap(i) for i, st_name in enumerate(unique_st_names)
            }

            # Plot each group of st_name with its corresponding color
            for st_name, group in filtered_ais_buffered.groupby("st_name"):
                group.plot(
                    ax=ax, color=st_name_to_color[st_name], alpha=0.2, label=st_name
                )

            # Create legend handles for st_name
            st_name_patches = [
                Patch(facecolor=st_name_to_color[st_name], label=st_name)
                for st_name in unique_st_names
            ]
        else:
            print(
                "Warning: 'st_name' column not found in ais_buffered or results. AIS buffered geometries will not be colored by 'st_name'."
            )
            # Plot AIS buffered geometries without color coding
            colors = plt.cm.get_cmap("viridis", len(ais_analyzer.ais_buffered))
            for row in ais_analyzer.ais_buffered.itertuples():
                geom = row.geometry
                gpd.GeoSeries(geom).plot(
                    ax=ax,
                    color=colors(row.Index),
                    alpha=0.8,
                    label="AIS Buffered" if row.Index == 0 else "",
                )
            st_name_patches = []

    # Plot the centroid
    centroid = slick_gdf.centroid.iloc[0]
    ax.plot(centroid.x, centroid.y, "k+", markersize=10, label="Centroid")

    # Plot the infrastructure points with coincidence scores if infra_analyzer is present
    if infra_analyzer is not None:
        scatter_infra = ax.scatter(
            infra_analyzer.infra_gdf.geometry.x,
            infra_analyzer.infra_gdf.geometry.y,
            c=infra_analyzer.coincidence_scores,
            cmap="Blues",
            s=10,
            vmin=0,
            vmax=1,
            alpha=0.8,
            edgecolor="blue" if black else None,
            label="Infrastructure Points",
        )
    else:
        scatter_infra = None

    # Plot the infrastructure points with coincidence scores if infra_analyzer is present
    if dark_analyzer is not None:
        scatter_dark = ax.scatter(
            dark_analyzer.dark_objects_gdf.geometry.x,
            dark_analyzer.dark_objects_gdf.geometry.y,
            c=dark_analyzer.coincidence_scores,
            cmap="Greens",
            s=10,
            vmin=0,
            vmax=1,
            alpha=0.8,
            edgecolor="green" if black else None,
            label="Dark Vessels",
        )
    else:
        scatter_dark = None

    # Plot the slick polygons
    slick_gdf.plot(
        edgecolor="red", linewidth=1, color="none", ax=ax, label="Slick Polygons"
    )

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

    # Initialize maximum coincidence values for each analyzer
    max_infra_coincidence = 0
    max_dark_coincidence = 0

    # Add colorbar for infrastructure points if available
    if scatter_infra is not None and not black:
        cbar_infra = plt.colorbar(scatter_infra, ax=ax, fraction=0.046, pad=0.04)
        cbar_infra.set_label("Infra Coincidence")
        if len(infra_analyzer.coincidence_scores):
            max_infra_coincidence = round(infra_analyzer.coincidence_scores.max(), 2)

    # Add colorbar for dark vessel points if available
    if scatter_dark is not None and not black:
        cbar_dark = plt.colorbar(scatter_dark, ax=ax, fraction=0.046, pad=0.04)
        cbar_dark.set_label("Dark Coincidence")
        if len(dark_analyzer.coincidence_scores):
            max_dark_coincidence = round(dark_analyzer.coincidence_scores.max(), 2)

    # Optionally, combine the two maximums if needed:
    max_coincidence = max(max_infra_coincidence, max_dark_coincidence)

    # Set titles and labels
    plt.title(f"Slick ID {slick_id}: Max Coincidence {max_coincidence}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Add grid
    plt.grid(True)

    # Optionally, add a legend
    handles, labels = ax.get_legend_handles_labels()

    # If st_name_patches exist, add them to the legend
    if "st_name_patches" in locals() and st_name_patches:
        # Existing handles (e.g., Slick Polygons, Centroid, Infrastructure Points)
        # plus st_name_patches
        combined_handles = st_name_patches.copy()
        combined_labels = [patch.get_label() for patch in st_name_patches]

        # Get existing handles excluding st_name patches
        existing_handles = []
        existing_labels = []
        for handle, label in zip(handles, labels):
            if label not in combined_labels and label not in [
                "AIS Buffered"
            ]:  # Exclude AIS Buffered to avoid duplicates
                existing_handles.append(handle)
                existing_labels.append(label)

        combined_handles.extend(existing_handles)
        combined_labels.extend(existing_labels)

        plt.legend(handles=combined_handles, labels=combined_labels, title="st_name")
    else:
        if handles:
            # Remove duplicate labels
            unique = {}
            for handle, label in zip(handles, labels):
                if label not in unique:
                    unique[label] = handle
            handles, labels = zip(*unique.items())
            plt.legend(handles=handles, labels=labels)

    # Show the plot
    plt.show()


analyzers: dict[int, SourceAnalyzer] = {}

# %%
slick_ids = [
    # 3476096,  # ridges error
    # 3216961,  # ridges error
    # 3049976,
    # 3045541,
    # 3537529,  # indonesia
    # 3045541,  # infra
    # 3573155,  # T&T
    # 3571486,  # missing from GFW
    # 3581392,  # low score ais?
    # 3581329,  # vessel
    # 3819454, # infra
    # 3820550, # infra
    # 3830066,  # infra
    # 3581639,  # dark
    # 3687248,  # dark
    # 3582305,  # dark tricky
    # 3581812,  # dark easy
    # 3581500,  # dark easy
    # 3581468,  # dark easy
    # 3318875,  # dark tricky
    # 3582208,  # dark easy
    3581669,  # dark tricky
]

accumulated_sources = []
for slick_id in slick_ids:
    geojson_file_path = download_geojson(slick_id)
    slick_gdf = gpd.read_file(geojson_file_path)
    slick_gdf["centerlines"] = slick_gdf["centerlines"].apply(json.loads)
    s1_scene = get_s1_scene(slick_gdf.s1_scene_id.iloc[0])

    source_types = []
    source_types += [1]  # ais
    source_types += [2]  # infra
    source_types += [3]  # dark
    # source_types += [4]  # natural
    if not (  # If the last analyzer is for the same scene, reuse it
        analyzers
        and next(iter(analyzers.items()))[1].s1_scene.scene_id == s1_scene.scene_id
    ):
        analyzers = {s_type: ASA_MAPPING[s_type](s1_scene) for s_type in source_types}

    ranked_sources = pd.DataFrame(
        columns=["type", "ext_id", "coincidence_score", "collated_score"]
    )
    for s_type, analyzer in analyzers.items():
        res = analyzer.compute_coincidence_scores(slick_gdf)
        if res is not None:
            res = res.dropna(axis=1, how="all")

        ranked_sources = pd.concat([ranked_sources, res], ignore_index=True)

    print(f"{len(ranked_sources)} sources found for Slick ID: {slick_id}")
    if len(ranked_sources) > 0:
        ranked_sources = ranked_sources.sort_values(
            "collated_score", ascending=False
        ).reset_index(drop=True)
        accumulated_sources.append(
            [
                slick_id,
                ranked_sources["ext_id"].iloc[0],
                float(ranked_sources["collated_score"].iloc[0]),
            ]
        )

    plot(analyzers, slick_id)

    print(
        ranked_sources[["type", "ext_id", "coincidence_score", "collated_score"]].head(
            8
        )
    )

# print(accumulated_sources)

# %%
# Plot out all potential dark sources
fake_dark_gdf = generate_infrastructure_points(slick_gdf, 50000)
dark_analyzer = DarkAnalyzer(s1_scene, dark_vessels_gdf=fake_dark_gdf)
coincidence_scores = dark_analyzer.compute_coincidence_scores(slick_gdf)
plot({3: dark_analyzer}, slick_id, False)

# %%
# Plot out all potential infra sources
fake_infra_gdf = generate_infrastructure_points(slick_gdf, 50000)
infra_analyzer = InfrastructureAnalyzer(s1_scene, infra_gdf=fake_infra_gdf)
coincidence_scores = infra_analyzer.compute_coincidence_scores(slick_gdf)
plot({2: infra_analyzer}, slick_id, False)

# %%
