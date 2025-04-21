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
import time
from types import SimpleNamespace
from shapely.geometry import shape, Point
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.collections import LineCollection
from geoalchemy2 import WKTElement
from matplotlib.patches import Patch
from shapely.geometry import MultiLineString

load_dotenv(".env")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from cerulean_cloud.cloud_function_asa.utils.analyzer import (  # noqa: E402; NaturalAnalyzer
    ASA_MAPPING,
    DarkAnalyzer,
    InfrastructureAnalyzer,
    SourceAnalyzer,
)


def plot_ais_trajectory(traj_gdf, raw_gdf):
    """
    Plot the AIS trajectory as a gradient line from red (earlier) to blue (later)
    and overlay the raw AIS data as points.

    traj_gdf: GeoDataFrame with the trajectory data. Its index must be timestamps.
    raw_gdf: GeoDataFrame with raw AIS measurements; must have a 'timestamp' column.
    """
    # Ensure the GeoDataFrames are sorted by time
    traj_gdf = traj_gdf.sort_index()
    raw_gdf = raw_gdf.sort_values("timestamp")

    # Extract the coordinate pairs for the trajectory line
    traj_coords = np.array([[pt.x, pt.y] for pt in traj_gdf.geometry])

    # Create line segments connecting each consecutive trajectory point
    segments = np.array([traj_coords[i : i + 2] for i in range(len(traj_coords) - 1)])

    # Convert timestamps to numeric seconds.
    # Here, we convert the DatetimeIndex (ns resolution) to seconds.
    timestamps = traj_gdf.index.astype(np.int64) / 1e9
    # Compute average timestamp for each segment to assign its color.
    seg_timestamps = (timestamps[:-1] + timestamps[1:]) / 2.0

    # Normalize timestamp values to the range [0,1].
    norm = plt.Normalize(timestamps.min(), timestamps.max())
    # Use the reversed "RdBu" colormap so that low values (earlier) are red and high values (later) are blue.
    cmap = plt.get_cmap("RdBu_r")

    # Create the line collection for the trajectory segments.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_timestamps)
    lc.set_linewidth(2)

    # Begin plotting.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(lc)
    ax.autoscale_view()

    # Plot the raw AIS measurements as scatter points.
    raw_coords = np.array([[pt.x, pt.y] for pt in raw_gdf.geometry])
    ax.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        color="green",
        edgecolor="k",
        marker="o",
        label="Raw AIS Data",
        zorder=5,
    )

    # Add a colorbar that maps the line color to the timestamps.
    _ = plt.colorbar(lc, ax=ax, label="Timestamp (s)")
    ax.set_title("AIS Trajectory (Red-to-Blue Gradient) and Raw Measurements")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.show()


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
    try:
        s1_scene = SimpleNamespace(
            scene_id=scene_id,
            scihub_ingestion_time=s1_gdf.scihub_ingestion_time.iloc[0],
            start_time=s1_gdf.start_time.iloc[0],
            end_time=s1_gdf.end_time.iloc[0],
            geometry=WKTElement(str(s1_gdf.geometry.iloc[0])),
        )
    except AttributeError as e:
        print(
            f"Scene {scene_id} has not been processed on PROD yet. Delete downloaded empty geojson and try again. {e}"
        )
        raise e
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


def get_closest_centerline_points(
    traj_gdf: gpd.GeoDataFrame,
    longest_centerline: MultiLineString,
    t_image: datetime = None,
) -> tuple[Point, pd.Timestamp, float, Point, pd.Timestamp, float]:
    """
    Returns the timestamp and distance of the closest points on the centerline to the vessel at the given image_timestamp.
    """
    # Create centerline endpoints
    cl_A = Point(longest_centerline.coords[0])
    cl_B = Point(longest_centerline.coords[-1])

    # Find nearest trajectory point indices for each endpoint
    t_A, d_A = get_closest_point_near_timestamp(cl_A, traj_gdf, t_image)
    t_B, d_B = get_closest_point_near_timestamp(cl_B, traj_gdf, t_image)

    # Sort the pairs by timestamp to determine head and tail
    (cl_tail, t_tail, d_tail), (cl_head, t_head, d_head) = sorted(
        [(cl_A, t_A, d_A), (cl_B, t_B, d_B)], key=lambda x: x[1]
    )

    # After finding the head (slick end closest to the AIS), project the tail to the nearest point independent of time.
    t_tail, d_tail = get_closest_point_near_timestamp(cl_tail, traj_gdf)

    return (cl_tail, t_tail, d_tail, cl_head, t_head, d_head)


def get_closest_point_near_timestamp(
    target: Point,
    traj_gdf: gpd.GeoDataFrame,
    t_image: datetime = None,
    n_points: int = 10,
) -> tuple[pd.Timestamp, float]:
    """
    Returns the trajectory row that is closest to the reference_point,
    using a turning-point heuristic starting at t_image.

    It starts at t_image and checks the immediate neighbors to determine
    in which temporal direction the distance to the reference point is decreasing.
    It then traverses in that single direction until the distance no longer decreases,
    returning the last point before an increase is detected.

    Parameters:
        reference_point (shapely.geometry.Point): The point to compare distances to.
        traj_gdf (geopandas.GeoDataFrame): A GeoDataFrame with a datetime-like index
            and a 'geometry' column.
        t_image (datetime): The starting timestamp for the search (guaranteed to be in the dataset).
        n_points (int): The number of subsequent points that must confirm the turning point.

    Returns:
        pd.Timestamp: The index corresponding to the selected trajectory row.
    """
    if t_image is None:
        # If no timestamp is provided, return the index of the point closest to the target.
        distances = traj_gdf.geometry.distance(target)
        return distances.idxmin(), distances.min()

    # Get the starting position for t_image.
    traj_gdf = traj_gdf.sort_index(ascending=True)
    pos = np.abs(traj_gdf.index - t_image).argmin()
    best_dist = traj_gdf.iloc[pos].geometry.distance(target)

    # Determine direction to traverse.
    if pos == 0:
        direction = 1
    else:
        backward_distance = traj_gdf.iloc[pos - 1].geometry.distance(target)
        # Pick the direction with a decreasing distance.
        direction = -1 if backward_distance < best_dist else 1

    while 0 <= pos + direction < len(traj_gdf):
        pos += direction
        d = traj_gdf.iloc[pos].geometry.distance(target)
        if d < best_dist:
            best_dist = d
        else:
            # Get the next n_points indices in the chosen direction that are within bounds.
            check_indices = [
                pos + i * direction
                for i in range(1, n_points + 1)
                if 0 <= pos + i * direction < len(traj_gdf)
            ]
            # Check that for each of these indices, the distance is greater than best_dist.
            distances = np.array(
                [traj_gdf.iloc[idx].geometry.distance(target) for idx in check_indices]
            )
            if all(distances > best_dist):
                break
    return traj_gdf.index[pos], best_dist


def plot(analyzers, slick_id, black=True, num_ais=5):
    """
    Combines the plots of AIS buffered geometries and infrastructure coincidence scores,
    with a legend showing colors corresponding to each st_name.

    Parameters:
    - analyzers (dict): Dictionary containing Analyzer objects with keys 1 and/or 2.
        - Key 1: AIS Analyzer containing results, and slick_gdf.
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
        # Ensure 'st_name' exists in results
        if "st_name" in ais_analyzer.results.columns:
            ranked_results = (
                ais_analyzer.results.sort_values("collated_score", ascending=False)
                .reset_index(drop=True)
                .iloc[:num_ais]
            )
            # Filter  where st_name is in results.st_name
            filtered_ais = ais_analyzer.results[
                ais_analyzer.results["st_name"].isin(ranked_results["st_name"].values)
            ]

            # Get unique st_name values
            unique_st_names = filtered_ais["st_name"].unique()
            num_st_names = len(unique_st_names)

            # Assign a unique color to each st_name using a colormap
            cmap = plt.get_cmap("rainbow", num_st_names)
            st_name_to_color = {
                st_name: cmap(i) for i, st_name in enumerate(unique_st_names)
            }

            # Plot each group of st_name with its corresponding color
            for st_name, group in filtered_ais.groupby("st_name"):
                group.plot(
                    ax=ax, color=st_name_to_color[st_name], alpha=0.5, label=st_name
                )

            # Create legend handles for st_name
            st_name_patches = [
                Patch(facecolor=st_name_to_color[st_name], label=st_name)
                for st_name in unique_st_names
            ]
        else:
            print(
                "Warning: 'st_name' column not found in results. AIS buffered geometries will not be colored by 'st_name'."
            )
            # Plot AIS  geometries without color coding
            colors = plt.cm.get_cmap("viridis", len(ais_analyzer.results))
            for row in ais_analyzer.results.itertuples():
                geom = row.geometry
                gpd.GeoSeries(geom).plot(
                    ax=ax,
                    color=colors(row.Index),
                    alpha=0.8,
                    label="AIS" if row.Index == 0 else "",
                )
            st_name_patches = []

        for st_name, group in filtered_ais.groupby("st_name"):
            # Loop over consecutive points and add arrows
            for idx, row in group.iterrows():
                geom = row.geometry
                # Ensure we're dealing with a LineString
                if geom.geom_type == "LineString":
                    # Place arrow_count arrows spaced evenly along the trajectory.
                    arrow_count = 3
                    # We'll compute fractional positions along the line's length.
                    for i in range(arrow_count):
                        # Compute a fraction that splits the line into 11 segments
                        frac = (i + 1) / (arrow_count + 1)

                        # Get the base point for the arrow using interpolation
                        base_point = geom.interpolate(frac, normalized=True)

                        # To determine direction, compute a point a small fraction ahead
                        # Make sure we don't go beyond the end of the line.
                        epsilon = 0.001
                        next_frac = min(frac + epsilon, 1.0)
                        next_point = geom.interpolate(next_frac, normalized=True)

                        # Draw the arrow from the base point toward the next point
                        ax.annotate(
                            "",
                            xy=(next_point.x, next_point.y),
                            xytext=(base_point.x, base_point.y),
                            arrowprops=dict(
                                arrowstyle="->", color=st_name_to_color[st_name], lw=1
                            ),
                        )

        # Add a marker dot at the point closest to s1_time for each vessel
        # Define the timestamp at which the marker should be placed
        s1_time = np.datetime64(ais_analyzer.s1_scene.start_time)

        for st_name, group in filtered_ais.groupby("st_name"):
            combined_features = []
            # Each row has a geojson_fc (a feature collection dict)
            for fc in group["geojson_fc"]:
                combined_features.extend(fc.get("features", []))

            # Skip if no features exist
            if len(combined_features) == 0:
                continue

            # Sort features by timestamp
            combined_features.sort(
                key=lambda feat: np.datetime64(feat["properties"]["timestamp"])
            )

            # Find the feature with timestamp closest to s1_time
            closest_feature = min(
                combined_features,
                key=lambda feat: abs(
                    np.datetime64(feat["properties"]["timestamp"]) - s1_time
                ),
            )

            # Plot small markers (size 2) for every feature except the closest one
            for feat in combined_features:
                geom = shape(feat["geometry"])
                x, y = geom.x, geom.y
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=1 if feat is not closest_feature else 2,
                    color=st_name_to_color.get(st_name, "black")
                    if feat is not closest_feature
                    else "black",
                )

        for st_name, group in filtered_ais.groupby("st_name"):
            # Obtain the longest centerline for the st_name; assumes slick_centerlines has a 'st_name' column
            gdf = ais_analyzer.ais_trajectories[st_name]["df"]
            longest_centerline = (
                ais_analyzer.slick_centerlines.sort_values("length", ascending=False)
                .iloc[0]
                .geometry
            )
            s1_time = ais_analyzer.s1_scene.start_time
            (cl_tail, t_tail, d_tail, cl_head, t_head, d_head) = (
                get_closest_centerline_points(gdf, longest_centerline, s1_time)
            )
            # Compute nearest trajectory point to the start and end coordinates of the longest centerline
            start_traj_point = gdf.loc[[t_tail]].iloc[0]["geometry"]
            end_traj_point = gdf.loc[[t_head]].iloc[0]["geometry"]
            # Plot dotted lines connecting the endpoints to their nearest trajectory points
            ax.plot(
                [cl_tail.x, start_traj_point.x],
                [cl_tail.y, start_traj_point.y],
                linestyle=":",
                color=st_name_to_color.get(st_name, "black"),
            )
            ax.plot(
                [cl_head.x, end_traj_point.x],
                [cl_head.y, end_traj_point.y],
                linestyle=":",
                color=st_name_to_color.get(st_name, "black"),
            )

            if s1_time in ais_analyzer.ais_trajectories[st_name]["df"].index:
                geom = ais_analyzer.ais_trajectories[st_name]["df"].geometry.loc[
                    s1_time
                ]
                ax.plot(
                    geom.x,
                    geom.y,
                    marker="d",
                    markersize=8,
                    markerfacecolor="none",
                    color=st_name_to_color.get(st_name, "black"),
                )

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
    # 3581756,  # vessel
    # 3518524,  # vessel
    # 3212229,  # infra
    # 3518454,  # vessel
    # 3582101,  # vessel
    # 3581643,  # vessel FAIL
    # 3518524,  # vessel
    # 3237854,  # vessel
    # 3581087,  # vessel
    # 3581091,  # vessel
    # 3573262,  # vessel
    # 3819454, # infra
    # 3820550, # infra
    # 3830066,  # infra
    # 22558,  # local
    # 23505,  # local
    # 27566,  # local
    # 20839,  # local
    # 27484,  # local
    # DARKS
    # 34314,
    # 34226,
    # 34321,
    # 38573,  # slow
    # 38583, # paired to 34251
    # 35734, # SLOW AIS
    # 38527, # slow
    # 34179,
    # 34236,
    # 34209,
    # 34216,
    # 34171,
    # 34268,
    # 34144,
    # 34205,
    # 34162,
    # 34201,
    # 34299,
    # 38660, # HARD WITHOUT SMART MAPPING
    # VESSELS
    # 34324,
    # 34385,
    # 34378,
    # 34537,
    # 34490,
    # 34362,
    # 34327,
    # 34366,
    # 34357,
    # 34332,
    # 34333,
    # 35611,
    # 38499, # UI breaker
    # 35744,
    # 36063,
    # 36557,
    # 38785,  # EXTRAPOLATION
    # 38749, # EXTRAPOLATION
    # 38531,
    # 39087,
    # 38546,
    # 38618,  # S1A_IW_GRDH_1SDV_20211210T094843_20211210T094908_040945_04DCDF_BD0B SO SO SLOWWWWW
    # 42317,  # not working for ethan
    # 38666,
    # 38745,
    # 38752,
    # 39115, # three source slick...
    # 38941, # Do we need to increase the slick drift rate?
]

accumulated_sources = []
for slick_id in slick_ids:
    geojson_file_path = download_geojson(slick_id, use_test_db=True)
    slick_gdf = gpd.read_file(geojson_file_path)
    slick_gdf["centerlines"] = slick_gdf["centerlines"].apply(json.loads)
    s1_scene = get_s1_scene(slick_gdf.s1_scene_id.iloc[0])
    start_time = time.time()

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

    print(
        f"Time taken: {time.time() - start_time} seconds \n"
        f"Time per {len(ranked_sources)} sources: {(time.time() - start_time) / (len(ranked_sources) or 1)} seconds"
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
a = analyzers[1]
ssvid = "477810800"
traj = a.ais_trajectories[ssvid]["df"]
raw = a.ais_gdf[a.ais_gdf["ssvid"] == ssvid]
plot_ais_trajectory(traj, raw)

# %%
