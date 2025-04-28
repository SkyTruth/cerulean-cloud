import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import json

from geoalchemy2 import WKTElement
from types import SimpleNamespace
from tqdm.auto import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import shape, MultiLineString, LineString, Point
from datetime import datetime

from cerulean_cloud.centerlines import (
    calculate_centerlines,
)


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


def is_not_ais_infra(geom):
    """Returns the longer side of the minimum rotated rectangle."""
    if geom is None or geom.is_empty:
        return np.nan  # Handle empty geometries

    rotated_rect = geom.minimum_rotated_rectangle
    if isinstance(rotated_rect, LineString):
        return rotated_rect.length > 1000
    if isinstance(rotated_rect, Point):
        return False
    coords = np.array(rotated_rect.exterior.coords[:-1])  # Remove duplicate last point

    # Compute Euclidean distances between consecutive points
    side_lengths = np.linalg.norm(coords - np.roll(coords, shift=-1, axis=0), axis=1)

    # Identify unique side lengths and return the longer one
    # print(side_lengths)
    longest_length = sorted(side_lengths, reverse=True)[0]
    return longest_length > 1000


def label_dark_vessel_results_with_distance(
    results, slick_id, dark_vess_groundtruth, sort_metric="coincidence_score"
):
    """
    Labels dark vessel detection results with ground truth information based on spatial distance.

    Parameters:
        results (GeoDataFrame):
            The GeoDataFrame containing detection results, including a `coincidence_score` column.
        slick_id (str or int):
            The unique identifier for the slick being analyzed.
        dark_vess_groundtruth (GeoDataFrame):
            The ground truth GeoDataFrame containing only true dark vessel sources for each slick_id.
        sort_metric (str, optional):
            The column name used to sort the results before assigning ranks. Defaults to "coincidence_score".

    Returns:
        GeoDataFrame: The input results, sorted by the given metric, with added columns:
            - `rank`: Rank of detections based on the sorting metric.
            - `truth`: Boolean indicating whether each detection is within 0.005 degrees of a known dark vessel.

    Notes:
        - The `dark_vess_groundtruth` GeoDataFrame should only contain true dark vessel sources per `slick_id`.
        - The function assumes each `slick_id` in the ground truth corresponds to a single dark vessel location.
    """
    if results is None:
        return results

    # Filter ground truth to get only the relevant dark vessel for this slick_id
    dark_vess = dark_vess_groundtruth[dark_vess_groundtruth["slick_id"] == slick_id]

    # Sort results by the specified metric and assign ranking
    coin = results.sort_values(by=sort_metric, ascending=False)
    coin["rank"] = list(range(1, len(coin) + 1))

    # Compute distances from each detection to the true dark vessel location
    distances = coin.distance(dark_vess.geometry.iloc[0])

    # Assign truth labels based on proximity threshold (0.005 degrees)
    coin["truth"] = distances <= 0.005

    return coin


def label_results_with_st_name(
    results, slick_id, hitl_groundtruth, sort_metric="coincidence_score"
):
    """
    Labels detection results based on a string match with ground truth source names.

    Parameters:
        results (GeoDataFrame):
            The GeoDataFrame containing detection results, including a `coincidence_score` column.
        slick_id (str or int):
            The unique identifier for the slick being analyzed.
        hitl_groundtruth (GeoDataFrame):
            The ground truth GeoDataFrame containing only true sources for each slick_id, with an `st_name` column.
        sort_metric (str, optional):
            The column name used to sort the results before assigning ranks. Defaults to "coincidence_score".

    Returns:
        GeoDataFrame: The input results, sorted by the given metric, with added columns:
            - `rank`: Rank of detections based on the sorting metric.
            - `truth`: Boolean indicating whether each detection's `st_name` matches the true source name.

    Notes:
        - The `hitl_groundtruth` GeoDataFrame should contain only verified true sources per `slick_id`.
        - The function assumes each `slick_id` in the ground truth corresponds to a single `st_name`.
    """
    if results is None:
        return results

    # Filter ground truth to get the correct source name for this slick_id
    groundtruth = hitl_groundtruth[hitl_groundtruth["slick_id"] == slick_id]

    # Sort results by the specified metric and assign ranking
    coin = results.sort_values(by=sort_metric, ascending=False)
    coin["rank"] = list(range(1, len(coin) + 1))

    # Assign truth labels based on match with ground truth `st_name`
    coin["truth"] = coin["st_name"] == groundtruth["st_name"].values[0]

    return coin


def apply_labeling(results, groundtruth_gdf, labelling_method):
    # Group by slick_id and apply the labeling function to each group
    results["truth"] = False
    return results.groupby("slick_id", group_keys=False).apply(
        lambda grp: labelling_method(grp, grp["slick_id"].iloc[0], groundtruth_gdf)
    )


def rank_results(results, sort_metric="coincidence_score"):
    """
    Rerank the results using the specified metric
    """

    def rerank(results):
        coin = results.sort_values(by=sort_metric, ascending=False)
        coin["rank"] = list(range(1, len(coin) + 1))
        return coin

    return results.groupby("slick_id", group_keys=False).apply(lambda grp: rerank(grp))


def calculate_metrics(results, coin_score_column="coincidence_score"):
    """
    Calculate five metrics for each of the labelled results GeoDataFrames.

    Parameters:
        results: dict
            A dictionary where keys are method names (e.g., 'extrema', 'centerline')
            and values are GeoDataFrames that include columns 'coincidence_score', 'truth', and 'rank'.

    Returns:
        A DataFrame where rows are the methods and columns are the computed metrics.
    """
    metrics = {}

    for name, df in results.items():
        # Filter rows where the ground truth is True/False
        truth_df = df[df["truth"]]
        false_df = df[df["truth"].eq(False)]

        # Compute top 3 source rate (as a percentage)
        if len(truth_df) > 0:
            top3_rate = ((truth_df["rank"] <= 3).sum() / len(truth_df)) * 100
            top1_rate = ((truth_df["rank"] == 1).sum() / len(truth_df)) * 100
            avg_true = truth_df[coin_score_column].mean()
        else:
            top3_rate = np.nan
            top1_rate = np.nan
            avg_true = np.nan

        # Compute average false coincidence
        avg_false = false_df[coin_score_column].mean() if len(false_df) > 0 else np.nan

        # Compute the ratio of average true to average false coincidence scores
        true_false_ratio = (
            avg_true / avg_false
            if pd.notnull(avg_true) and pd.notnull(avg_false) and avg_false != 0
            else np.nan
        )

        # Store the metrics for this method
        metrics[name] = {
            "top_3_source_rate (%)": top3_rate,
            "top_1_source_rate (%)": top1_rate,
            "average_true_coincidence": avg_true,
            "average_false_coincidence": avg_false,
            "true_false_ratio": true_false_ratio,
        }

    return pd.DataFrame(metrics).T


def filter_and_truncate_trajectories(self):
    """
    Filters self.ais_trajectories spatially and temporally using the analyzer's properties.

    Uses:
      - self.ais_start_time: Lower bound of the time window.
      - self.ais_end_time: Upper bound of the time window.
      - self.slick_gdf: GeoDataFrame from which to derive the spatial area.
      - self.ais_buffered: Buffer distance used to create the search area.
    """
    # Create the buffered search area from slick_gdf
    search_area = (
        self.slick_gdf.geometry.to_crs(self.crs_meters)
        .buffer(self.ais_buffer)
        .to_crs("4326")
    )
    # Assuming the buffered search area is defined by the first geometry
    buffered_geometry = search_area.iloc[0]

    filtered_trajectories = {}
    for ssvid, traj in self.ais_trajectories.items():
        # Retrieve the full interpolated GeoDataFrame (with timestamps as index)
        traj_gdf = traj["df"]

        # Filter based on time: keep points between ais_start_time and ais_end_time.
        time_filtered_gdf = traj_gdf[
            (traj_gdf.index >= self.ais_start_time)
            & (traj_gdf.index <= self.ais_end_time)
        ]

        # Further filter based on spatial intersection with the buffered search area.
        spatially_filtered_gdf = time_filtered_gdf[
            time_filtered_gdf.geometry.intersects(buffered_geometry)
        ]

        # If any points remain, update the trajectory.
        if not spatially_filtered_gdf.empty:
            traj["df"] = spatially_filtered_gdf
            traj["geojson_fc"] = {
                "type": "FeatureCollection",
                "features": json.loads(spatially_filtered_gdf.to_json())["features"],
            }
            filtered_trajectories[ssvid] = traj

    # Update the trajectories with the filtered ones.
    return filtered_trajectories


def process_groundtruth_on_analyzer(
    analyzer_class,
    groundtruth_gdf,
    points_gdf=None,
    use_synthetic_points=0,
    synthetic_points_buffer=1000,
    analyzer_params=None,
    filter_ais_infra=False,
    post_filter_dark=False,
    reuse_ais_gdf=False,
    return_analyzer=False,
    pickle_dir=os.getenv("ASA_DOWNLOAD_PATH"),
):
    """
    Generalized function to process an analyzer over a ground truth GeoDataFrame.

    Parameters:
        analyzer_class: Class
            The analyzer class to be instantiated.
        groundtruth_gdf: GeoDataFrame
            The GeoDataFrame containing only the true sources (e.g., true dark vessel detections).
        points_gdf: GeoDataFrame, optional
            Optional points GeoDataFrame (e.g., SAR detections or infrastructure data) used for analysis.
        analyzer_params: dict, optional
            Parameters to be passed when initializing the analyzer instance.
        filter_ais_infra: bool, optional
            Whether to filter out AIS infrastructure results.
        reuse_ais_trajectories: bool, optional
            Whether to reuse previously saved AIS trajectories if available.
        pickle_dir: str, optional
            Directory where AIS trajectories pickle files are stored.

    Returns:
        GeoDataFrame containing accumulated results.
    """
    results_gdf = None  # Empty GeoDataFrame to accumulate results

    # Ensure the pickle directory exists
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    for ex_id in tqdm(range(len(groundtruth_gdf)), leave=False):
        groundtruth_source = groundtruth_gdf.iloc[[ex_id]].reset_index()

        slick_id = groundtruth_source["slick_id"].values[0]
        s1_scene_id = groundtruth_source["scene_id"].values[0]
        geojson_file_path = download_geojson(slick_id)
        slick_gdf = gpd.read_file(geojson_file_path)
        s1_scene = get_s1_scene(s1_scene_id)

        # Prepare parameters for the analyzer
        analyzer_kwargs = analyzer_params.copy() if analyzer_params else {}

        # If using AISAnalyzer, check for existing ais_trajectories pickle file
        ais_pickle_filename = pickle_dir + f"/ais_gdfs/{s1_scene_id}.pkl"

        if use_synthetic_points != 0:
            temp_analyzer = analyzer_class(s1_scene, infra_gdf="PLACEHOLDER")
            fake_dark_gdf = generate_infrastructure_points(
                slick_gdf, use_synthetic_points, expansion_factor=1.0
            )
            fake_dark_gdf_meters = fake_dark_gdf.to_crs(temp_analyzer.crs_meters)
            buff_geom = (
                slick_gdf.to_crs(temp_analyzer.crs_meters)
                .iloc[0]
                .geometry.buffer(synthetic_points_buffer)
            )
            points_gdf = fake_dark_gdf_meters[
                fake_dark_gdf_meters.within(buff_geom).eq(False)
            ].to_crs(slick_gdf.crs)
        # Instantiate the analyzer with parameters
        if points_gdf is not None:
            if analyzer_class.__name__ == "DarkAnalyzer":
                analyzer_kwargs["dark_vessels_gdf"] = points_gdf.reset_index()
            elif analyzer_class.__name__ == "InfrastructureAnalyzer":
                analyzer_kwargs["infra_gdf"] = points_gdf.reset_index()

        analyzer = analyzer_class(s1_scene, **analyzer_kwargs)

        centerline, arf = calculate_centerlines(
            slick_gdf, crs_meters=analyzer.crs_meters
        )
        slick_gdf["centerlines"] = [centerline]

        if analyzer_class.__name__ == "AISAnalyzer" and (reuse_ais_gdf):
            if os.path.exists(ais_pickle_filename):
                print("REUSING DOWNLOADED AIS GDF")
                with open(ais_pickle_filename, "rb") as f:
                    ais_gdf = pickle.load(f)
                ais_gdf = ais_gdf[
                    ais_gdf.geometry.within(analyzer.ais_envelope.iloc[0])
                ]
                ais_gdf = ais_gdf[ais_gdf["timestamp"] >= analyzer.ais_start_time]
                ais_gdf = ais_gdf[ais_gdf["timestamp"] <= analyzer.ais_end_time]
                analyzer.ais_gdf = ais_gdf
            else:
                print("AIS GDF FILE DOES NOT EXIST")

        res = analyzer.compute_coincidence_scores(slick_gdf)
        if res is None:
            continue

        res["slick_id"] = slick_id  # Track which slick_id it corresponds to
        print("FOUND ", len(res), "VESSEL RESULTS")
        if filter_ais_infra:
            res = res[
                res.to_crs(analyzer.crs_meters)["geometry"].apply(is_not_ais_infra)
            ]
            print("FILTERING DOWN TO", len(res), "VALID VESSEL")

        if analyzer_class.__name__ == "DarkAnalyzer" and post_filter_dark:
            res = res[res["structure_id"].isna()]
            res = res[res["ssvid"].isna()]
            res = res[res["length_m"] > 30]

        # Accumulate results using pd.concat
        res["arf"] = arf
        if results_gdf is None:
            results_gdf = res.copy()
        else:
            results_gdf = pd.concat([results_gdf, res], ignore_index=True)

    if return_analyzer:
        return results_gdf, analyzer
    return results_gdf


def plot_metrics(
    metrics_df,
    title="Evaluation Metrics",
    plot_ratio=False,
    legend_title="Method",
    value_font_size=8,
):
    """
    Plot metrics from three methods side by side using two subplots:
      - Left subplot: top source rates (scaled to 0-1) and average coincidence scores.
      - Right subplot: the ratio metric (true to false average coincidence), if plot_ratio is True.

    Each bar is annotated with its rounded value above it.

    Parameters:
        metrics_df: DataFrame with method names as rows and metric names as columns.
                    Expected columns:
                      'top_3_source_rate (%)', 'top_1_source_rate (%)',
                      'average_true_coincidence', 'average_false_coincidence', 'true_false_ratio'
        plot_ratio: Boolean flag to include or exclude the ratio plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Reset index so that method names become a column.
    df_plot = metrics_df.reset_index().rename(columns={"index": "Method"})

    # Define which metrics go to which subplot.
    left_metrics = [
        "top_3_source_rate (%)",
        "top_1_source_rate (%)",
        "average_true_coincidence",
        "average_false_coincidence",
    ]
    right_metrics = ["true_false_ratio"] if plot_ratio else []

    # Prepare data for the left subplot.
    df_left = df_plot.melt(
        id_vars="Method", value_vars=left_metrics, var_name="Metric", value_name="Value"
    )
    # Convert top source rate metrics from percentages to 0-1 scale.
    top_source_metrics = ["top_3_source_rate (%)", "top_1_source_rate (%)"]
    df_left.loc[df_left["Metric"].isin(top_source_metrics), "Value"] /= 100.0

    # Create subplots conditionally based on plot_ratio.
    num_cols = 2 if plot_ratio else 1
    fig, axes = plt.subplots(ncols=num_cols, figsize=(16 * num_cols, 8), sharey=False)

    if not plot_ratio:
        axes = [axes]  # Ensure consistent indexing when there's only one subplot.

    # Left subplot for 0-1 scaled metrics and averages.
    ax_left = axes[0]
    sns.barplot(data=df_left, x="Metric", y="Value", hue="Method", ax=ax_left)
    ax_left.set_title("Metrics (0-1 scale) excluding Ratio")
    ax_left.set_ylabel("Value")
    ax_left.tick_params(axis="x", rotation=45)

    # Annotate each bar on the left subplot.
    for container in ax_left.containers:
        for bar in container:
            height = bar.get_height()
            ax_left.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_font_size,
            )

    if plot_ratio:
        # Prepare data for the right subplot.
        df_right = df_plot.melt(
            id_vars="Method",
            value_vars=right_metrics,
            var_name="Metric",
            value_name="Value",
        )

        # Right subplot for the true/false ratio.
        ax_right = axes[1]
        sns.barplot(data=df_right, x="Metric", y="Value", hue="Method", ax=ax_right)
        ax_right.set_title("True/False Average Coincidence Ratio")
        ax_right.set_ylabel("Value")
        ax_right.tick_params(axis="x", rotation=45)

        # Annotate each bar on the right subplot.
        for container in ax_right.containers:
            for bar in container:
                height = bar.get_height()
                ax_right.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Remove duplicate legend from the right subplot.
        handles, labels = ax_left.get_legend_handles_labels()
        ax_left.legend(handles=handles, labels=labels, title=legend_title)
        ax_right.get_legend().remove()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_3d_scatter(
    points1, axis_labels, points2=None, points3=None, optimized_point=None
):
    """
    Plots a 3D scatter plot with up to three sets of points and an optional optimized point.

    Parameters:
    - points1: List of (x, y, z) tuples for the first dataset (required).
    - axis_labels: Tuple (x_label, y_label, z_label) for the axes (required).
    - points2: List of (x, y, z) tuples for the second dataset (optional).
    - points3: List of (x, y, z) tuples for the third dataset (optional).
    - optimized_point: Tuple (x, y, z) representing the optimized point (optional).
    """
    # Create the figure
    fig = go.Figure()

    # First dataset
    x1, y1, z1 = np.array(points1).transpose()
    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            mode="markers",
            marker=dict(
                size=6,
                color=z1,  # Use z-values for color
                colorscale="Blues",
                opacity=1.0,
            ),
            name="Dataset 1",
        )
    )

    # Second dataset (if provided)
    if points2 is not None:
        x2, y2, z2 = np.array(points2).transpose()
        fig.add_trace(
            go.Scatter3d(
                x=x2,
                y=y2,
                z=z2,
                mode="markers",
                marker=dict(
                    size=6,
                    color=z2,  # Use z-values for color
                    colorscale="Reds",
                    opacity=1.0,
                ),
                name="Dataset 2",
            )
        )

    # Third dataset (if provided)
    if points3 is not None:
        x3, y3, z3 = np.array(points3).transpose()
        fig.add_trace(
            go.Scatter3d(
                x=x3,
                y=y3,
                z=z3,
                mode="markers",
                marker=dict(
                    size=6,
                    color=z3,  # Use z-values for color
                    colorscale="Greens",
                    opacity=1.0,
                ),
                name="Dataset 3",
            )
        )

    # Optimized Point (if provided)
    if optimized_point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[optimized_point[0]],
                y=[optimized_point[1]],
                z=[optimized_point[2]],
                mode="markers",
                marker=dict(size=10, color="green", opacity=1.0),
                name="Optimized Point",
            )
        )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            aspectratio=dict(x=1, y=1, z=1.0),
        ),
        title=f"{axis_labels[2]} at various {axis_labels[0]} and {axis_labels[1]}",
        showlegend=True,
    )

    fig.show()


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


def add_missing_groundtruth(
    results: pd.DataFrame, groundtruth: pd.DataFrame
) -> pd.DataFrame:
    # List to hold new rows that need to be added
    new_rows = []

    # Group the dataframe by 'slick_id'
    for slick_id, g in groundtruth.groupby("slick_id"):
        # Check if there is no row with truth == True in the group
        group = results[results["slick_id"] == slick_id]
        if not group["truth"].any():
            # Prepare a new row with the specified fields
            new_row = {
                "truth": True,
                "coincidence_score": 0,
                "slick_id": slick_id,
                "rank": np.inf,  # use numpy.inf for infinity
            }
            new_rows.append(new_row)

    # If there are new rows, concatenate them with the original dataframe
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        results = pd.concat([results, new_rows_df], ignore_index=True)

    return results
