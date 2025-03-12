import os
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from geoalchemy2 import WKTElement
from tqdm.auto import tqdm


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


def download_geojson(id, download_path=os.getenv("ASA_DOWNLOAD_PATH")):
    """
    Downloads a GeoJSON file from the specified URL if it hasn't been downloaded already.

    Parameters:
    - id (int): The unique identifier for the GeoJSON item.
    - download_path (str): The directory path where the GeoJSON will be saved.

    Returns:
    - geojson_file_path (str): The file path to the downloaded GeoJSON.
    """
    url = f"https://api.cerulean.skytruth.org/collections/public.slick/items?id={id}&f=geojson"
    geojson_file_path = os.path.join(download_path, f"{id}.geojson")

    if not os.path.exists(geojson_file_path):
        print(f"Downloading GeoJSON file for ID {id}...")
        os.system(f'curl "{url}" -o "{geojson_file_path}"')
        print(f"Downloaded GeoJSON to {geojson_file_path}")
    else:
        print(f"GeoJSON file already exists at {geojson_file_path}. Skipping download.")

    return geojson_file_path


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
    return results.groupby("slick_id", group_keys=False).apply(
        lambda grp: labelling_method(grp, grp["slick_id"].iloc[0], groundtruth_gdf)
    )


def calculate_metrics(results):
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
        false_df = df[df["truth"]]

        # Compute top 3 source rate (as a percentage)
        if len(truth_df) > 0:
            top3_rate = ((truth_df["rank"] <= 3).sum() / len(truth_df)) * 100
            top1_rate = ((truth_df["rank"] == 1).sum() / len(truth_df)) * 100
            avg_true = truth_df["coincidence_score"].mean()
        else:
            top3_rate = np.nan
            top1_rate = np.nan
            avg_true = np.nan

        # Compute average false coincidence
        avg_false = (
            false_df["coincidence_score"].mean() if len(false_df) > 0 else np.nan
        )

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


def process_groundtruth_on_analyzer(
    analyzer_class, groundtruth_gdf, points_gdf=None, analyzer_params=None
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

    Returns:
        GeoDataFrame containing accumulated results.
    """
    results_gdf = None  # Empty GeoDataFrame to accumulate results

    for ex_id in tqdm(range(len(groundtruth_gdf)), leave=False):
        groundtruth_source = groundtruth_gdf.iloc[[ex_id]].reset_index()

        slick_id = groundtruth_source["slick_id"].values[0]
        s1_scene_id = groundtruth_source["scene_id"].values[0]
        geojson_file_path = download_geojson(slick_id)
        slick_gdf = gpd.read_file(geojson_file_path)
        s1_scene = get_s1_scene(s1_scene_id)

        # Prepare parameters for the analyzer
        analyzer_kwargs = analyzer_params or {}

        # Assign points_gdf to the correct parameter name based on analyzer type
        if points_gdf is not None:
            if analyzer_class.__name__ == "DarkAnalyzer":
                analyzer_kwargs["dark_vessels_gdf"] = points_gdf.reset_index()
            elif analyzer_class.__name__ == "InfrastructureAnalyzer":
                analyzer_kwargs["infra_gdf"] = points_gdf.reset_index()

        # Instantiate the analyzer with parameters
        analyzer = analyzer_class(s1_scene, **analyzer_kwargs)

        # Compute coincidence scores
        res = analyzer.compute_coincidence_scores(slick_gdf)
        res["slick_id"] = slick_id  # Track which slick_id it corresponds to

        # Accumulate results using pd.concat
        if results_gdf is None:
            results_gdf = res.copy()
        else:
            results_gdf = pd.concat([results_gdf, res], ignore_index=True)

    return results_gdf


def plot_metrics(metrics_df):
    """
    Plot metrics from three methods side by side using two subplots:
      - Left subplot: top source rates (scaled to 0-1) and average coincidence scores.
      - Right subplot: the ratio metric (true to false average coincidence).

    Each bar is annotated with its rounded value above it.

    Parameters:
        metrics_df: DataFrame with method names as rows and metric names as columns.
                    Expected columns:
                      'top_3_source_rate (%)', 'top_1_source_rate (%)',
                      'average_true_coincidence', 'average_false_coincidence', 'true_false_ratio'
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
    right_metrics = ["true_false_ratio"]

    # Prepare data for the left subplot.
    df_left = df_plot.melt(
        id_vars="Method", value_vars=left_metrics, var_name="Metric", value_name="Value"
    )
    # Convert top source rate metrics from percentages to 0-1 scale.
    top_source_metrics = ["top_3_source_rate (%)", "top_1_source_rate (%)"]
    df_left.loc[df_left["Metric"].isin(top_source_metrics), "Value"] /= 100.0

    # Prepare data for the right subplot.
    df_right = df_plot.melt(
        id_vars="Method",
        value_vars=right_metrics,
        var_name="Metric",
        value_name="Value",
    )

    # Create two subplots aligned horizontally.
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6), sharey=False)

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
                fontsize=8,
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
    ax_left.legend(handles=handles, labels=labels, title="Method")
    ax_right.get_legend().remove()

    fig.suptitle("Evaluation Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
