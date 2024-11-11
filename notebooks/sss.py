# %%
"""
Slick Plus Infrastructure Coincidence Score Calculator

Processes GeoJSON files to compute coincidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""

# %load_ext autoreload
# %autoreload 2

import datetime
import os
import sys
from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from geoalchemy2 import WKTElement

load_dotenv(".env")
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from cerulean_cloud.cloud_function_ais_analysis.utils.analyzer import (  # noqa: E402
    AISAnalyzer,
    InfrastructureAnalyzer,
)
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

    max_coincidence = (
        round(coincidence_scores.max(), 2) if len(coincidence_scores) else 0
    )

    # Set titles and labels
    plt.title(f"Slick ID {id}: Max Coincidence {max_coincidence}")
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
ids = [
    # [3479812, 'S1A_IW_GRDH_1SDV_20240424T051434_20240424T051459_053571_0680D2_0854'],
    # [3013300, 'S1A_IW_GRDH_1SDV_20240901T175307_20240901T175332_055475_06C482_3EA4'],
    # [3522449, 'S1A_IW_GRDH_1SDV_20241007T175309_20241007T175334_056000_06D942_BC27'],
    # [3343987, 'S1A_IW_GRDH_1SDV_20240609T053018_20240609T053047_054242_0698FD_4B01'],
    # [3070818, 'S1A_IW_GRDH_1SDV_20231125T174456_20231125T174521_051377_06332E_E619'],
    # [3173928, "S1A_IW_GRDH_1SDV_20230704T174453_20230704T174518_049277_05ECE7_7DE1"],
    # [3229370, 'S1A_IW_GRDH_1SDV_20231007T171227_20231007T171252_050662_061A9E_BC8B'],
    # [3105854, 'S1A_IW_GRDH_1SDV_20230806T221833_20230806T221858_049761_05FBD2_577C'],
    [3411218, "S1A_IW_GRDH_1SDV_20240226T221831_20240226T221856_052736_066190_8A37"],
]

accumulated_pairs = []
for slick_id, scene_id in ids:
    geojson_file_path = download_geojson(slick_id)
    slick_gdf = gpd.read_file(geojson_file_path, crs="epsg:4326")

    ia = InfrastructureAnalyzer(slick_gdf, scene_id)
    res = ia.compute_coincidence_scores()

    if not res.empty:
        top_row = res.loc[res["coincidence_score"].idxmax()]
        structure_id = top_row["structure_id"]
        accumulated_pairs.append({"slick_id": slick_id, "structure_id": structure_id})
    else:
        print(f"No associations found for slick_id: {slick_id}")

    plot_coincidence(ia.infra_gdf, ia.slick_gdf, ia.coincidence_scores, slick_id)
print(accumulated_pairs)

# %%
infra_gdf = generate_infrastructure_points(slick_gdf, 50000)
coincidence_scores = associate_infra_to_slick(infra_gdf, slick_gdf)
plot_coincidence(infra_gdf, slick_gdf, coincidence_scores, slick_id, False)

# %%

scene_id = "S1A_IW_GRDH_1SDV_20230711T160632_20230711T160657_049378_05F013_448A"
s1 = SimpleNamespace(
    id=69,
    scene_id=scene_id,
    absolute_orbit_number=49378,
    mode="IW",
    polarization="DV",
    scihub_ingestion_time=datetime.datetime(2023, 7, 11, 17, 0, 44, 705000),
    start_time=datetime.datetime(2023, 7, 11, 16, 6, 32),
    end_time=datetime.datetime(2023, 7, 11, 16, 6, 57),
    meta={"key": "value"},
    url="http://example.com",
    geometry=WKTElement(
        "POLYGON((27.25877432902152 33.772309925795675,27.663991468237075 33.83643596487585,27.945526383317603 33.8801453845104,28.368361808395143 33.94450422643601,28.650602342685367 33.98660415338177,29.07447962628422 34.04854096653544,29.357403549219118 34.08902106313116,29.640594459658455 34.128850435890584,29.934431272085337 34.16944366756764,29.78891655910936 34.89087313481798,29.63062722274537 35.672109011899686,29.33131173198989 35.6318916336964,29.04285374913959 35.59238617078463,28.75468097143034 35.55219240037903,28.32296568450625 35.49061480402974,28.035524056468923 35.448707639173,27.604924536739723 35.38456808689098,27.318239230682664 35.34095844599587,26.905637045158333 35.27691057625879,27.0900410473316 34.49469629172755,27.25877432902152 33.772309925795675))"
    ),
)

aa = AISAnalyzer(slick_gdf, scene_id, s1)
res = aa.compute_coincidence_scores()

# %%
