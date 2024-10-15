"""
Slick Plus Infrastructure Confidence Score Calculator

Processes GeoJSON files to compute confidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""

# %%
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


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


def select_extreme_points(polygon, N, overall_centroid):
    """
    Selects N extremity points from the polygon that are furthest from the overall centroid
    and previously selected points.

    Parameters:
    - polygon (Polygon): A Shapely Polygon object.
    - N (int): Number of extremity points to select.
    - overall_centroid (np.ndarray): Coordinates of the overall centroid (x, y).

    Returns:
    - selected_points (np.ndarray): Array of selected extremity points.
    """
    centroid_point = overall_centroid
    exterior_coords = np.array(polygon.exterior.coords[:-1])  # Exclude closing point
    selected_points = []
    reference_points = [centroid_point]

    for _ in range(N):
        # Compute distances from all exterior points to all reference points
        diff = exterior_coords[:, np.newaxis, :] - reference_points  # Shape: (M, K, 2)
        dists = np.linalg.norm(diff, axis=2)  # Shape: (M, K)
        min_dists = dists.min(axis=1)  # Shape: (M,)

        # Select the point with the maximum of these minimum distances
        idx = np.argmax(min_dists)
        selected_point = exterior_coords[idx]
        selected_points.append(selected_point)
        reference_points.append(selected_point)

    return np.array(selected_points)


def read_and_prepare_geojson(geojson_file_path, closing_distance):
    """
    Reads the GeoJSON file, checks and sets the CRS, and applies buffering.

    Parameters:
    - geojson_file_path (str): Path to the GeoJSON file.

    Returns:
    - geo_df (GeoDataFrame): Processed GeoDataFrame with buffering applied.
    - original_crs (dict): The original Coordinate Reference System of the GeoDataFrame.
    """
    geo_df = gpd.read_file(geojson_file_path)

    if geo_df.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Please define it before proceeding.")

    original_crs = geo_df.crs

    # If the current CRS is geographic (degrees), project to a UTM zone
    if geo_df.crs.is_geographic:
        centroid = geo_df.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        utm_crs = f"+proj=utm +zone={utm_zone} +{'north' if hemisphere == 'north' else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        geo_df = geo_df.to_crs(utm_crs)
    else:
        # Ensure the CRS is in meters
        if geo_df.crs.axis_info[0].unit_name != "metre":
            raise ValueError(
                "Current CRS is not in meters. Please choose an appropriate projected CRS."
            )

    # Apply buffer in meters (e.g., 1000 meters)
    geo_df["geometry"] = geo_df["geometry"].buffer(closing_distance)
    geo_df["geometry"] = geo_df["geometry"].buffer(-closing_distance)

    # Reproject back to original CRS if it was geographic
    if original_crs.is_geographic:
        geo_df = geo_df.to_crs(original_crs)

    return geo_df, original_crs


def extract_polygons(combined_geometry):
    """
    Extracts individual polygons from a combined geometry.

    Parameters:
    - combined_geometry (GeometryCollection | MultiPolygon | Polygon): The combined geometry.

    Returns:
    - polygons (list of Polygon): List of extracted Polygon objects.
    """
    polygons = []
    if isinstance(combined_geometry, MultiPolygon):
        polygons = list(combined_geometry.geoms)
    elif isinstance(combined_geometry, Polygon):
        polygons = [combined_geometry]
    elif isinstance(combined_geometry, GeometryCollection):
        polygons = [
            geom for geom in combined_geometry.geoms if isinstance(geom, Polygon)
        ]
    else:
        raise ValueError(f"Unsupported geometry type: {combined_geometry.geom_type}")

    if not polygons:
        raise ValueError("No polygons found in the GeoJSON file.")

    return polygons


def generate_infrastructure_points(combined_geometry, num_points, expansion_factor=0.5):
    """
    Generates random infrastructure points within an expanded bounding box of the combined geometry.

    Parameters:
    - combined_geometry (GeometryCollection | MultiPolygon | Polygon): The combined geometry.
    - num_points (int): Number of infrastructure points to generate.
    - expansion_factor (float): Fraction to expand the bounding box.

    Returns:
    - infra_points (np.ndarray): Array of infrastructure points coordinates.
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
    infra_points = np.column_stack((infra_x, infra_y))  # Shape: (num_points, 2)
    return infra_points


def collect_extremity_points(polygons, N, overall_centroid, largest_polygon_area):
    """
    Collects extremity points and their scaled area fractions from all polygons.

    Parameters:
    - polygons (list of Polygon): List of Polygon objects.
    - N (int): Number of extremity points to select per polygon.
    - overall_centroid (np.ndarray): Coordinates of the overall centroid (x, y).
    - largest_polygon_area (float): Area of the largest polygon.

    Returns:
    - all_extremity_points (np.ndarray): Array of all extremity points.
    - all_area_fractions (np.ndarray): Array of corresponding area fractions.
    """
    extremity_points_list = []
    area_fractions_list = []

    MIN_AREA_THRESHOLD = 0.1 * largest_polygon_area  # Adjust based on data

    for polygon in polygons:
        if polygon.area < MIN_AREA_THRESHOLD:
            continue  # Skip tiny polygons

        extremity_points = select_extreme_points(polygon, N, overall_centroid)
        extremity_points_list.append(extremity_points)

        # Compute area fraction for this polygon
        area_fraction = polygon.area / largest_polygon_area
        scaled_area_fraction = np.sqrt(area_fraction)  # More sensitive to small areas
        area_fractions_list.extend([scaled_area_fraction] * N)

    if not extremity_points_list:
        raise ValueError("No extremity points collected from polygons.")

    all_extremity_points = np.vstack(
        extremity_points_list
    )  # Shape: (total_extremity_points, 2)
    all_area_fractions = np.array(
        area_fractions_list
    )  # Shape: (total_extremity_points,)

    return all_extremity_points, all_area_fractions


def compute_weights(all_extremity_points, overall_centroid, all_area_fractions):
    """
    Computes normalized weights based on distances from the centroid and area fractions.

    Parameters:
    - all_extremity_points (np.ndarray): Array of all extremity points.
    - overall_centroid (np.ndarray): Coordinates of the overall centroid (x, y).
    - all_area_fractions (np.ndarray): Array of area fractions.

    Returns:
    - all_weights (np.ndarray): Normalized weights for each extremity point.
    """
    distances_sq_all = np.sum((all_extremity_points - overall_centroid) ** 2, axis=1)
    scaled_weights = distances_sq_all * all_area_fractions

    I_max = scaled_weights.max()
    if I_max == 0:
        all_weights = np.ones_like(scaled_weights)
    else:
        all_weights = scaled_weights / I_max  # Normalize so that max weight=1

    return all_weights


def compute_confidence_scores(
    infra_points,
    extremity_tree,
    all_extremity_points,
    all_weights,
    k,
    D,
    batch_size=10000,
):
    """
    Computes confidence scores for infrastructure points based on proximity to extremity points.

    Parameters:
    - infra_points (np.ndarray): Array of infrastructure points coordinates.
    - extremity_tree (KDTree): KDTree built from extremity points.
    - all_extremity_points (np.ndarray): Array of extremity points coordinates.
    - all_weights (np.ndarray): Array of weights for extremity points.
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - D (float): Maximum distance to consider for proximity.
    - batch_size (int): Number of points to process per batch.

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points.
    """
    num_infra = infra_points.shape[0]
    confidence_scores = np.zeros(num_infra)

    for start_idx in range(0, num_infra, batch_size):
        end_idx = min(start_idx + batch_size, num_infra)
        batch_infra = infra_points[start_idx:end_idx]

        # Find extremity points within distance D
        extremity_indices = extremity_tree.query_ball_point(batch_infra, r=D)

        has_neighbors = np.array(
            [len(neighbors) > 0 for neighbors in extremity_indices]
        )
        valid_indices = np.where(has_neighbors)[0]

        if valid_indices.size > 0:
            valid_infra = batch_infra[valid_indices]
            C_max = np.zeros(valid_indices.shape[0])

            for i, neighbors in enumerate(extremity_indices[valid_indices]):
                neighbor_points = all_extremity_points[neighbors]
                neighbor_weights = all_weights[neighbors]
                dists = np.linalg.norm(neighbor_points - valid_infra[i], axis=1)
                C_i = neighbor_weights * np.exp(-k * dists)
                C_max[i] = C_i.max()

            C_max = np.clip(C_max, 0, 1)
            confidence_scores[start_idx + valid_indices] = C_max

    return confidence_scores


def plot_confidence(
    infra_points,
    confidence_scores,
    polygons,
    overall_centroid,
    sample_size=50000,
    id=None,
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
    sample_size = min(sample_size, infra_points.shape[0])
    plt.figure(figsize=(10, 10))

    # Plot the polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.plot(x, y, "r", linewidth=3.0)

    # Plot infrastructure points with confidence scores
    plt.scatter(
        infra_points[:sample_size, 0],
        infra_points[:sample_size, 1],
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
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


def process_geojson(
    geojson_file_path,
    N=3,
    num_infra_points=100000,
    k=0.05,
    D=50,
    plot_sample=False,
    id=None,
    closing_distance=5000,
):
    """
    Main function to process the GeoJSON file and compute confidence scores.

    Parameters:
    - geojson_file_path (str): Path to the GeoJSON file.
    - N (int): Number of extremity points to select per polygon.
    - num_infra_points (int): Number of infrastructure points to generate.
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - D (float): Maximum distance to consider for infrastructure points near polygons.
    - plot_sample (bool): Whether to plot a sample of the data.
    - id (int): Identifier for plotting purposes.

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points.
    """
    start_time = time.time()

    # Step 1: Read and prepare GeoJSON
    geo_df, original_crs = read_and_prepare_geojson(geojson_file_path, closing_distance)

    # Step 2: Combine all geometries
    combined_geometry = geo_df.unary_union

    # Step 3: Extract individual polygons
    polygons = extract_polygons(combined_geometry)

    # Step 4: Compute the area of the largest polygon
    largest_polygon_area = max(polygon.area for polygon in polygons)

    # Step 5: Compute the overall centroid
    overall_centroid = combined_geometry.centroid.coords[0]

    # Step 6: Generate random infrastructure points
    infra_points = generate_infrastructure_points(combined_geometry, num_infra_points)

    # Step 7: Collect extremity points and area fractions
    all_extremity_points, all_area_fractions = collect_extremity_points(
        polygons, N, overall_centroid, largest_polygon_area
    )

    # Step 8: Compute weights
    all_weights = compute_weights(
        all_extremity_points, overall_centroid, all_area_fractions
    )

    # Step 9: Build KD-Tree
    extremity_tree = KDTree(all_extremity_points)

    # Step 10: Compute confidence scores
    confidence_scores = compute_confidence_scores(
        infra_points, extremity_tree, all_extremity_points, all_weights, k, D
    )

    end_time = time.time()
    print(f"Slick {id} completed in {end_time - start_time:.2f} seconds.")

    # Step 11: Plotting (optional)
    if plot_sample and num_infra_points > 0:
        plot_confidence(
            infra_points,
            confidence_scores,
            polygons,
            overall_centroid,
            sample_size=50000,
            id=id,
        )

    return confidence_scores


# %%
# Usage example

# Parameters
id = 3032494
geojson_file_path = download_geojson(id)

N = 10  # Number of extremity points per polygon
num_infra_points = 50000  # Number of infrastructure points
k = 50  # Decay constant for the confidence function C = e^{-k * d}
D = 1  # Maximum distance to consider before confidence is 0
closing_distance = 500  # Closing distance in meters
plot_sample = True

confidence_scores = process_geojson(
    geojson_file_path, N, num_infra_points, k, D, plot_sample, id, closing_distance
)

# %%
