"""
Slick Plus Infrastructure Confidence Score Calculator

Processes GeoJSON files to compute confidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""

# %%
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

id = 3264306
url = f"https://api.cerulean.skytruth.org/collections/public.slick_plus/items?id={id}&f=geojson"
geojson_file_path = f"/Users/jonathanraphael/Downloads/{id}.geojson"
# !curl "{url}" -o {geojson_file_path}  # download the geojson file


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


def process_geojson(
    geojson_file_path,
    N=3,
    num_infra_points=100000,
    k=0.05,
    D=50,
    plot_sample=False,
):
    """
    Main function to read polygons from a geojson file,
    generate infrastructure points,
    compute confidence values efficiently using the sum function.

    Parameters:
    - geojson_file_path (str): Path to the geojson file containing the MultiPolygon.
    - N (int): Number of extremity points to select per polygon.
    - num_infra_points (int): Number of infrastructure points to generate.
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - D (float): Maximum distance to consider for infrastructure points near polygons.
    - plot_sample (bool): Whether to plot a sample of the data.

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points.
    """
    start_time = time.time()

    # Read the geojson file
    geo_df = gpd.read_file(geojson_file_path)

    # Check the current CRS and project to a CRS with meters as units
    if geo_df.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Please define it before proceeding.")

    # If the current CRS is geographic (degrees), project to a UTM zone
    if geo_df.crs.is_geographic:
        # Calculate the centroid to determine the appropriate UTM zone
        centroid = geo_df.unary_union.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        utm_crs = f"+proj=utm +zone={utm_zone} +{'north' if hemisphere == 'north' else 'south'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

        # Project to the UTM CRS
        geo_df = geo_df.to_crs(utm_crs)

    else:
        # If already projected, ensure it's in meters
        if geo_df.crs.axis_info[0].unit_name != "metre":
            raise ValueError(
                "Current CRS is not in meters. Please choose an appropriate projected CRS."
            )

    # Apply buffer in meters (e.g., 500 meters)
    buffer_distance = 1000  # Adjust the buffer distance as needed
    geo_df["geometry"] = geo_df["geometry"].buffer(buffer_distance)
    geo_df["geometry"] = geo_df["geometry"].buffer(-buffer_distance)

    # Optionally, reproject back to the original CRS (e.g., WGS84)
    geo_df = geo_df.to_crs("EPSG:4326")

    # Combine all geometries into a single geometry (could be MultiPolygon or other types)
    combined_geometry = geo_df.unary_union

    # Extract polygons from the combined geometry
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

    # **Compute the area of the largest polygon**
    largest_polygon_area = max(polygon.area for polygon in polygons)

    # Compute the overall centroid of the MultiPolygon
    overall_centroid = combined_geometry.centroid
    overall_centroid_point = np.array([overall_centroid.x, overall_centroid.y])

    # Generate random infrastructure points within a bounding box expanded by D
    minx, miny, maxx, maxy = combined_geometry.bounds
    infra_x = np.random.uniform(minx - 0.05, maxx + 0.05, num_infra_points)
    infra_y = np.random.uniform(miny - 0.05, maxy + 0.05, num_infra_points)
    infra_points = np.column_stack((infra_x, infra_y))  # Shape: (num_infra_points, 2)

    # **First Pass: Collect all extremity points from all polygons**
    extremity_points_list = []
    area_fractions_list = []

    # Set a minimum area threshold (adjust based on your data)
    MIN_AREA_THRESHOLD = 0.1 * largest_polygon_area

    for polygon in polygons:
        if polygon.area < MIN_AREA_THRESHOLD:
            # Skip tiny polygons
            continue

        extremity_points = select_extreme_points(polygon, N, overall_centroid_point)
        extremity_points_list.append(extremity_points)

        # Compute area fraction for this polygon
        area_fraction = polygon.area / largest_polygon_area
        scaled_area_fraction = np.sqrt(area_fraction)  # More sensitive to small areas
        area_fractions_list.extend([scaled_area_fraction] * N)

    if not extremity_points_list:
        raise ValueError("No extremity points collected from polygons.")

    # Concatenate all extremity points for global processing
    all_extremity_points = np.vstack(
        extremity_points_list
    )  # Shape: (total_extremity_points, 2)
    all_area_fractions = np.array(
        area_fractions_list
    )  # Shape: (total_extremity_points,)

    # **Apply Area Fraction Scaling Before Normalization**
    distances_sq_all = np.sum(
        (all_extremity_points - overall_centroid_point) ** 2, axis=1
    )
    scaled_weights = distances_sq_all * all_area_fractions

    # **Compute I_max Globally After Scaling**
    I_max = scaled_weights.max()
    if I_max == 0:
        all_weights = np.ones_like(scaled_weights)
    else:
        all_weights = (
            scaled_weights / I_max
        )  # Normalize after scaling so that max weight=1

    # **Second Pass: Assign Weights**
    # Since area fractions are already applied, we don't need to scale again here
    # Therefore, all_weights already incorporate area fractions and normalization

    # Build a KD-Tree for extremity points only
    extremity_tree = KDTree(all_extremity_points)

    # Compute confidence values for infrastructure points near the extremity points
    confidence_scores = np.zeros(num_infra_points)

    batch_size = 10000  # Adjust based on memory and performance

    for start_idx in range(0, num_infra_points, batch_size):
        end_idx = min(start_idx + batch_size, num_infra_points)
        batch_infra = infra_points[start_idx:end_idx]

        # Find extremity points within D for each infrastructure point in the batch
        extremity_indices = extremity_tree.query_ball_point(batch_infra, r=D)

        # Prepare a mask for infrastructure points that have at least one extremity point within D
        has_neighbors = np.array(
            [len(neighbors) > 0 for neighbors in extremity_indices]
        )
        valid_indices = np.where(has_neighbors)[0]

        if valid_indices.size > 0:
            # Extract the infrastructure points that have neighbors
            valid_infra = batch_infra[valid_indices]

            # Initialize an array to hold the confidence sums
            C_max = np.zeros(valid_indices.shape[0])

            # For each valid infrastructure point, compute the sum of weighted confidences
            for i, neighbors in enumerate(extremity_indices[valid_indices]):
                neighbor_points = all_extremity_points[neighbors]
                neighbor_weights = all_weights[neighbors]
                dists = np.linalg.norm(neighbor_points - valid_infra[i], axis=1)
                C_i = neighbor_weights * np.exp(-k * dists)
                C_max[i] = C_i.max()

            # Ensure that C_max does not exceed 1
            C_max = np.clip(C_max, 0, 1)

            # Assign to the correct indices
            confidence_scores[start_idx + valid_indices] = C_max

        # Optional: Print progress every 10 batches
        if (start_idx // batch_size) % 10 == 0:
            print(f"Processed {end_idx} / {num_infra_points} infrastructure points...")

    end_time = time.time()
    print(f"Confidence computation completed in {end_time - start_time:.2f} seconds.")

    # Optionally, plot a small subset for verification
    if plot_sample and num_infra_points > 0:
        sample_size = min(50000, num_infra_points)
        plt.figure(figsize=(10, 10))
        # Plot the polygons
        for poly in polygons:
            x, y = poly.exterior.xy
            plt.plot(x, y, "r", linewidth=3.0)
        plt.scatter(
            infra_points[:sample_size, 0],
            infra_points[:sample_size, 1],
            c=confidence_scores[:sample_size],
            cmap="Blues",
            s=10,
            vmin=0,
            vmax=1,
        )
        plt.plot(
            overall_centroid_point[0],
            overall_centroid_point[1],
            "k+",
            markersize=10,
        )
        plt.colorbar(label="Confidence")
        plt.title(f"Slick ID {id}: Max Confidence {round(max(confidence_scores), 2)}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return confidence_scores


# Usage example
# Set parameters
N = 10
num_infra_points = 50000  # Number of infrastructure points
k = 50  # Decay constant
D = 1  # Maximum distance to consider

confidence_scores = process_geojson(
    geojson_file_path=geojson_file_path,
    N=N,
    num_infra_points=num_infra_points,
    k=k,
    D=D,
    plot_sample=True,
)

# %%
