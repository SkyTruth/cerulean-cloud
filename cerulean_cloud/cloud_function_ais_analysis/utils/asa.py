"""
Automatic Source Analysis utils
"""

import time
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def estimate_utm_crs(geometry):
    """
    Estimates an appropriate UTM CRS based on the centroid of the geometry.
    """
    return CRS.from_dict(
        {
            "proj": "utm",
            "zone": int((geometry.centroid.x + 180) / 6) + 1,
            "south": geometry.centroid.y < 0,
        }
    )


def apply_closing_buffer(geo_df, closing_buffer):
    """
    Applies a closing buffer to geometries in the GeoDataFrame.

    Parameters:
    - geo_df (GeoDataFrame): GeoDataFrame with geometries.
    - closing_buffer (float): Distance for buffering in meters.

    Returns:
    - geo_df (GeoDataFrame): GeoDataFrame with updated geometries.
    """
    # Apply closing buffer
    geo_df["geometry"] = (
        geo_df["geometry"].buffer(closing_buffer).buffer(-closing_buffer)
    )
    return geo_df


def extract_polygons(geometry):
    """
    Extracts individual polygons from a geometry.

    Returns:
    - polygons (list of Polygon): List of extracted Polygon objects.
    """
    return (
        [geom for geom in geometry.geoms if isinstance(geom, Polygon)]
        if isinstance(geometry, (MultiPolygon, GeometryCollection))
        else [geometry]
    )


def select_extreme_points(
    polygon: Polygon, N: int, reference_points: List[np.ndarray]
) -> np.ndarray:
    """
    Selects N extremity points from the polygon based on their distance from reference points.

    Parameters:
    - polygon (Polygon): A Shapely Polygon object.
    - N (int): Number of extremity points to select.
    - reference_points (List[np.ndarray]): List of reference points (x, y).

    Returns:
    - np.ndarray: Array of selected extremity points with shape (N, 2).
    """
    exterior_coords = np.array(polygon.exterior.coords[:-1])  # Exclude closing point
    selected_points = []

    for _ in range(N):
        # Compute distances from all exterior points to reference points
        diff = exterior_coords[:, np.newaxis, :] - reference_points  # Shape: (M, K, 2)
        dists = np.linalg.norm(diff, axis=2)  # Shape: (M, K)
        min_dists = dists.min(axis=1)  # Shape: (M,)

        # Select the point with the maximum of these minimum distances
        idx = np.argmax(min_dists)
        selected_point = exterior_coords[idx]
        selected_points.append(selected_point)
        reference_points.append(
            selected_point
        )  # Update reference points for next iteration

    return np.array(selected_points)


def collect_extremity_points(
    polygons: List[Polygon],
    N: int,
    overall_centroid: np.ndarray,
    largest_polygon_area: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects extremity points and their scaled area fractions from all polygons.

    Parameters:
    - polygons (List[Polygon]): List of Polygon objects.
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
            continue  # Skip small polygons

        # Select N extremity points for the current polygon
        selected_points = select_extreme_points(polygon, N, [overall_centroid])
        extremity_points_list.append(selected_points)

        # Compute scaled area fraction for weighting
        area_fraction = polygon.area / largest_polygon_area
        scaled_area_fraction = np.sqrt(area_fraction)  # More sensitive to small areas
        area_fractions_list.extend([scaled_area_fraction] * N)

    if not extremity_points_list:
        raise ValueError("No extremity points collected from polygons.")

    all_extremity_points = np.vstack(extremity_points_list)
    all_area_fractions = np.array(area_fractions_list)

    return all_extremity_points, all_area_fractions


def compute_weights(all_extremity_points, overall_centroid, all_area_fractions):
    """
    Computes normalized weights based on distances from the centroid and area fractions.

    Parameters:
    - all_extremity_points (np.ndarray): Array of extremity points.
    - overall_centroid (np.ndarray): Coordinates of the overall centroid.
    - all_area_fractions (np.ndarray): Area fractions for weighting.

    Returns:
    - all_weights (np.ndarray): Normalized weights for each extremity point.
    """
    distances_sq = np.sum((all_extremity_points - overall_centroid) ** 2, axis=1)
    scaled_weights = distances_sq * all_area_fractions

    max_weight = scaled_weights.max()

    return (
        scaled_weights / max_weight if max_weight != 0 else np.ones_like(scaled_weights)
    )


def compute_confidence_scores(
    infra_gdf: gpd.GeoDataFrame,
    extremity_tree: cKDTree,
    all_extremity_points: np.ndarray,
    all_weights: np.ndarray,
    k: float,
    radius_of_interest: float,
) -> np.ndarray:
    """
    Computes confidence scores for infrastructure points based on proximity to extremity points.

    Parameters:
    - infra_gdf (gpd.GeoDataFrame): GeoDataFrame of infrastructure points.
    - extremity_tree (KDTree): KDTree built from extremity points.
    - all_extremity_points (np.ndarray): Array of extremity points coordinates (shape: [n_extremities, 2]).
    - all_weights (np.ndarray): Array of weights for extremity points (shape: [n_extremities]).
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - radius_of_interest (float): Maximum distance to consider for proximity.

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points (shape: [n_infra]).
    """
    infra_coords = np.array([(geom.x, geom.y) for geom in infra_gdf.geometry])
    extremity_indices = extremity_tree.query_ball_point(
        infra_coords, r=radius_of_interest
    )
    confidence_scores = np.zeros(len(infra_coords))

    for i, neighbors in enumerate(extremity_indices):
        if neighbors:
            neighbor_points = all_extremity_points[neighbors]
            neighbor_weights = all_weights[neighbors]
            dists = np.linalg.norm(neighbor_points - infra_coords[i], axis=1)
            C_i = neighbor_weights * np.exp(-k * dists)
            confidence_scores[i] = np.clip(C_i.max(), 0, 1)

    return confidence_scores


def associate_infra_to_slick(
    infra_gdf: gpd.GeoDataFrame,  # GeoDataFrame of infrastructure points
    slick_gdf: gpd.GeoDataFrame,  # GeoDataFrame of slick points
    k: float = 0.0005,  # Decay constant for the confidence function C = e^{-k * d}
    N: int = 10,  # Number of extremity points per polygon
    closing_buffer: int = 500,  # Closing distance in meters
    radius_of_interest: int = 5000,  # Maximum distance to consider (in meters)
):
    """
    Main function to compute confidence scores.

    Parameters:
    - infra_gdf (GeoDataFrame): GeoDataFrame of infrastructure points.
    - slick_gdf (GeoDataFrame): GeoDataFrame of slick geometries.
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - N (int): Number of extremity points to select per polygon.
    - closing_buffer (int): Closing buffer distance in meters.
    - radius_of_interest (int): Maximum distance to consider for proximity (in meters).

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points.
    """
    start_time = time.time()

    crs_meters = estimate_utm_crs(slick_gdf.unary_union)
    slick_gdf = slick_gdf.to_crs(crs_meters)
    infra_gdf = infra_gdf.to_crs(crs_meters)

    # Initialize confidence_scores with zeros
    confidence_scores = np.zeros(len(infra_gdf))

    # Apply closing buffer and project slick_gdf
    slick_gdf = apply_closing_buffer(slick_gdf, closing_buffer)

    # Combine geometries and extract polygons
    combined_geometry = slick_gdf.unary_union
    polygons = extract_polygons(combined_geometry)

    slick_buffered = combined_geometry.buffer(radius_of_interest)

    # Spatial join or intersection to filter infra_gdf
    infra_within_radius = infra_gdf[infra_gdf.geometry.within(slick_buffered)]

    if infra_within_radius.empty:
        print(
            "No infrastructure points within the radius of interest. Returning zero confidence scores."
        )
        # Return an array of zeros with the same length as the original infra_gdf
        return confidence_scores

    # Keep track of original indices to map back to the full infra_gdf
    infra_indices = infra_within_radius.index

    # Compute largest area and overall centroid
    largest_polygon_area = max(polygon.area for polygon in polygons)
    overall_centroid = np.array(combined_geometry.centroid.coords[0])

    # Collect extremity points and compute weights
    all_extremity_points, all_area_fractions = collect_extremity_points(
        polygons, N, overall_centroid, largest_polygon_area
    )
    all_weights = compute_weights(
        all_extremity_points, overall_centroid, all_area_fractions
    )

    # Build KD-Tree and compute confidence scores
    extremity_tree = cKDTree(all_extremity_points)
    confidence_filtered = compute_confidence_scores(
        infra_within_radius,
        extremity_tree,
        all_extremity_points,
        all_weights,
        k,
        radius_of_interest,
    )

    confidence_scores[infra_indices] = confidence_filtered

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

    return confidence_scores
