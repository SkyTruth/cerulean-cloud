"""
Slick Plus Infrastructure Confidence Score Calculator

Processes GeoJSON files to compute confidence scores for infrastructure points based on their proximity to polygon extremity points.
Features include projection handling, extremity point selection, efficient scoring algorithms, and optional data visualization.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import MultiPolygon, Polygon


def generate_random_non_convex_polygon(
    num_vertices=20, outer_radius=10, inner_radius=2, angle_jitter=1
):
    """
    Generates a random non-convex polygon by creating vertices with randomized angles and radii.

    Parameters:
    - num_vertices (int): Number of vertices for the polygon.
    - outer_radius (float): Maximum radius for any vertex.
    - inner_radius (float): Minimum radius for any vertex.
    - angle_jitter (float): Maximum jitter to apply to each angle (in radians).

    Returns:
    - polygon (Polygon): A valid Shapely Polygon object.
    """
    # Generate base angles evenly spaced around the circle
    base_angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

    # Apply random jitter to each angle to make the polygon non-regular
    jittered_angles = base_angles + np.random.uniform(
        -angle_jitter, angle_jitter, size=num_vertices
    )
    jittered_angles = np.mod(
        jittered_angles, 2 * np.pi
    )  # Ensure angles are within [0, 2π)

    # Sort the angles to maintain the correct order around the polygon
    jittered_angles = np.sort(jittered_angles)

    # Assign radii uniformly between inner_radius and outer_radius for all vertices
    # Apply jitter to radii to add randomness
    radii = np.random.uniform(inner_radius, outer_radius, size=num_vertices)

    # Ensure all radii are within the specified bounds
    radii = np.clip(radii, a_min=inner_radius, a_max=outer_radius)

    # Convert polar coordinates (radii and angles) to Cartesian coordinates (x, y)
    x = radii * np.cos(jittered_angles)
    y = radii * np.sin(jittered_angles)
    points = np.column_stack((x, y))

    # Create the polygon and ensure its validity
    polygon = Polygon(points)

    # If the polygon is invalid (e.g., self-intersecting), regenerate it recursively
    if not polygon.is_valid or not polygon.is_simple or polygon.area == 0:
        return generate_random_non_convex_polygon(
            num_vertices, outer_radius, inner_radius, angle_jitter
        )

    return polygon


def select_extreme_points(polygon, N):
    """
    Selects N extremity points from the polygon based on exterior angles closest to 180 degrees.

    Parameters:
    - polygon (Polygon): A Shapely Polygon object.
    - N (int): Number of extremity points to select.

    Returns:
    - centroid_point (np.ndarray): Coordinates of the centroid (x, y).
    - selected_points (np.ndarray): Array of selected extremity points.
    - selected_weights (np.ndarray): Array of weights for the selected extremity points.
    """
    centroid = polygon.centroid
    centroid_point = np.array([centroid.x, centroid.y])
    vertices = np.array(polygon.exterior.coords[:-1])  # Exclude closing point

    # Compute previous and next vertices
    v_prev = np.roll(vertices, 1, axis=0)
    v_next = np.roll(vertices, -1, axis=0)

    # Compute vectors
    vec1 = vertices - v_prev  # From previous vertex to current vertex
    vec2 = v_next - vertices  # From current vertex to next vertex

    # Compute cross product and dot product
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    dot = vec1[:, 0] * vec2[:, 0] + vec1[:, 1] * vec2[:, 1]

    # Compute the angle between vec1 and vec2
    angles = np.arctan2(cross, dot)  # Angles between -pi and pi

    # Compute interior angles
    interior_angles = np.mod(angles, 2 * np.pi)  # Angles between 0 and 2*pi

    # Compute exterior angles
    exterior_angles = 2 * np.pi - interior_angles  # Exterior angles between 0 and 2*pi

    # Compute weights:
    # - Assign weights only to vertices with exterior_angle >= pi
    # - Weight peaks at pi and decreases linearly to 0 at 2*pi
    weights = np.where(
        exterior_angles >= np.pi, 1 - (exterior_angles - np.pi) / np.pi, 0
    )
    weights = np.clip(weights, 0, 1)  # Ensure weights are between 0 and 1

    # Select N vertices with highest weights
    sorted_indices = np.argsort(weights)[::-1]  # Descending order
    top_indices = sorted_indices[:N]

    # In case fewer than N vertices have weights > 0
    valid_top_indices = top_indices[weights[top_indices] > 0]

    selected_points = vertices[valid_top_indices]
    selected_weights = weights[valid_top_indices]

    # If fewer than N points are selected, pad with zeros
    if len(selected_points) < N:
        pad_size = N - len(selected_points)
        pad_points = np.zeros((pad_size, 2))
        pad_weights = np.zeros(pad_size)
        selected_points = np.vstack([selected_points, pad_points])
        selected_weights = np.concatenate([selected_weights, pad_weights])

    return centroid_point, selected_points, selected_weights


def compute_confidence_max(infra_points, extremity_points, weights, k):
    """
    Computes the confidence values for multiple infrastructure points given extremity points and weights.
    Instead of summing the weighted confidences, it takes the maximum confidence value.

    Parameters:
    - infra_points (np.ndarray): Array of infrastructure points, shape (M, 2).
    - extremity_points (np.ndarray): Array of extremity points, shape (N_total, 2).
    - weights (np.ndarray): Array of weights for each extremity point, shape (N_total,).
    - k (float): Decay constant.

    Returns:
    - confidence (np.ndarray): Array of confidence scores, shape (M,).
    """
    # Compute distances between all infrastructure points and extremity points
    diff = infra_points[:, np.newaxis, :] - extremity_points  # Shape: (M, N, 2)
    dists = np.linalg.norm(diff, axis=2)  # Shape: (M, N)

    # Compute C = e^{-k * d} for all combinations
    C = np.exp(-k * dists)  # Shape: (M, N)

    # Multiply by weights
    C_weighted = C * weights  # Shape: (M, N)

    # Take the maximum confidence value across extremity points
    confidence = C_weighted.max(axis=1)  # Shape: (M,)

    # Ensure confidence values are between 0 and 1
    confidence = np.clip(confidence, 0, 1)

    return confidence


def process(
    N=3, num_polygons=1000, num_infra_points=100000, k=0.05, D=50, plot_sample=False
):
    """
    Main function to generate multiple polygons and infrastructure points,
    compute confidence values efficiently using the max function.

    Parameters:
    - N (int): Number of extremity points to select per polygon.
    - num_polygons (int): Number of polygons to generate.
    - num_infra_points (int): Number of infrastructure points to generate.
    - k (float): Decay constant for the confidence function C = e^{-k * d}.
    - D (float): Maximum distance to consider for infrastructure points near polygons.
    - plot_sample (bool): Whether to plot a sample of the data.

    Returns:
    - confidence_scores (np.ndarray): Array of confidence scores for infrastructure points.
    """
    start_time = time.time()

    # Generate multiple random polygons
    polygons = []
    for _ in range(num_polygons):
        poly = generate_random_non_convex_polygon()
        polygons.append(poly)
    multipolygon = MultiPolygon(polygons)

    # Generate random infrastructure points within a bounding box expanded by D
    minx, miny, maxx, maxy = multipolygon.bounds
    infra_x = np.random.uniform(minx - 10, maxx + 10, num_infra_points)
    infra_y = np.random.uniform(miny - 10, maxy + 10, num_infra_points)
    infra_points = np.column_stack((infra_x, infra_y))  # Shape: (num_infra_points, 2)

    # Precompute extremity points and weights for all polygons
    extremity_points_list = []
    weights_list = []
    for polygon in polygons:
        centroid, extremity_points, weights = select_extreme_points(polygon, N)
        extremity_points_list.append(extremity_points)
        weights_list.append(weights)

    # Concatenate extremity points and weights for all polygons
    all_extremity_points = np.vstack(
        extremity_points_list
    )  # Shape: (num_polygons * N, 2)
    all_weights = np.concatenate(weights_list)  # Shape: (num_polygons * N,)

    # Build a KD-Tree for extremity points only
    extremity_tree = KDTree(all_extremity_points)

    # Compute confidence values for infrastructure points near the polygons
    batch_size = 10000  # Adjust based on memory and performance
    confidence_scores = np.zeros(num_infra_points)

    # Process infrastructure points in batches
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

            # Extract the relevant extremity points and weights
            relevant_ext_indices = [
                indices for indices in extremity_indices[valid_indices]
            ]
            # Flatten the list and get unique indices to avoid redundant computations
            flattened_indices = np.unique(np.concatenate(relevant_ext_indices))
            relevant_ext_points = all_extremity_points[flattened_indices]
            relevant_weights = all_weights[flattened_indices]

            # Compute distances
            diffs = (
                relevant_ext_points - valid_infra[:, np.newaxis, :]
            )  # Shape: (valid, N_relevant, 2)
            dists = np.linalg.norm(diffs, axis=2)  # Shape: (valid, N_relevant)

            # Compute C = max(w_i * e^{-k*d_i} for all relevant extremity points
            C = np.exp(-k * dists) * relevant_weights  # Shape: (valid, N_relevant)
            C_max = C.max(axis=1)
            confidence_scores[start_idx + valid_indices] = (
                C_max  # Assign to the correct indices
            )

        # Optional: Print progress
        if ((start_idx // batch_size) % 10 == 0) or (end_idx == num_infra_points):
            print(f"Processed {end_idx} / {num_infra_points} infrastructure points...")

    end_time = time.time()
    print(f"Confidence computation completed in {end_time - start_time:.2f} seconds.")

    # Optionally, plot a small subset for verification
    if plot_sample and num_polygons > 0 and num_infra_points > 0:
        sample_size = min(5000, num_infra_points)
        plt.figure(figsize=(10, 10))
        for poly in polygons[:5000]:
            x, y = poly.exterior.xy
            plt.plot(x, y, "r", linewidth=3.0)
        sc = plt.scatter(
            infra_points[:sample_size, 0],
            infra_points[:sample_size, 1],
            c=confidence_scores[:sample_size],
            cmap="Blues",
            s=10,
            vmax=1,
            vmin=0,
        )
        plt.colorbar(sc, label="Confidence")
        plt.title("Infrastructure Points with Confidence Scores (Sample)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return confidence_scores


# Set parameters for high efficiency
N = 10
num_polygons = 1  # Number of polygons
num_infra_points = 5000  # Number of infrastructure points
k = 1  # Decay constant
D = 1000  # Maximum distance to consider
confidence_scores = process(
    N=N,
    num_polygons=num_polygons,
    num_infra_points=num_infra_points,
    k=k,
    D=D,
    plot_sample=True,
)

# %%
