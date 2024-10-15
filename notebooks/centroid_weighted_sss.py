"""
Slick Plus Infrastructure Confidence Score Calculator

Processes GeoJSON files to compute confidence scores for infrastructure points based on their proximity to polygon centroid.
Features include projection handling, centroid selection, efficient scoring algorithms, and optional data visualization.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import MultiPolygon, Polygon


def generate_random_non_convex_polygon(
    num_vertices=10, outer_radius=10, inner_radius=2, angle_jitter=1, radius_jitter=5
):
    """
    Generates a random non-convex polygon by creating vertices with randomized angles and radii.

    Parameters:
    - num_vertices (int): Number of vertices for the polygon.
    - outer_radius (float): Maximum radius for any vertex.
    - inner_radius (float): Minimum radius for any vertex.
    - angle_jitter (float): Maximum jitter to apply to each angle (in radians).
    - radius_jitter (float): Maximum jitter to apply to each radius.

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
    radii = np.random.uniform(
        inner_radius, outer_radius, size=num_vertices
    ) + np.random.uniform(-radius_jitter, radius_jitter, size=num_vertices)

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
            num_vertices, outer_radius, inner_radius, angle_jitter, radius_jitter
        )

    return polygon


def select_extreme_points(polygon, N):
    """
    Selects N extremity points from the polygon that are furthest from the centroid
    and previously selected points.

    Parameters:
    - polygon (Polygon): A Shapely Polygon object.
    - N (int): Number of extremity points to select.

    Returns:
    - centroid_point (np.ndarray): Coordinates of the centroid (x, y).
    - selected_points (np.ndarray): Array of selected extremity points.
    """
    centroid = polygon.centroid
    centroid_point = np.array([centroid.x, centroid.y])
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
        if _ == 0:
            reference_points.pop(0)

    return centroid_point, np.array(selected_points)


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
    infra_x = np.random.uniform(minx - 5, maxx + 5, num_infra_points)
    infra_y = np.random.uniform(miny - 5, maxy + 5, num_infra_points)
    infra_points = np.column_stack((infra_x, infra_y))  # Shape: (num_infra_points, 2)

    # Query infrastructure points within distance D from any polygon boundary
    # This returns a list of lists; each sublist contains indices of boundary points within D of the infra point
    # To optimize, we can use query_ball_point in a vectorized way
    # However, for very large data, it's more efficient to process in batches
    batch_size = 10000  # Adjust based on memory and performance
    confidence_scores = np.zeros(num_infra_points)

    # Precompute extremity points and weights for all polygons
    extremity_points_list = []
    weights_list = []
    for polygon in polygons:
        centroid, extremity_points = select_extreme_points(polygon, N)
        # Calculate moment of inertia weights normalized by I_max to ensure max weight=1
        distances_sq = np.sum((extremity_points - centroid) ** 2, axis=1)
        I_max = distances_sq.max()
        if I_max == 0:
            weights = np.ones(N)  # Avoid division by zero if all distances are zero
        else:
            weights = distances_sq / I_max  # Normalize by I_max so that max weight=1
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
    # To handle large data efficiently, process in batches
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

            # For each valid infrastructure point, compute C = max(w_i * e^{-k*d_i})
            diffs = (
                all_extremity_points - valid_infra[:, np.newaxis, :]
            )  # Shape: (valid, N_total, 2)
            dists = np.linalg.norm(diffs, axis=2)  # Shape: (valid, N_total)
            C = np.exp(-k * dists) * all_weights  # Shape: (valid, N_total)
            C_max = C.max(axis=1)
            confidence_scores[start_idx + valid_indices] = (
                C_max  # Assign to the correct indices
            )

        # Optional: Print progress
        if (start_idx // batch_size) % 10 == 0:
            print(f"Processed {end_idx} / {num_infra_points} infrastructure points...")

    end_time = time.time()
    print(f"Confidence computation completed in {end_time - start_time:.2f} seconds.")

    # Optionally, plot a small subset for verification
    if plot_sample and num_polygons > 0 and num_infra_points > 0:
        sample_size = min(5000, num_infra_points)
        plt.figure(figsize=(10, 10))
        for poly in polygons[:5]:
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
        plt.colorbar(label="Confidence")
        plt.title("Infrastructure Points with Confidence Scores (Sample)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    return confidence_scores


# Set parameters for high efficiency
N = 2
num_polygons = 1  # Number of polygons
num_infra_points = 5000  # Number of infrastructure points
k = 2  # Decay constant
D = 10  # Maximum distance to consider
confidence_scores = process(
    N=N,
    num_polygons=num_polygons,
    num_infra_points=num_infra_points,
    k=k,
    D=D,
    plot_sample=True,
)

# %%
