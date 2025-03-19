"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import datetime
import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point


def nearest_index(point: tuple, collection) -> int:
    """
    Find the index of the element in 'collection' that is closest to the given 'point'.
    The collection can be a list of shapely Points or coordinate tuples.
    """
    if isinstance(point, tuple):
        point = Point(point)

    def to_point(x):
        return x if isinstance(x, Point) else Point(x)

    return min(
        range(len(collection)), key=lambda i: point.distance(to_point(collection[i]))
    )


def compute_proximity_score(
    traj_gdf: gpd.GeoDataFrame,
    longest_centerline: MultiLineString,
    spread_rate: float,
    image_timestamp: datetime.datetime,
) -> float:
    """
    Compute the Frechet distance-based score between an AIS trajectory and an oil slick centerline.

    Score definition:
        score = exp( - (Frechet distance / ais_ref_dist) )
    """
    # Get the trajectory points and timestamps
    traj_points = list(traj_gdf["geometry"])
    traj_timestamps = list(traj_gdf.index)

    # Create centerline endpoints
    cl_A = Point(longest_centerline.coords[0])
    cl_B = Point(longest_centerline.coords[-1])

    # Find nearest trajectory point indices for each endpoint
    traj_idx_A = nearest_index(cl_A, traj_points)
    traj_idx_B = nearest_index(cl_B, traj_points)

    # Create tuples for each endpoint: (timestamp, distance, centerline_point)
    ends = [
        (traj_timestamps[traj_idx_A], traj_points[traj_idx_A].distance(cl_A), cl_A),
        (traj_timestamps[traj_idx_B], traj_points[traj_idx_B].distance(cl_B), cl_B),
    ]

    # Sort the pairs by timestamp to determine head and tail
    (t_tail, d_tail, cl_tail), (t_head, d_head, cl_head) = sorted(
        ends, key=lambda x: x[0]
    )

    # closest centerline point to the vessel at image_timestamp
    # d_0 = traj_gdf.loc[image_timestamp].geometry.distance(cl_head)
    idx = np.abs(traj_gdf.index - image_timestamp).argmin()
    d_0 = traj_gdf.iloc[idx].geometry.distance(cl_head)

    delta_tail = (image_timestamp - t_tail).total_seconds() / 3600
    if delta_tail <= 0:  # The tail is in front of the vessel
        P_t = 0  # No grace distance
    else:
        P_t = math.exp(-d_tail / (spread_rate * delta_tail))

    delta_head = (image_timestamp - t_head).total_seconds() / 3600
    if delta_head <= 0:  # The head is in front of the vessel
        P_h = math.exp(-d_0 / 500)  # 500m grace distance
    else:
        P_h = math.exp(-d_head / (spread_rate * delta_head))

    return P_t * P_h


def compute_parity_score(
    traj_gdf: gpd.GeoDataFrame,
    longest_centerline: MultiLineString,
    sensitivity_parity: float,
) -> float:
    """
    Compute the parity score, which measures the similarity between the length of an oil slick centerline
    and the length of the projected AIS trajectory.

    Args:
        traj_gdf (gpd.GeoDataFrame): AIS trajectory
        centerlines (gpd.GeoDataFrame): Oil slick centerlines
    Returns:
        float: Parity score between 0 and 1
    """
    traj_points = list(traj_gdf["geometry"])

    # Identify trajectory points closest to the centerline endpoints.
    idx_first = nearest_index(longest_centerline.coords[0], traj_points)
    idx_last = nearest_index(longest_centerline.coords[-1], traj_points)

    start_idx, end_idx = min(idx_first, idx_last), max(idx_first, idx_last)

    if traj_points[start_idx] == traj_points[end_idx]:
        return 0.0

    # Extract the relevant substring of the trajectory.
    traj_substring = LineString(traj_points[start_idx : end_idx + 1])

    return math.exp(
        -(math.log(longest_centerline.length / traj_substring.length) ** 2)
        * sensitivity_parity
    )


def compute_temporal_score(
    traj_gdf: gpd.GeoDataFrame,
    longest_centerline: MultiLineString,
    image_timestamp: datetime.datetime,
    ais_ref_time_over: float,
    ais_ref_time_under: float,
) -> float:
    """
    Compute the temporal score between an AIS trajectory and an oil slick centerline.

    Let x be the time difference (in seconds) between the more recent trajectory point (closest to a centerline endpoint)
    and the reference timestamp t_ref. The score is computed as:

        if (newer_timestamp > t_ref):
            score = exp( - ((x - a) / A)^2 )
        else:
            score = exp( - ((x - a) / B)^2 )
    """
    traj_points = list(traj_gdf["geometry"])
    traj_timestamps = list(traj_gdf.index)

    # Identify trajectory points closest to the centerline endpoints.
    idx_first = nearest_index(longest_centerline.coords[0], traj_points)
    idx_last = nearest_index(longest_centerline.coords[-1], traj_points)

    # Retrieve corresponding timestamps.
    head_timestamp = max(
        traj_timestamps[idx_first],
        traj_timestamps[idx_last],
    )

    # Calculate the time difference (in seconds) from the reference time.
    time_delta = (image_timestamp - head_timestamp).total_seconds()

    if time_delta < 0:
        # time_delta is negative, so the head_timestamp is in front of the image_timestamp.
        # This means the trajectory is less likely to be associated with the oil slick.
        score = math.exp(-((time_delta / ais_ref_time_over) ** 2))
    else:
        # time_delta is positive, so the head_timestamp is behind the image_timestamp.
        # This means the trajectory is more likely to be associated with the oil slick.
        score = math.exp(-((time_delta / ais_ref_time_under) ** 2))
    return score


def vessel_compute_total_score(
    temporal_score: float,
    proximity_score: float,
    parity_score: float,
    w_temporal: float,
    w_proximity: float,
    w_parity: float,
):
    """
    Compute the weighted total score.

    Args:
        temporal_score (float): temporal score between a weighted AIS trajectory and an oil slick
        overlap_score (float): overlap score between a buffered AIS trajectory and an oil slick
        distance_score (float): distance score between an AIS trajectory and an oil slick centerline
        w_temporal (float): Weight for the temporal score.
        w_overlap (float): Weight for the overlap score.
        w_distance (float): Weight for the distance score.

    Returns:
        float: Weighted total score between 0 and 1.
    """
    # Normalize weights
    total_weight = w_temporal + w_proximity + w_parity
    w_temporal /= total_weight
    w_proximity /= total_weight
    w_parity /= total_weight

    # Compute weighted sum
    total_score = (
        w_temporal * temporal_score
        + w_proximity * proximity_score
        + w_parity * parity_score
    )

    return total_score
