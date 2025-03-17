"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import datetime
import math

import geopandas as gpd
from shapely import MultiLineString, frechet_distance
from shapely.geometry import LineString, Point


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
    traj_points = list(traj_gdf["geometry"])
    traj_timestamps = list(traj_gdf.index)

    # Identify trajectory points closest to the centerline endpoints.
    idx_first = nearest_index(longest_centerline.coords[0], traj_points)
    idx_last = nearest_index(longest_centerline.coords[-1], traj_points)
    start_idx, end_idx = min(idx_first, idx_last), max(idx_first, idx_last)
    if start_idx == end_idx:
        # The centerline is all the way to one end of the trajectory, so double the end point
        traj_substring = LineString([traj_points[start_idx], traj_points[end_idx]])
    else:
        traj_substring = LineString(traj_points[start_idx : end_idx + 1])

    if start_idx == idx_last:
        # The two linestrings are anti-parallel so we need to change the orientation by reversing the slick centerline
        longest_centerline = LineString(list(longest_centerline.coords)[::-1])

    # Retrieve corresponding timestamps.
    tail_timestamp = min(
        traj_timestamps[idx_first],
        traj_timestamps[idx_last],
    )

    # Calculate the time difference (in hours) from the reference time.
    time_delta = (image_timestamp - tail_timestamp).total_seconds() / 3600
    if time_delta < 0:
        # time_delta is negative, so the tail_timestamp is in front of the image_timestamp.
        # This means the trajectory is extremely unlikely to be associated with the oil slick.
        # XXX Can this logic be brought up a level?
        return 0.0
    else:
        # time_delta is positive, so the tail_timestamp is behind the image_timestamp.
        # This means the trajectory is more likely to be associated with the oil slick.
        ref_dist = spread_rate * time_delta

    # Compute Frechet distance and transform it into a score.
    dist = frechet_distance(traj_substring, longest_centerline)

    return math.exp(-dist / ref_dist)


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
