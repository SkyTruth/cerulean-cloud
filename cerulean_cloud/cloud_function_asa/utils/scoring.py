"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import datetime
import math

import geopandas as gpd
import shapely.geometry
import shapely.ops
from shapely import frechet_distance


def nearest_index(point: tuple, collection) -> int:
    """
    Find the index of the element in 'collection' that is closest to the given 'point'.
    The collection can be a list of shapely Points or coordinate tuples.
    """
    if isinstance(point, tuple):
        point = shapely.geometry.Point(point)

    def to_point(x):
        return x if isinstance(x, shapely.geometry.Point) else shapely.geometry.Point(x)

    return min(
        range(len(collection)), key=lambda i: point.distance(to_point(collection[i]))
    )


def compute_proximity_score(
    traj_gdf: gpd.GeoDataFrame,
    curve: shapely.geometry.LineString,
    spread_rate: float,
    image_timestamp: datetime.datetime,
) -> float:
    """
    Compute the Frechet distance-based score between an AIS trajectory and an oil slick curve.

    Score definition:
        score = exp( - (Frechet distance / ais_ref_dist) )
    """
    traj_points = list(traj_gdf["geometry"])

    # Adjust curve orientation to ensure the starting endpoint is closest to the trajectory start.
    start_pt = traj_points[0]
    if shapely.geometry.Point(curve.coords[-1]).distance(
        start_pt
    ) < shapely.geometry.Point(curve.coords[0]).distance(start_pt):
        curve = shapely.geometry.LineString(list(curve.coords)[::-1])

    # Sample trajectory points corresponding to each point on the curve.
    sampled_traj_points = []
    for cp in curve.coords:
        cp_point = shapely.geometry.Point(cp)
        nearest_pt = min(traj_points, key=lambda pt: cp_point.distance(pt))
        sampled_traj_points.append(nearest_pt)

    traj_line_sampled = shapely.geometry.LineString(sampled_traj_points)

    # Compute Frechet distance and transform it into a score.
    dist = frechet_distance(traj_line_sampled, curve)

    traj_timestamps = list(traj_gdf["geometry"].index)

    # Identify trajectory points closest to the curve endpoints.
    idx_first = nearest_index(curve.coords[0], traj_points)
    idx_last = nearest_index(curve.coords[-1], traj_points)

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

    return math.exp(-dist / ref_dist)


def compute_parity_score(
    traj_gdf: gpd.GeoDataFrame,
    curve: shapely.geometry.LineString,
) -> float:
    """
    Compute the parity score, which measures the similarity between the length of an oil slick curve
    and the length of the projected AIS trajectory.

    Args:
        traj_gdf (gpd.GeoDataFrame): AIS trajectory
        curve (shapely.geometry.LineString): Oil slick curve

    Returns:
        float: Parity score between 0 and 1
    """
    traj_points = list(traj_gdf["geometry"])
    # Determine the indices of the trajectory points closest to the curve's endpoints.
    idx_first = nearest_index(curve.coords[0], traj_points)
    idx_last = nearest_index(curve.coords[-1], traj_points)
    start_idx, end_idx = min(idx_first, idx_last), max(idx_first, idx_last)

    if traj_points[start_idx] == traj_points[end_idx]:
        return 0.0

    # Extract the relevant substring of the trajectory.
    traj_substring = shapely.geometry.LineString(traj_points[start_idx : end_idx + 1])

    return math.exp(-(math.log(curve.length / traj_substring.length) ** 2))


def compute_temporal_score(
    traj_gdf: gpd.GeoDataFrame,
    curve: shapely.geometry.LineString,
    image_timestamp: datetime.datetime,
    ais_ref_time_over: float,
    ais_ref_time_under: float,
) -> float:
    """
    Compute the temporal score between an AIS trajectory and an oil slick curve.

    Let x be the time difference (in seconds) between the more recent trajectory point (closest to a curve endpoint)
    and the reference timestamp t_ref. The score is computed as:

        if (newer_timestamp > t_ref):
            score = exp( - ((x - a) / A)^2 )
        else:
            score = exp( - ((x - a) / B)^2 )
    """
    traj_points = list(traj_gdf["geometry"])

    traj_timestamps = list(traj_gdf["geometry"].index)

    # Identify trajectory points closest to the curve endpoints.
    idx_first = nearest_index(curve.coords[0], traj_points)
    idx_last = nearest_index(curve.coords[-1], traj_points)

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


def compute_total_score(
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
        distance_score (float): distance score between an AIS trajectory and an oil slick curve
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
