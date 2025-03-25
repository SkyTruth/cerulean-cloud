"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import datetime
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point


def compute_proximity_score(
    traj_gdf: gpd.GeoDataFrame,
    spread_rate: float,
    t_image: datetime.datetime,
    slick_to_traj_mapping: tuple[
        pd.Timestamp, Point, float, pd.Timestamp, Point, float
    ],
) -> float:
    """
    Compute a distance-based score between an AIS trajectory and an oil slick centerline.

    Score definition:
        score = exp( - (distance / ais_ref_dist) )
    """
    t_tail, cl_tail, d_tail, t_head, cl_head, d_head = slick_to_traj_mapping

    # Get an additional distance of interest: the centerline head and the track at t=0
    near_0_idx = np.abs(traj_gdf.index - t_image).argmin()
    d_0 = traj_gdf.iloc[near_0_idx].geometry.distance(cl_head)

    delta_tail = (t_image - t_tail).total_seconds() / 3600
    if delta_tail <= 0:  # The tail is in front of the vessel
        P_t = 0  # No grace distance
    else:
        P_t = math.exp(-d_tail / (spread_rate * delta_tail))

    delta_head = (t_image - t_head).total_seconds() / 3600
    if delta_head <= 0:  # The head is in front of the vessel
        P_h = math.exp(-d_0 / 500)  # 500m grace distance
    else:
        P_h = math.exp(-d_head / (spread_rate * delta_head))
    return np.sqrt(P_t * P_h)


def compute_parity_score(
    traj_gdf: gpd.GeoDataFrame,
    longest_centerline: MultiLineString,
    sensitivity_parity: float,
    slick_to_traj_mapping: tuple[
        pd.Timestamp, Point, float, pd.Timestamp, Point, float
    ],
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
    t_tail, cl_tail, d_tail, t_head, cl_head, d_head = slick_to_traj_mapping

    if t_tail == t_head:
        # The head and tail map to the same point, so the parity score is 0.
        return 0.0

    # Extract the relevant substring of the trajectory.
    traj_substring = LineString(traj_gdf.sort_index().loc[t_tail:t_head]["geometry"])

    return math.exp(
        -(math.log(longest_centerline.length / traj_substring.length) ** 2)
        * sensitivity_parity
    )


def compute_temporal_score(
    t_image: datetime.datetime,
    ais_ref_time_over: float,
    ais_ref_time_under: float,
    slick_to_traj_mapping: tuple[
        pd.Timestamp, Point, float, pd.Timestamp, Point, float
    ],
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

    t_tail, cl_tail, d_tail, t_head, cl_head, d_head = slick_to_traj_mapping

    # Calculate the time difference (in seconds) from the image timestamp to the closest endpoint.
    t_closer_end = t_head if d_head < d_tail else t_tail
    time_delta = (t_image - t_closer_end).total_seconds()

    if time_delta < 0:
        # time_delta is negative, so the slick head is in front of the t_image.
        # This means the trajectory is less likely to be associated with the oil slick.
        score = math.exp(-((time_delta / ais_ref_time_over) ** 2))
    else:
        # time_delta is positive, so the slick head is behind the t_image.
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
