"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import math

import geopandas as gpd
import movingpandas as mpd
import shapely.geometry
import shapely.ops


def compute_distance_score(
    traj: mpd.Trajectory,
    curves: gpd.GeoDataFrame,
    crs_meters: str,
    ais_ref_dist: float,
):
    """
    Alternative to compute_frechet_score.
    For every point on the slick curve, find the closest point on the trajectory and
    compute the average minimum distance.
    """
    # Only use the longest curve
    curve = curves.to_crs(crs_meters).iloc[0]["geometry"]

    # Get the trajectory coordinates as points in descending time order
    traj_gdf = (
        traj.to_point_gdf()
        .sort_values(by="timestamp", ascending=False)
        .set_crs("4326")
        .to_crs(crs_meters)
    )

    # Create a LineString from trajectory points
    traj_line = shapely.geometry.LineString(traj_gdf.geometry)

    # Get the first and last points of the slick curve
    first_point = shapely.geometry.Point(curve.coords[0])
    last_point = shapely.geometry.Point(curve.coords[-1])

    # Compute distances from these points to the start of the trajectory
    first_dist = first_point.distance(shapely.geometry.Point(traj_line.coords[0]))
    last_dist = last_point.distance(shapely.geometry.Point(traj_line.coords[0]))

    # Reverse curve orientation if necessary
    if last_dist < first_dist:
        curve = shapely.geometry.LineString(list(curve.coords)[::-1])

    # For every point on the curve, find the closest point on the trajectory
    min_distances = []
    for curve_point in curve.coords:
        curve_pt = shapely.geometry.Point(curve_point)
        distances = [
            curve_pt.distance(shapely.geometry.Point(traj_point))
            for traj_point in traj_line.coords
        ]
        min_distance = min(distances)
        min_distances.append(min_distance)

    # Compute the median of the minimum distances
    median_min_distance = sorted(min_distances)[len(min_distances) // 2]  # Median

    # Compute the score
    score = math.exp(-median_min_distance / ais_ref_dist)

    return score


def compute_temporal_score(
    weighted_traj: gpd.GeoDataFrame, slick_gdf: gpd.GeoDataFrame
):
    """
    Compute the temporal score between a weighted AIS trajectory and an oil slick

    Args:
        weighted_traj (gpd.GeoDataFrame): weighted AIS trajectory, with convex hull geometries and associated weights
        slick (shapely.geometry.Polygon): oil slick polygon
    Returns:
        float: temporal score between weighted_traj and slick
    """
    # spatially join the weighted convex hulls to the slick geometry
    matches = gpd.sjoin(weighted_traj, slick_gdf, how="inner", predicate="intersects")

    # take the sum of the weights of the matched convex hulls
    # Sums to 1 if all hulls intersect the slick
    temporal_score = matches["weight"].sum() if not matches.empty else 0.0

    return temporal_score


def compute_overlap_score(
    buffered_traj: gpd.GeoDataFrame,
    slick_gdf: gpd.GeoDataFrame,
    crs_meters: str,
):
    """
    Compute the amount of overlap between a buffered AIS trajectory and an oil slick

    Args:
        buffered_traj (shapely.geometry.Polygon): buffered AIS trajectory created by a convex hull operation
        slick_gdf (gpd.GeoDataFrame): oil slick polygon
    Returns:
        float: overlap score between buffered_traj and slick
    """

    buffered_traj = buffered_traj.to_crs(crs_meters)
    slick_gdf = slick_gdf.to_crs(crs_meters)
    slick_area = slick_gdf.unary_union.area
    intersection = buffered_traj.overlay(slick_gdf, how="intersection")
    intersection_area = intersection.unary_union.area

    overlap_score = intersection_area / slick_area
    # XXX this strongly benefits smaller slicks
    return overlap_score


def compute_total_score(
    temporal_score: float,
    overlap_score: float,
    distance_score: float,
    w_temporal: float,
    w_overlap: float,
    w_distance: float,
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
    total_weight = w_temporal + w_overlap + w_distance
    w_temporal /= total_weight
    w_overlap /= total_weight
    w_distance /= total_weight

    # Compute weighted sum
    total_score = (
        w_temporal * temporal_score
        + w_overlap * overlap_score
        + w_distance * distance_score
    )

    return total_score
