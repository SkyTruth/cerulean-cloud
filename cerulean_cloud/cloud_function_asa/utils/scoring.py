"""
Utilities for calculating scoring metrics between AIS trajectories and oil slick detections
"""

import math

import geopandas as gpd
import movingpandas as mpd
import shapely.geometry
import shapely.ops
from shapely import frechet_distance


def compute_distance_score(
    traj: mpd.Trajectory,
    centerlines: gpd.GeoDataFrame,
    crs_meters: str,
    ais_ref_dist: float,
):
    """
    Compute the frechet distance between an AIS trajectory and an oil slick centerline

    Args:
        traj (mpd.Trajectory): AIS trajectory
        centerlines (gpd.GeoDataFrame): oil slick centerlines

    Returns:
        float: frechet distance between traj and centerline
    """
    # Only use the longest centerline
    centerlines = centerlines.sort_values("length", ascending=False)
    longest_centerline = centerlines.to_crs(crs_meters).iloc[0]["geometry"]

    # get the trajectory coordinates as points in descending time order from collect
    traj_gdf = (
        traj.to_point_gdf()
        .sort_values(by="timestamp", ascending=False)
        .set_crs("4326")
        .to_crs(crs_meters)
    )

    # take the points and put them in a linestring
    traj_line = shapely.geometry.LineString(traj_gdf.geometry)

    # get the first and last points of the slick centerline
    first_point = shapely.geometry.Point(longest_centerline.coords[0])
    last_point = shapely.geometry.Point(longest_centerline.coords[-1])

    # compute the distance from these points to the start of the trajectory
    first_dist = first_point.distance(shapely.geometry.Point(traj_line.coords[0]))
    last_dist = last_point.distance(shapely.geometry.Point(traj_line.coords[0]))

    if last_dist < first_dist:
        # change input orientation by reversing the slick centerline
        longest_centerline = shapely.geometry.LineString(
            list(longest_centerline.coords)[::-1]
        )

    # for every point in the centerline, find the closest trajectory point and store it off
    traj_points = list()
    for centerline_point in longest_centerline.coords:
        # compute the distance between this point and every point in the trajectory
        these_distances = list()
        for traj_point in traj_line.coords:
            dist = shapely.geometry.Point(centerline_point).distance(
                shapely.geometry.Point(traj_point)
            )
            these_distances.append(dist)

        closest_distance = min(these_distances)
        closest_idx = these_distances.index(closest_distance)
        traj_points.append(shapely.geometry.Point(traj_line.coords[closest_idx]))

    # compute the frechet distance between the sampled trajectory and the slick centerline
    traj_line_clip = shapely.geometry.LineString(traj_points)
    dist = frechet_distance(traj_line_clip, longest_centerline)

    frechet_score = math.exp(-dist / ais_ref_dist)

    return frechet_score


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


def vessel_compute_total_score(
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
        distance_score (float): distance score between an AIS trajectory and an oil slick centerline
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
