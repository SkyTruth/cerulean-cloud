"""
Utilities and helper functions for the oil slick Leaflet map
"""

import json
from typing import List

import centerline.geometry
import geopandas as gpd
import movingpandas as mpd
import numpy as np
import scipy.interpolate
import scipy.spatial.distance
import shapely.geometry
import shapely.ops

from .constants import SPREAD_RATE
from .scoring import (
    compute_frechet_distance,
    compute_overlap_score,
    compute_temporal_score,
    compute_total_score,
)


def calculate_moment_of_inertia(geometry, point):
    """Given an multi/polygon and a point, calculate moment of inertia"""
    if isinstance(geometry, shapely.geometry.Polygon):
        return geometry.area * point.distance(geometry.centroid) ** 2
    elif isinstance(geometry, shapely.geometry.MultiPolygon):
        return sum(
            [poly.area * point.distance(poly.centroid) ** 2 for poly in geometry.geoms]
        )
    else:
        raise ValueError("Input must be a Polygon or MultiPolygon")


def calculate_maximum_moi(geometry):
    """Given an multi/polygon estimate an upper limit to reasonable moment of interia values"""
    # Calculate the moment of inertia for each corner of the minimum rotated bounding box and update the maximum
    return max(
        [
            calculate_moment_of_inertia(geometry, shapely.geometry.Point(corner))
            for corner in geometry.minimum_rotated_rectangle.exterior.coords
        ]
    )


def associate_infra_to_slick(
    infra_gdf: gpd.GeoDataFrame, slick_gdf: gpd.GeoDataFrame, crs_meters
):
    """Associate a given slick to the global Infrastructure database"""
    # Define the columns for the associations GeoDataFrame
    columns = [
        "st_name",
        "geometry",
        "coincidence_score",
        "type",
        "ext_id",
        "geojson_fc",
    ]

    # Create an empty list to store associations
    entries = []

    # Calculate the maximum moment of inertia for the slick geometry
    max_moi = calculate_maximum_moi(slick_gdf.to_crs(crs_meters).iloc[0]["geometry"])

    # Load infrastructure data from a file and create a GeoDataFrame
    # Create a buffered version of the 'slick' GeoDataFrame
    buffered = slick_gdf.copy()
    buffered["geometry"] = (
        slick_gdf.to_crs(crs_meters).buffer(SPREAD_RATE).to_crs("4326")
    )  # Buffer the slick geometries

    # Perform a spatial join to find infrastructure points that intersect with the buffered slick geometries
    nearby_infra = gpd.sjoin(infra_gdf, buffered, how="inner", predicate="intersects")

    if not nearby_infra.empty:
        # Calculate a moment of inertia score for the nearby infrastructure, normalized against the maximum moment of inertia
        nearby_infra["moi_score"] = (
            nearby_infra.to_crs(crs_meters)["geometry"].apply(
                lambda point: calculate_moment_of_inertia(
                    slick_gdf.to_crs(crs_meters).iloc[0]["geometry"], point
                )
            )
            / max_moi
            * 3  # Adjust the score [0,1] to a range in line with other scores [0,~4]
        )

        # Iterate over the nearby infrastructure to populate the associations GeoDataFrame
        for _, row in nearby_infra.iterrows():
            gdf = gpd.GeoDataFrame(
                {
                    "timestamp": ["20231103T00:00:00"],
                    "geometry": [row["geometry"]],
                },  # XXX FRAGILE timestamp needs to be made programmatically generated with file date
                crs="4326",
            )
            entry = {
                "st_name": row["st_name"],
                "geometry": row["geometry"],
                "coincidence_score": row["moi_score"],
                "type": 2,  # As defined in SourceType table
                "ext_id": row["detect_id"],
                "geojson_fc": {
                    "type": "FeatureCollection",
                    "features": json.loads(gdf.to_json())["features"],
                },
            }
            entries.append(entry)

    # Return the populated associations GeoDataFrame
    associations = gpd.GeoDataFrame(entries, columns=columns, crs="4326")

    return associations


def associate_ais_to_slick(
    ais: mpd.TrajectoryCollection,
    buffered: gpd.GeoDataFrame,
    weighted: List[gpd.GeoDataFrame],
    slick_gdf: gpd.GeoDataFrame,
    curves: gpd.GeoDataFrame,
    crs_meters: str,
):
    """
    Measure association by computing multiple metrics between AIS trajectories and slicks

    Inputs:
        ais: TrajectoryCollection of AIS trajectories
        buffered: GeoDataFrame of buffered AIS trajectories
        weighted: list of GeoDataFrames weighted AIS trajectories
        slick: GeoDataFrame of slick detections
        curves: GeoDataFrame of slick curves
    Returns:
        GeoDataFrame of slick associations
    """
    # only consider trajectories that intersect slick detections
    ais_filt = list()
    weighted_filt = list()
    buffered_filt = list()
    for idx, t in enumerate(ais):
        w = weighted[idx]
        b = buffered.iloc[idx]

        # spatially join the weighted trajectory to the slick
        b_gdf = gpd.GeoDataFrame(index=[0], geometry=[b.geometry], crs="4326")
        matches = gpd.sjoin(b_gdf, slick_gdf, how="inner", predicate="intersects")
        if matches.empty:
            continue
        else:
            ais_filt.append(t)
            weighted_filt.append(w)
            buffered_filt.append(b_gdf)

    columns = [
        "st_name",
        "geometry",
        "coincidence_score",
        "type",
        "ext_name",
        "ext_shiptype",
        "flag",
        "geojson_fc",
    ]

    entries = []
    # Skip the loop if weighted_filt is empty
    if weighted_filt:
        # create trajectory collection from filtered trajectories
        ais_filt = mpd.TrajectoryCollection(ais_filt)

        # iterate over filtered trajectories
        for t, w, b in zip(ais_filt, weighted_filt, buffered_filt):
            # compute temporal score
            temporal_score = compute_temporal_score(w, slick_gdf)

            # compute overlap score
            overlap_score = compute_overlap_score(b, slick_gdf, crs_meters)

            # compute frechet distance between trajectory and slick curve
            frechet_dist = compute_frechet_distance(t, curves, crs_meters)

            # compute total score from these three metrics
            coincidence_score = compute_total_score(
                temporal_score, overlap_score, frechet_dist
            )

            print(
                f"st_name {t.id}: coincidence_score ({coincidence_score}) = overlap_score ({overlap_score}) * temporal_score ({temporal_score}) + 2000/frechet_dist ({frechet_dist})"
            )

            entry = {
                "st_name": t.id,
                "geometry": shapely.geometry.LineString(
                    [p.coords[0] for p in t.df["geometry"]]
                ),
                "coincidence_score": coincidence_score,
                "type": 1,  # As defined in SourceType table
                "ext_name": t.ext_name,
                "ext_shiptype": t.ext_shiptype,
                "flag": t.flag,
                "geojson_fc": t.geojson_fc,
            }
            entries.append(entry)
    associations = gpd.GeoDataFrame(entries, columns=columns, crs="4326")

    return associations


def slick_to_curves(
    slick_gdf: gpd.GeoDataFrame,
    crs_meters: str,
    buf_size: int = 2000,
    interp_dist: int = 200,
    smoothing_factor: float = 1e9,
):
    """
    From a set of oil slick detections, estimate curves that go through the detections
    This process transforms a set of slick detections into LineStrings for each detection

    Inputs:
        slick: GeoDataFrame of slick detections
        buf_size: buffer size for cleaning up slick detections
        interp_dist: interpolation distance for centerline
        smoothing_factor: smoothing factor for smoothing centerline
    Returns:
        GeoDataFrame of slick curves
    """
    # clean up the slick detections by dilation followed by erosion
    # this process can merge some polygons but not others, depending on proximity
    slick_clean = slick_gdf.copy()
    slick_clean["geometry"] = (
        slick_clean.to_crs(crs_meters).buffer(buf_size).buffer(-buf_size)
    )

    # split slicks into individual polygons
    slick_clean = slick_clean.explode(ignore_index=True, index_parts=False)

    # find a centerline through detections
    slick_curves = list()
    for idx, row in slick_clean.iterrows():
        # create centerline -> MultiLineString
        try:
            cl = centerline.geometry.Centerline(
                row.geometry, interpolation_distance=interp_dist
            )
        except (
            Exception
        ) as e:  # noqa # unclear what exception was originally thrown here.
            # sometimes the voronoi polygonization fails
            # in this case, just fit a a simple line from the start to the end
            exterior_coords = row.geometry.exterior.coords
            start_point = exterior_coords[0]
            end_point = exterior_coords[-1]
            curve = shapely.geometry.LineString([start_point, end_point])
            slick_curves.append(curve)
            print(
                f"XXX ~WARNING~ Blanket try/except caught error but continued on anyway: {e}"
            )
            continue

        # grab coordinates from centerline
        x = list()
        y = list()
        if type(cl.geometry) == shapely.geometry.MultiLineString:
            # iterate through each linestring
            for geom in cl.geometry.geoms:
                x.extend(geom.coords.xy[0])
                y.extend(geom.coords.xy[1])
        else:
            x.extend(cl.geometry.coords.xy[0])
            y.extend(cl.geometry.coords.xy[1])

        # sort coordinates in both X and Y directions
        coords = [(xc, yc) for xc, yc in zip(x, y)]
        coords_sort_x = sorted(coords, key=lambda c: c[0])
        coords_sort_y = sorted(coords, key=lambda c: c[1])

        # remove coordinate duplicates, preserving sorted order
        coords_seen_x = set()
        coords_unique_x = list()
        for c in coords_sort_x:
            if c not in coords_seen_x:
                coords_unique_x.append(c)
                coords_seen_x.add(c)

        coords_seen_y = set()
        coords_unique_y = list()
        for c in coords_sort_y:
            if c not in coords_seen_y:
                coords_unique_y.append(c)
                coords_seen_y.add(c)

        # grab x and y coordinates for spline fit
        x_fit_sort_x = [c[0] for c in coords_unique_x]
        x_fit_sort_y = [c[0] for c in coords_unique_y]
        y_fit_sort_x = [c[1] for c in coords_unique_x]
        y_fit_sort_y = [c[1] for c in coords_unique_y]

        # Check if there are enough points for spline fitting
        min_points_required = 4  # for cubic spline, k=3, need at least 4 points
        if len(coords_unique_x) >= min_points_required:
            # fit a B-spline to the centerline
            tck_sort_x, fp_sort_x, _, _ = scipy.interpolate.splrep(
                x_fit_sort_x, y_fit_sort_x, k=3, s=smoothing_factor, full_output=True
            )
            tck_sort_y, fp_sort_y, _, _ = scipy.interpolate.splrep(
                y_fit_sort_y, x_fit_sort_y, k=3, s=smoothing_factor, full_output=True
            )

            # choose the spline that has the lowest fit error
            if fp_sort_x <= fp_sort_y:
                tck = tck_sort_x
                x_fit = x_fit_sort_x
                y_fit = y_fit_sort_x

                num_points = max(round((x_fit[-1] - x_fit[0]) / 100), 5)
                x_new = np.linspace(x_fit[0], x_fit[-1], 10)
                y_new = scipy.interpolate.BSpline(*tck)(x_new)
            else:
                tck = tck_sort_y
                x_fit = x_fit_sort_y
                y_fit = y_fit_sort_y

                num_points = max(round((y_fit[-1] - y_fit[0]) / 100), 5)
                y_new = np.linspace(y_fit[0], y_fit[-1], num_points)
                x_new = scipy.interpolate.BSpline(*tck)(y_new)

            # store as LineString
            curve = shapely.geometry.LineString(zip(x_new, y_new))
        else:
            curve = shapely.geometry.LineString(
                [coords_unique_x[0], coords_unique_x[-1]]
            )
        slick_curves.append(curve)

    slick_curves_gdf = gpd.GeoDataFrame(geometry=slick_curves, crs=crs_meters)
    slick_curves_gdf["length"] = slick_curves_gdf.geometry.length
    slick_curves_gdf = slick_curves_gdf.sort_values("length", ascending=False).to_crs(
        "4326"
    )

    return slick_clean, slick_curves_gdf
