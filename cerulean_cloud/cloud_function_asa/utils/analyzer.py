"""
Unified Source Analysis Module
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import geopandas as gpd
import mapbox_vector_tile
import morecantile
import numpy as np
import pandas as pd
import pandas_gbq
import requests
from geoalchemy2.shape import to_shape
from google.oauth2.service_account import Credentials
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
)
from shapely.ops import unary_union

from . import constants as c
from .scoring import (
    compute_parity_score,
    compute_proximity_score,
    compute_temporal_score,
    vessel_compute_total_score,
)


class SourceAnalyzer:
    """
    Base class for source analysis.

    Inputs:
        s1_scene (Scene): A Scene object containing the scene metadata and geometry.

    Attributes:
        slick_gdf (GeoDataFrame): GeoDataFrame containing the slick geometries.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the SourceAnalyzer.
        """
        self.s1_scene = s1_scene
        geom_shape = to_shape(self.s1_scene.geometry)  # Convert to Shapely geometry
        centroid = geom_shape.centroid
        utm_crs = CRS(
            proj="utm", zone=int((centroid.x + 180) / 6) + 1, south=centroid.y < 0
        )
        self.crs_meters = utm_crs.to_string()

        # Placeholders
        self.coinc_mean = None
        self.coinc_std = None

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Placeholder method to be overridden
        """
        pass

    def collate(self, score: float):
        """
        Normalize the coincidence scores using the Source Type specific mean and standard deviation
        """
        return (score - self.coinc_mean) / self.coinc_std

    def apply_closing_buffer(self, geo_df: gpd.GeoDataFrame, closing_buffer: float):
        """
        Applies a closing buffer to geometries in the GeoDataFrame.
        """
        geo_df["geometry"] = (
            geo_df["geometry"].buffer(closing_buffer).buffer(-closing_buffer)
        )
        return geo_df

    def load_slick_centerlines(self):
        """
        Loads the slick centerlines from the GeoDataFrame.
        """
        self.slick_centerlines = gpd.GeoDataFrame.from_features(
            self.slick_gdf["centerlines"].iloc[0]["features"], crs="EPSG:4326"
        )


class AISAnalyzer(SourceAnalyzer):
    """
    Analyzer for AIS broadcasting vessels.

    Attributes:
        s1_scene (object): The Sentinel-1 scene object.
        ais_gdf (GeoDataFrame): Retrieved AIS data.
        ais_trajectories (TrajectoryCollection): Collection of vessel trajectories.
        results (DataFrame): Final association results.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the AISAnalyzer.
        """
        super().__init__(s1_scene, **kwargs)
        self.source_type = 1
        self.s1_scene = s1_scene
        # Default parameters
        self.hours_before = kwargs.get("hours_before", c.HOURS_BEFORE)
        self.hours_after = kwargs.get("hours_after", c.HOURS_AFTER)
        self.ais_scene_buffer = kwargs.get("ais_scene_buffer", c.AIS_SCENE_BUFFER)
        self.ais_slick_buffer = kwargs.get("ais_slick_buffer", c.AIS_SLICK_BUFFER)
        self.num_timesteps = kwargs.get("num_timesteps", c.NUM_TIMESTEPS)
        self.ais_project_id = kwargs.get("ais_project_id", c.AIS_PROJECT_ID)
        self.w_temporal = kwargs.get("w_temporal", c.W_TEMPORAL)
        self.w_proximity = kwargs.get("w_proximity", c.W_PROXIMITY)
        self.w_parity = kwargs.get("w_parity", c.W_PARITY)
        self.sharpness_parity = kwargs.get("sharpness_parity", c.SHARPNESS_PARITY)
        self.sharpness_prox = kwargs.get("sharpness_prox", c.SHARPNESS_PROX)
        self.sharpness_temp = kwargs.get("sharpness_temp", c.SHARPNESS_TEMP)
        self.ais_ref_time_over = kwargs.get("ais_ref_time_over", c.AIS_REF_TIME_OVER)
        self.ais_ref_time_under = kwargs.get("ais_ref_time_under", c.AIS_REF_TIME_UNDER)
        self.spread_rate = kwargs.get("spread_rate", c.SPREAD_RATE)
        self.grace_distance = kwargs.get("grace_distance", c.GRACE_DISTANCE)
        self.coinc_mean = kwargs.get("coinc_mean", c.VESSEL_MEAN)
        self.coinc_std = kwargs.get("coinc_std", c.VESSEL_STD)

        # Calculated values
        self.ais_start_time = self.s1_scene.start_time - timedelta(
            hours=self.hours_before
        )
        self.ais_end_time = self.s1_scene.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.ais_start_time,
            end=self.ais_end_time,
            periods=self.num_timesteps,
        ).astype("datetime64[s]")
        self.s1_env = gpd.GeoDataFrame(
            {"geometry": [to_shape(self.s1_scene.geometry)]}, crs="4326"
        )
        self.ais_envelope = (
            self.s1_env.to_crs(self.crs_meters)
            .buffer(self.ais_scene_buffer)
            .to_crs("4326")
        )
        self.credentials = Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        )

        # Initialize other attributes
        self.sql = None
        self.slick_centerlines = None
        self.ais_gdf = None
        self.ais_filtered = None
        self.ais_trajectories = {}
        self.results = gpd.GeoDataFrame()

    def retrieve_ais_data(self):
        """
        Retrieves AIS data from BigQuery.
        """
        start_time = datetime.strftime(self.ais_start_time, c.T_FORMAT)
        end_time = datetime.strftime(self.ais_end_time, c.T_FORMAT)
        ais_envelope = self.ais_envelope[0]

        # print("Retrieving AIS data")
        sql = f"""
            SELECT
                seg.ssvid as ssvid,
                seg.timestamp as timestamp,
                seg.lon as lon,
                seg.lat as lat,
                seg.course as course,
                seg.speed_knots as speed_knots,
                ves.ais_identity.shipname_mostcommon.value as shipname,
                ves.ais_identity.shiptype[SAFE_OFFSET(0)].value as shiptype,
                ves.best.best_flag as flag,
                ves.best.best_vessel_class as best_shiptype
            FROM
                `world-fishing-827.pipe_ais_v3_published.messages` as seg
            LEFT JOIN
                `world-fishing-827.pipe_ais_v3_published.vi_ssvid_v20250201` as ves
                ON seg.ssvid = ves.ssvid
            WHERE TRUE
                -- AND clean_segs IS TRUE
                AND seg.timestamp between '{start_time}' AND '{end_time}'
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{ais_envelope}'))
            """
        df = pandas_gbq.read_gbq(
            sql,
            project_id=self.ais_project_id,
            credentials=self.credentials,
        )
        df["timestamp"] = (
            df["timestamp"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[s]")
        )

        df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
        self.ais_gdf = (
            gpd.GeoDataFrame(df, crs="4326")
            .sort_values(by=["ssvid", "timestamp"])
            .reset_index(drop=True)
        )

    def build_trajectories(self):
        """
        Build and store AIS trajectories using available AIS data.
        """
        s1_time = self.s1_scene.start_time
        detail_lower = s1_time - timedelta(hours=3)
        detail_upper = s1_time + timedelta(hours=1)

        # Prepare search polygon in lat/lon
        search_polygon_m = (
            self.slick_gdf.geometry.to_crs(self.crs_meters)
            .buffer(self.ais_slick_buffer)
            .iloc[0]
        )

        ais_data_m = self.ais_gdf.to_crs(self.crs_meters).sort_values("timestamp")
        existing_ids = set(self.ais_trajectories.keys())
        ais_data_m = ais_data_m[~ais_data_m["ssvid"].isin(existing_ids)]

        for source_id, group in ais_data_m.groupby("ssvid"):
            # Project once to metric CRS for distances
            group_m = group.to_crs(self.crs_meters)
            geom_m = group_m.geometry

            t_known = pd.DatetimeIndex(group["timestamp"])
            t_first, t_last = t_known[0], t_known[-1]
            t_combined = t_known.union(self.time_vec).sort_values()

            if len(group_m) == 1:
                new_times = pd.DatetimeIndex([t_combined[0], t_first, t_combined[-1]])
                new_positions = [geom_m.iloc[0]] * 3

            elif t_first < detail_lower and t_last > detail_upper:
                if not box(*geom_m.total_bounds).intersects(search_polygon_m):
                    continue
                new_times = t_combined[(t_combined >= t_first) & (t_combined <= t_last)]
                new_positions = self.interpolate_positions(t_known, geom_m, new_times)

            else:
                # Extrapolation case
                max_speed = self.estimate_speed(group_m)  # m/s
                lower_delta = max((t_first - detail_lower).total_seconds(), 0)
                upper_delta = max((detail_upper - t_last).total_seconds(), 0)
                r0 = max_speed * lower_delta
                r1 = max_speed * upper_delta
                b0 = geom_m.iloc[0].buffer(r0) if r0 > 0 else None
                b1 = geom_m.iloc[-1].buffer(r1) if r1 > 0 else None
                pieces = [g for g in (b0, b1) if g is not None and not g.is_empty]
                if not unary_union(pieces).intersects(search_polygon_m):
                    continue
                new_times = t_combined[t_combined <= detail_upper]
                new_positions = self.bezier_extrapolation(
                    t_known,
                    geom_m,
                    new_times,
                    group_m["course"],
                    group_m["speed_knots"],
                )

            # Build trajectory GeoDataFrame and convert back to lat/lon
            interp_gdf = (
                gpd.GeoDataFrame(
                    {
                        "timestamp": new_times,
                        "geometry": new_positions,
                        "ssvid": source_id,
                        "extrapolated": (new_times <= t_first) | (new_times >= t_last),
                    },
                    crs=self.crs_meters,
                )
                .set_index("timestamp")
                .to_crs("EPSG:4326")
            )

            # Prepare GeoJSON for display
            display_gdf = interp_gdf[interp_gdf.index <= s1_time].copy()
            display_gdf["timestamp"] = display_gdf.index.strftime("%Y-%m-%dT%H:%M:%S")

            traj = {
                "id": source_id,
                "ext_name": group.iloc[0]["shipname"],
                "ext_shiptype": group.iloc[0]["best_shiptype"],
                "flag": group.iloc[0]["flag"],
                "first_timestamp": t_first,
                "last_timestamp": t_last,
                "df": interp_gdf,
                "geojson_fc": {
                    "type": "FeatureCollection",
                    "features": json.loads(display_gdf.to_json())["features"],
                },
            }
            self.ais_trajectories[source_id] = traj

    def estimate_speed(self, group_m, percentile=95):
        """
        group_m: GeoDataFrame in metric CRS with 'timestamp' and 'geometry'.
        Returns the 95th percentile speed (m/s) to robustly capture higher speeds.
        """
        gdf = group_m.sort_values("timestamp")
        geom = gdf.geometry
        ts = gdf["timestamp"]
        speeds = []
        for p1, p2, t1, t2 in zip(geom[:-1], geom[1:], ts[:-1], ts[1:]):
            dt = (t2 - t1).total_seconds()
            if dt > 0:
                speeds.append(p1.distance(p2) / dt)
        if not speeds:
            return 0.0
        return float(np.percentile(speeds, percentile))

    def bezier_extrapolation(
        self,
        known_times: pd.DatetimeIndex,
        known_points: gpd.GeoSeries,
        extrap_times: pd.DatetimeIndex,
        course_deg: gpd.GeoSeries,
        speed_knots: gpd.GeoSeries,
    ) -> list[Point]:
        """
        Piece‑wise interpolation / extrapolation.
        Returns one shapely Point per extrap_times entry.
        """
        # ensure times are sorted
        if not known_times.is_monotonic_increasing:
            order = np.argsort(known_times)
            known_times = known_times[order]
            known_points = known_points.iloc[order].reset_index(drop=True)
            course_deg = course_deg.iloc[order].reset_index(drop=True)
            speed_knots = speed_knots.iloc[order].reset_index(drop=True)

        xs = known_points.x.to_numpy()
        ys = known_points.y.to_numpy()

        thetas = np.deg2rad(course_deg.to_numpy())
        speeds = speed_knots.to_numpy() * 0.514444444

        vx = speeds * np.sin(thetas)
        vy = speeds * np.cos(thetas)

        n = len(known_times)
        dt_list = [
            (known_times[i + 1] - known_times[i]).total_seconds() for i in range(n - 1)
        ]

        # precompute control points for cubic Bezier on each segment
        if n > 1:
            B0x = xs[:-1]
            B0y = ys[:-1]
            B3x = xs[1:]
            B3y = ys[1:]
            arr_dt = np.array(dt_list)
            B1x = B0x + vx[:-1] * arr_dt / 3
            B1y = B0y + vy[:-1] * arr_dt / 3
            B2x = B3x - vx[1:] * arr_dt / 3
            B2y = B3y - vy[1:] * arr_dt / 3

        result = []
        times = extrap_times.to_numpy()
        idxs = np.searchsorted(known_times.to_numpy(), times)

        for t, idx in zip(times, idxs):
            t = pd.to_datetime(t)
            if idx == 0:
                # linear extrapolation before first point
                dt = (t - known_times[0]).total_seconds()
                x = xs[0] + vx[0] * dt
                y = ys[0] + vy[0] * dt
            elif idx >= n:
                # linear extrapolation after last point
                dt = (t - known_times[-1]).total_seconds()
                x = xs[-1] + vx[-1] * dt
                y = ys[-1] + vy[-1] * dt
            else:
                # cubic Bezier interpolation
                i = idx - 1
                u = (t - known_times[i]).total_seconds() / dt_list[i]
                one_u = 1 - u
                x = (
                    (one_u**3) * B0x[i]
                    + 3 * (one_u**2) * u * B1x[i]
                    + 3 * one_u * (u**2) * B2x[i]
                    + (u**3) * B3x[i]
                )
                y = (
                    (one_u**3) * B0y[i]
                    + 3 * (one_u**2) * u * B1y[i]
                    + 3 * one_u * (u**2) * B2y[i]
                    + (u**3) * B3y[i]
                )
            result.append(Point(x, y))

        return result

    def interpolate_positions(self, known_times, known_points, interp_times):
        """
        Given a sorted group (by timestamp) with a "geometry" column (shapely Points),
        and a Series of interpolation times, perform linear interpolation on the x and y
        coordinates using numpy.interp.

        Returns a list of shapely Point objects corresponding to the interpolated positions.
        """
        # Convert timestamps to numeric values (seconds since epoch)
        t_orig = known_times.astype("int64").values
        t_interp = interp_times.astype("int64").values

        # Extract x and y coordinates from the geometry column.
        xs = [pt.x for pt in known_points]
        ys = [pt.y for pt in known_points]

        # Use NumPy's vectorized linear interpolation.
        x_interp = np.interp(t_interp, t_orig, xs)
        y_interp = np.interp(t_interp, t_orig, ys)

        # Reconstruct shapely Points from the interpolated x and y.
        pos_out = [Point(x, y) for x, y in zip(x_interp, y_interp)]
        return pos_out

    def filter_ais_data(self):
        """
        Prune AIS data to only include trajectories that are within the AIS buffer.
        """
        search_polygon = (
            self.slick_gdf.geometry.to_crs(self.crs_meters)
            .buffer(self.ais_slick_buffer)
            .to_crs("4326")
            .iloc[0]
        )

        search_bounds = search_polygon.bounds  # (minx, miny, maxx, maxy)

        candidate_ssvids = []
        # Iterate over each trajectory.
        for ssvid, trajectory in self.ais_trajectories.items():
            traj_df = trajectory["df"]
            # Compute the bounding box of the trajectory.
            traj_bounds = traj_df.total_bounds  # [minx, miny, maxx, maxy]

            # Quickly check if the bounding boxes intersect.
            # Two bounding boxes intersect if:
            #   traj_bounds[0] <= search_bounds[2] and traj_bounds[2] >= search_bounds[0]
            #   and similarly in the y direction.
            if (
                traj_bounds[0] > search_bounds[2]
                or traj_bounds[2] < search_bounds[0]
                or traj_bounds[1] > search_bounds[3]
                or traj_bounds[3] < search_bounds[1]
            ):
                continue  # Bounding boxes do not intersect: skip this trajectory.

            # Now do a detailed check using the trajectory's spatial index.
            # This is faster than iterating over every point if many points are present.
            if traj_df.sindex.query(search_polygon, predicate="intersects").size > 0:
                # Double-check by testing the actual geometries.
                if traj_df.geometry.intersects(search_polygon).any():
                    candidate_ssvids.append(ssvid)

        # Filter the trajectories to only include those with candidate ssvids.
        self.filtered_ssvids = candidate_ssvids

    def score_trajectories(self):
        """
        Measure association by computing multiple metrics between AIS trajectories and slicks
        using multiple candidate centerlines. For each trajectory:
        - Aggregate the temporal score using the maximum value among candidates.
        - Aggregate the proximity and parity scores using a length-weighted average.

        Returns:
            GeoDataFrame of slick associations.
        """
        # Define the output columns.
        columns = [
            "st_name",
            "ext_id",
            "geometry",
            "coincidence_score",
            "type",
            "ext_name",
            "ext_shiptype",
            "flag",
            "geojson_fc",
        ]
        entries = []

        # Sort the slick centerlines by length (largest first) and convert to meters CRS.
        centerlines = self.slick_centerlines.sort_values(
            "length", ascending=False
        ).to_crs(self.crs_meters)

        # Determine candidate centerlines that contribute to 80% of the total length.
        # XXX Do we need this, after we've filtered on area?
        percent_to_keep = 0.8
        cumsum = centerlines["length"].cumsum()
        index_of_min = cumsum.searchsorted(percent_to_keep * cumsum.iloc[-1])
        candidate_centerlines = centerlines.iloc[: index_of_min + 1]

        # Get the list of AIS trajectories.
        relevant_trajectories = [
            self.ais_trajectories[ssvid] for ssvid in self.filtered_ssvids
        ]

        # Iterate over each AIS trajectory.
        for traj in relevant_trajectories:
            traj_gdf = (
                traj["df"]
                .sort_values(by="timestamp", ascending=False)
                .set_crs("4326")
                .to_crs(self.crs_meters)
            )

            # Initialize lists to collect scores from each candidate centerline.
            temporal_scores = []
            proximity_scores = []
            parity_scores = []
            weights = []  # We'll use the centerline lengths as weights.

            # Compute metrics for each candidate centerline.
            for idx, row in candidate_centerlines.iterrows():
                centerline_geom = row["geometry"]
                weight = row["length"]
                # Get the mapping between the trajectory and the candidate centerline.
                slick_to_traj_mapping = self.get_closest_centerline_points(
                    traj_gdf, centerline_geom, self.s1_scene.start_time
                )

                # Compute individual scores.
                temp_score = compute_temporal_score(
                    self.s1_scene.start_time,
                    self.ais_ref_time_over,
                    self.ais_ref_time_under,
                    self.sharpness_temp,
                    slick_to_traj_mapping,
                )
                prox_score = compute_proximity_score(
                    traj_gdf,
                    self.spread_rate,
                    self.grace_distance,
                    self.s1_scene.start_time,
                    self.sharpness_prox,
                    slick_to_traj_mapping,
                )
                par_score = compute_parity_score(
                    traj_gdf,
                    centerline_geom,
                    self.sharpness_parity,
                    slick_to_traj_mapping,
                )

                temporal_scores.append(temp_score)
                proximity_scores.append(prox_score)
                parity_scores.append(par_score)
                weights.append(weight)

                print(
                    f"Candidate centerline {idx}: temporal_score={round(temp_score, 2)}, "
                    f"proximity_score={round(prox_score, 2)}, parity_score={round(par_score, 2)}, "
                    f"weight={round(weight, 2)}"
                )

            # Aggregate the temporal score by taking the maximum.
            aggregated_temporal = max(temporal_scores) if temporal_scores else 0

            # Compute weighted averages for proximity and parity scores.
            if weights and sum(weights) > 0:
                aggregated_proximity = sum(
                    s * w for s, w in zip(proximity_scores, weights)
                ) / sum(weights)
                aggregated_parity = sum(
                    s * w for s, w in zip(parity_scores, weights)
                ) / sum(weights)
            else:
                aggregated_proximity = 0
                aggregated_parity = 0

            # Compute the final coincidence score using the aggregated metrics.
            aggregated_total = vessel_compute_total_score(
                aggregated_temporal,
                aggregated_proximity,
                aggregated_parity,
                self.w_temporal,
                self.w_proximity,
                self.w_parity,
            )

            print(
                f"Trajectory {traj['id']}: aggregated_total={round(aggregated_total, 2)} "
                f"(temporal={round(aggregated_temporal, 2)}, proximity={round(aggregated_proximity, 2)}, "
                f"parity={round(aggregated_parity, 2)})"
            )

            # Create a LineString geometry from the AIS trajectory points.
            traj_line = LineString([p.coords[0] for p in traj["df"]["geometry"]])
            entry = {
                "st_name": traj["id"],
                "ext_id": traj["id"],
                "geometry": traj_line,
                "coincidence_score": aggregated_total,
                "type": self.source_type,
                "ext_name": traj["ext_name"],
                "ext_shiptype": traj["ext_shiptype"],
                "flag": traj["flag"],
                "geojson_fc": traj["geojson_fc"],
            }
            entries.append(entry)

        sources = gpd.GeoDataFrame(entries, columns=columns, crs="4326")
        self.results = sources[sources["coincidence_score"] > 0]
        return self.results

    def get_closest_centerline_points(
        self,
        traj_gdf: gpd.GeoDataFrame,
        longest_centerline: MultiLineString,
        t_image: datetime = None,
    ) -> tuple[Point, pd.Timestamp, float, Point, pd.Timestamp, float]:
        """
        Returns the timestamp and distance of the closest points on the centerline to the vessel at the given image_timestamp.
        """
        # Create centerline endpoints
        cl_A = Point(longest_centerline.coords[0])
        cl_B = Point(longest_centerline.coords[-1])

        # Find nearest trajectory point indices for each endpoint
        t_A, d_A = self.get_closest_point_near_timestamp(cl_A, traj_gdf, t_image)
        t_B, d_B = self.get_closest_point_near_timestamp(cl_B, traj_gdf, t_image)

        # Sort the pairs by timestamp to determine head and tail
        (cl_tail, t_tail, d_tail), (cl_head, t_head, d_head) = sorted(
            [(cl_A, t_A, d_A), (cl_B, t_B, d_B)], key=lambda x: x[1]
        )

        # After finding the head (slick end closest to the AIS), project the tail to the nearest point independent of time.
        t_tail, d_tail = self.get_closest_point_near_timestamp(cl_tail, traj_gdf)

        return (cl_tail, t_tail, d_tail, cl_head, t_head, d_head)

    def get_closest_point_near_timestamp(
        self,
        target: Point,
        traj_gdf: gpd.GeoDataFrame,
        t_image: datetime = None,
        n_points: int = 10,
    ) -> tuple[pd.Timestamp, float]:
        """
        Returns the trajectory row that is closest to the reference_point,
        using a turning-point heuristic starting at t_image.

        It starts at t_image and checks the immediate neighbors to determine
        in which temporal direction the distance to the reference point is decreasing.
        It then traverses in that single direction until the distance no longer decreases,
        returning the last point before an increase is detected.

        Parameters:
            reference_point (shapely.geometry.Point): The point to compare distances to.
            traj_gdf (geopandas.GeoDataFrame): A GeoDataFrame with a datetime-like index
                and a 'geometry' column.
            t_image (datetime): The starting timestamp for the search (guaranteed to be in the dataset).
            n_points (int): The number of subsequent points that must confirm the turning point.

        Returns:
            pd.Timestamp: The index corresponding to the selected trajectory row.
        """
        if t_image is None:
            # If no timestamp is provided, return the index of the point closest to the target.
            distances = traj_gdf.geometry.distance(target)
            return distances.idxmin(), distances.min()

        # Get the starting position for t_image.
        traj_gdf = traj_gdf.sort_index(ascending=True)
        pos = np.abs(traj_gdf.index - t_image).argmin()
        best_dist = traj_gdf.iloc[pos].geometry.distance(target)

        # Determine direction to traverse.
        if pos == 0:
            direction = 1
        else:
            backward_distance = traj_gdf.iloc[pos - 1].geometry.distance(target)
            # Pick the direction with a decreasing distance.
            direction = -1 if backward_distance < best_dist else 1

        while 0 <= pos + direction < len(traj_gdf):
            pos += direction
            d = traj_gdf.iloc[pos].geometry.distance(target)
            if d < best_dist:
                best_dist = d
            else:
                # Get the next n_points indices in the chosen direction that are within bounds.
                check_indices = [
                    pos + i * direction
                    for i in range(1, n_points + 1)
                    if 0 <= pos + i * direction < len(traj_gdf)
                ]
                # Check that for each of these indices, the distance is greater than best_dist.
                distances = np.array(
                    [
                        traj_gdf.iloc[idx].geometry.distance(target)
                        for idx in check_indices
                    ]
                )
                if all(distances > best_dist):
                    break
        return traj_gdf.index[pos], best_dist

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Associates AIS trajectories with slicks.
        """
        self.results = gpd.GeoDataFrame()
        self.filtered_ssvids = []

        if self.ais_gdf is None:
            self.retrieve_ais_data()
        if self.ais_gdf.empty:
            return pd.DataFrame()

        self.slick_gdf = slick_gdf
        self.load_slick_centerlines()
        self.build_trajectories()
        self.filter_ais_data()
        self.score_trajectories()
        self.results["collated_score"] = self.results["coincidence_score"].apply(
            self.collate
        )
        return self.results


class PointAnalyzer(SourceAnalyzer):
    """
    Analyzer for specified points.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the PointAnalyzer.
        """
        super().__init__(s1_scene, **kwargs)
        # Placeholders
        self.cutoff_radius = 0
        self.decay_radius = 0
        self.decay_theta = 0
        self.num_vertices = 0

    def process_slicks(self):
        """
        Processes slicks and returns combined geometry, overall centroid, polygons, and largest polygon area.
        """
        slick_gdf = self.slick_gdf
        # Reproject to meters
        slick_m = slick_gdf.to_crs(self.crs_meters)

        # Generate closed polygons
        slick_closed = self.apply_closing_buffer(slick_m, self.closing_buffer)
        combined_geometry = slick_closed.unary_union
        if isinstance(combined_geometry, (MultiPolygon, GeometryCollection)):
            polygons = [g for g in combined_geometry.geoms if isinstance(g, Polygon)]
        else:
            polygons = [combined_geometry]
        largest_polygon_area = max(polygon.area for polygon in polygons)

        return (combined_geometry, polygons, largest_polygon_area)

    def filter_points(
        self,
        combined_geometry_m: Polygon,
        points_gdf: gpd.GeoDataFrame,
    ):
        """
        Filters points based on their proximity to the combined geometry.

        Args:
            combined_geometry (Polygon): The combined geometry to filter points by.
            points_gdf (gpd.GeoDataFrame): The points to filter.

        Returns:
            gpd.GeoDataFrame: The filtered points.
        """
        # Reproject to meters
        points_m = points_gdf.to_crs(self.crs_meters)

        # Buffer and filter target points
        slick_buffered = combined_geometry_m.buffer(self.cutoff_radius)

        # scene_date = pd.to_datetime(self.s1_scene.scene_id[17:25], format="%Y%m%d") # XXX Not working because of a bug in how GFW records the first and last date.
        filtered_points = points_m[
            points_m.geometry.within(slick_buffered)
            # & (infra_gdf["structure_start_date"] < scene_date)
            # & (infra_gdf["structure_end_date"] > scene_date)
        ]

        return filtered_points

    def aggregate_extrema_and_area_fractions(
        self,
        polygons: List[Polygon],
        combined_geometry: Polygon,
        largest_polygon_area: float,
        min_area_threshold: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a list of polygons, and a number of vertices, collects that many extremity points for each polygon
        and then calculates a score for each point based on their distance from the centroid and their scaled
        area fraction compared to the largest polygon.

        Returns:
            all_extrema: A list of all collected extremity points.
            all_weights: A list of weights for each extremity point. (the max weight is 1, representing the
                furthest point that on the polygon that has the most area)
        """
        overall_centroid = np.array(combined_geometry.centroid.coords[0])

        extrema_list = []
        area_fractions_list = []

        for polygon in polygons:
            if polygon.area < min_area_threshold * largest_polygon_area:
                continue  # Skip small polygons

            # Select N extremity points for the current polygon
            selected_points = self.select_N_polygon_extrema(
                polygon, self.num_vertices, [overall_centroid]
            )
            extrema_list.append(selected_points)

            # Compute scaled area fraction for weighting
            area_fraction = polygon.area / largest_polygon_area

            # XXX This is a hack to make the area fraction more sensitive to small areas
            scaled_area_fraction = np.sqrt(area_fraction)
            area_fractions_list.extend([scaled_area_fraction] * self.num_vertices)

        if len(extrema_list) == 0:
            raise ValueError("No extremity points collected from polygons.")

        all_extrema = np.vstack(extrema_list)
        all_area_fractions = np.array(area_fractions_list)

        # Calculate distances from centroid
        distances_sq = np.sum((all_extrema - overall_centroid) ** 2, axis=1)
        # Scale weights by area fraction
        scaled_weights = distances_sq * all_area_fractions
        # Normalize weights to ensure the maximum weight is 1
        max_weight = scaled_weights.max()
        if max_weight != 0:
            all_weights = scaled_weights / max_weight
        else:
            all_weights = np.ones_like(scaled_weights)

        return all_extrema, all_weights

    def select_N_polygon_extrema(
        self, polygon: Polygon, num_vertices: int, reference_points: List[np.ndarray]
    ) -> np.ndarray:
        """
        Selects N extremity points from a single polygon based on their distance from reference points.
        """
        exterior_coords = np.array(
            polygon.exterior.coords[:-1]
        )  # Exclude closing point
        selected_points = []

        for _ in range(num_vertices):
            # Compute distances from all exterior points to reference points
            diff = (
                exterior_coords[:, np.newaxis, :] - reference_points
            )  # Shape: (M, K, 2)
            dists = np.linalg.norm(diff, axis=2)  # Shape: (M, K)
            min_dists = dists.min(axis=1)  # Shape: (M,)

            # Select the point with the maximum of these minimum distances
            idx = np.argmax(min_dists)
            selected_point = exterior_coords[idx]
            selected_points.append(selected_point)
            reference_points.append(selected_point)  # Update reference points

        return np.array(selected_points)

    def scaled_inner_angles(self, a, b_set, c_set):
        """
        Calculate scaled inner angles at vertex C for triangles formed by a fixed point A,
        and corresponding points in b_set and c_set.

        Each triangle is defined by the vertices (A, B, C), and the inner angle at C
        is computed using the vectors CA (from C to A) and CB (from C to B).

        Parameters:
            a: Tuple representing point A (x, y) (fixed for all triangles).
            b_set: List or array of tuples representing point B (x, y) for each triangle.
            c_set: List or array of tuples representing point C (x, y) for each triangle.

        Returns:
            A NumPy array of scaled inner angles at vertex C for each triangle (range: 0-1,
            where 0 corresponds to 0° and 1 to 180°).
        """
        # Convert inputs to NumPy arrays for vectorized operations
        a = np.array(a)
        b_set = np.array(b_set)
        c_set = np.array(c_set)

        # Compute vectors CA and CB for each triangle (vertex C is the reference)
        ca = a - c_set  # Vector from C to A
        cb = b_set - c_set  # Vector from C to B

        # Compute dot products for each pair of vectors
        dot_products = np.sum(ca * cb, axis=1)

        # Compute the magnitudes of each vector
        norm_ca = np.linalg.norm(ca, axis=1)
        norm_cb = np.linalg.norm(cb, axis=1)

        # Compute the cosine of the angle at vertex C
        cos_angles = dot_products / (norm_ca * norm_cb)
        # Clip values to ensure they lie in [-1, 1] to prevent numerical errors in arccos
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # Compute the angle at vertex C in radians
        angles_radians = np.arccos(cos_angles)

        # Scale the angles to the range 0-1 (0 rad = 0, π rad = 1)
        scaled_angles = angles_radians / np.pi

        return 1 - scaled_angles

    def calc_points_to_extrema_scores(
        self,
        points_gdf: gpd.GeoDataFrame,
        extrema: np.ndarray,
        weights: np.ndarray,
        secondary_points: np.ndarray,
    ) -> np.ndarray:
        """
        Computes confidence scores for points based on their proximity to extremity points.
        """
        points_coords = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])
        coincidence_scores = np.zeros(len(points_coords))
        if len(secondary_points) == 1:
            secondary_points = np.tile(secondary_points, (len(extrema), 1))

        point_to_neighbor_idxs = cKDTree(extrema).query_ball_point(
            points_coords, r=self.cutoff_radius
        )

        for i, neighbor_idxs in enumerate(point_to_neighbor_idxs):
            if neighbor_idxs:
                extrema_reduced = extrema[neighbor_idxs]
                weights_reduced = weights[neighbor_idxs]
                secondary_reduced = secondary_points[neighbor_idxs]
                dists = np.linalg.norm(extrema_reduced - points_coords[i], axis=1)

                scaled_angles = self.scaled_inner_angles(
                    points_coords[i], secondary_reduced, extrema_reduced
                )

                C_i = (
                    weights_reduced
                    * np.exp(-scaled_angles * self.decay_theta)
                    * np.exp(-dists / self.decay_radius)
                )

                coincidence_scores[i] = np.clip(C_i.max(), 0, 1)

        return coincidence_scores

    def make_geojson_feature(self, geom):
        """
        Creates a GeoJSON feature from a Shapely geometry.

        Args:
            geom (Polygon): The geometry to convert to GeoJSON.

        Returns:
            dict: The GeoJSON feature.
        """
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "0",
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {},  # add extra properties as needed
                }
            ],
        }

    def get_endpoint_pairs(
        self, line: LineString, gap_pct: float, primary_pct: float = 0.0
    ):
        """
        Select endpoint pairs from a line from a specified offset and gap in meters
        """
        if not (0 <= primary_pct <= 0.5):
            raise ValueError(
                f"primary_pct must be between 0 and 0.5. Primary: {primary_pct}"
            )
        if gap_pct + primary_pct > 1:
            raise ValueError(
                f"gap_pct + primary_pct must be less than or equal to 1. Gap: {gap_pct}, Primary: {primary_pct}"
            )

        total_length = line.length
        secondary_pct = primary_pct + gap_pct

        # For the start of the line:
        start_primary = line.interpolate(total_length * primary_pct)
        start_secondary = line.interpolate(total_length * secondary_pct)

        # For the end of the line:
        end_primary = line.interpolate(total_length * (1 - primary_pct))
        end_secondary = line.interpolate(total_length * (1 - secondary_pct))

        return (
            (start_primary.coords[0], end_primary.coords[0]),
            (start_secondary.coords[0], end_secondary.coords[0]),
        )

    def calc_cl_extrema_and_weights(
        self,
        offset=None,
        gap=None,
        min_length_threshold: float = 0.1,
    ):
        """
        Given a list of LineString geometries and a center point, this function:
        - Extracts endpoints (primary_points) and delta points (secondary_points) from each LineString.
        - Calculates base weights from distances from centroid.
        - Scales weights by length fraction.
        - Normalizes weights to ensure the maximum weight is 1.

        Args:
            offset (float, optional): The offset from the start of the line to the primary point. Defaults to None.
            gap (float, optional): The gap between the primary and secondary points. Defaults to None.
            expects to be called by a class that has a slick_centerlines and combined_geometry attribute.

        Returns:
            primary_points: a numpy array containing the selected endpoints.
            secondary_points: a numpy array containing the corresponding delta points.
            normalized_weights: a numpy array of weights for each point.
        """
        cl_array = self.slick_centerlines.to_crs(self.crs_meters).geometry.values
        offset = self.endpoints_offset if offset is None else offset
        gap = self.endpoints_gap if gap is None else gap

        primary_points = []
        secondary_points = []
        base_weights = np.array([])

        for line in cl_array:
            primaries, secondaries = self.get_endpoint_pairs(line, gap, offset)
            primary_points.extend(primaries)
            secondary_points.extend(secondaries)

            # Use the length of the line as weights for both ends of the line
            base_weights = np.append(base_weights, [line.length] * 2)

        # Scale weights by length fraction
        base_weights = np.array(
            [
                length if length / max(base_weights) > min_length_threshold else 0
                for length in base_weights
            ]
        )
        scaled_weights = base_weights

        # Normalize weights to ensure the maximum weight is 1
        normalized_weights = scaled_weights / scaled_weights.max()
        return np.array(primary_points), np.array(secondary_points), normalized_weights


class InfrastructureAnalyzer(PointAnalyzer):
    """
    Analyzer for fixed infrastructure sources.

    Attributes:
        infra_gdf (GeoDataFrame): GeoDataFrame containing infrastructure points.
        coincidence_scores (np.ndarray): Computed confidence scores.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the InfrastructureAnalyzer.
        """
        super().__init__(s1_scene, **kwargs)
        self.source_type = 2
        self.num_vertices = kwargs.get("num_vertices", c.INFRA_NUM_VERTICES)
        self.closing_buffer = kwargs.get("closing_buffer", c.INFRA_CLOSING_BUFFER)
        self.cutoff_radius = kwargs.get("cutoff_radius", c.INFRA_CUTOFF_RADIUS)
        self.decay_radius = kwargs.get("decay_radius", c.INFRA_DECAY_RADIUS)
        self.decay_theta = kwargs.get("decay_theta", c.INFRA_DECAY_THETA)
        self.min_area_threshold = kwargs.get("min_area_threshold", c.MIN_AREA_THRESHOLD)
        self.coinc_mean = kwargs.get("coinc_mean", c.INFRA_MEAN)
        self.coinc_std = kwargs.get("coinc_std", c.INFRA_STD)

        self.infra_gdf = kwargs.get("infra_gdf", None)

        if self.infra_gdf is None:
            self.infra_api_token = os.getenv("INFRA_API_TOKEN")
            self.infra_gdf = self.retrieve_infrastructure_data()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))

    def retrieve_infrastructure_data(self, only_oil=True):
        """
        Loads infrastructure data from the GFW API.

        Parameters:
            only_oil (bool): Whether to filter the data to only include oil infrastructure.

        Returns:
            GeoDataFrame: GeoDataFrame containing infrastructure points.
        """

        # zxy = self.select_enveloping_tile() # Something's wrong with this code. Ex. [3105854, 'S1A_IW_GRDH_1SDV_20230806T221833_20230806T221858_049761_05FBD2_577C"] should have 2 nearby infras
        mvt_data = self.download_mvt_tile()
        df = pd.DataFrame([d["properties"] for d in mvt_data["main"]["features"]])
        if only_oil:
            df = df[df["label"] == "oil"].copy()
        df["st_name"] = df["structure_id"].apply(str)
        df["ext_id"] = df["structure_id"].apply(str)
        df["type"] = self.source_type

        datetime_fields = ["structure_start_date", "structure_end_date"]
        for field in datetime_fields:
            if field in df.columns:
                df.loc[df[field] == "", field] = str(int(time.time() * 1000))
                df[field] = pd.to_numeric(df[field], errors="coerce")
                df[field] = pd.to_datetime(df[field], unit="ms", errors="coerce")
        df = df.reset_index(drop=True)
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="epsg:4326"
        )

    def select_enveloping_tile(self, max_zoom=20):
        """
        Determine the minimal zoom level and tile coordinates (x, y, z)
        that cover the area of interest (slick_gdf buffered by cutoff_radius)
        in a single tile.
        """

        buffered_slick_gdf = (
            self.slick_gdf.to_crs(self.crs_meters)
            .envelope.buffer(self.cutoff_radius)
            .to_crs(epsg=4326)
        )
        bbox = buffered_slick_gdf.total_bounds

        TMS = morecantile.tms.get("WebMercatorQuad")
        for z in reversed(range(max_zoom + 1)):
            tiles = list(TMS.tiles(*bbox, zooms=z))
            if len(tiles) == 1:
                tile = tiles[0]
                return z, tile.x, tile.y

    def download_mvt_tile(self, z=0, x=0, y=0):
        """
        Downloads MVT tile data for given z, x, y.

        Parameters:
            z (int): Zoom level.
            x (int): Tile x coordinate.
            y (int): Tile y coordinate.
            token (str): Authorization token.

        Returns:
            bytes: The content of the MVT tile.
        """
        url = f"https://gateway.api.globalfishingwatch.org/v3/datasets/public-fixed-infrastructure-filtered:latest/context-layers/{z}/{x}/{y}"
        headers = {"Authorization": f"Bearer {self.infra_api_token}"}
        response = requests.get(url, headers=headers)
        try:
            decoded_tile = mapbox_vector_tile.decode(response.content)
            return decoded_tile
        except Exception:
            print(f"Error decoding tile z={z}, x={x}, y={y}: {response.content}")
            raise Exception(response.content)

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Computes coincidence scores for infrastructure points.
        """
        start_time = time.time()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))
        self.slick_gdf = slick_gdf

        combined_geometry, polygons, largest_polygon_area = self.process_slicks()
        filtered_infra = self.filter_points(combined_geometry, self.infra_gdf)

        if filtered_infra.empty:
            print(
                "No infrastructure within the dates / radius of interest. No coincidence scores edited."
            )
            return

        # Collect extremity points and compute weights
        extrema, weights = self.aggregate_extrema_and_area_fractions(
            polygons, combined_geometry, largest_polygon_area, self.min_area_threshold
        )
        secondary_point = np.array([combined_geometry.centroid.coords[0]])

        # Build KD-Tree and compute confidence scores
        coincidence_filtered = self.calc_points_to_extrema_scores(
            filtered_infra, extrema, weights, secondary_point
        )

        self.coincidence_scores[filtered_infra.index] = coincidence_filtered

        # Return a DataFrame with infra_gdf and coincidence_scores
        results = self.infra_gdf.copy()
        results["coincidence_score"] = self.coincidence_scores
        results = results[results["coincidence_score"] > 0]
        results["geojson_fc"] = results["geometry"].apply(self.make_geojson_feature)
        results["collated_score"] = results["coincidence_score"].apply(self.collate)
        self.results = results

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")

        return self.results


class DarkAnalyzer(PointAnalyzer):
    """
    Analyzer for dark vessels (non-AIS broadcasting vessels).
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the DarkAnalyzer.
        """
        super().__init__(s1_scene, **kwargs)
        self.source_type = 3
        self.credentials = Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        )
        self.gfw_project_id = "world-fishing-827"
        self.s1_scene = s1_scene
        self.dark_objects_gdf = kwargs.get("dark_vessels_gdf", None)
        if self.dark_objects_gdf is None:
            self.retrieve_sar_detection_data()
        self.closing_buffer = kwargs.get("closing_buffer", c.DARK_CLOSING_BUFFER)
        self.decay_theta = kwargs.get("decay_theta", c.DARK_DECAY_THETA)
        self.decay_radius = kwargs.get("decay_radius", c.DARK_DECAY_RADIUS)
        self.coinc_mean = kwargs.get("coinc_mean", c.DARK_MEAN)
        self.coinc_std = kwargs.get("coinc_std", c.DARK_STD)
        self.cutoff_radius = kwargs.get("cutoff_radius", c.DARK_CUTOFF_RADIUS)
        self.num_vertices = kwargs.get("num_vertices", c.DARK_NUM_VERTICES)
        self.endpoints_offset = kwargs.get("endpoints_offset", 0.05)
        self.endpoints_gap = kwargs.get("endpoints_gap", 0.05)
        self.slick_centerlines = None

    def retrieve_sar_detection_data(self):
        """
        Retrieves SAR detections from GFW.
        """
        sql = f"""
        -- Step 1: Define the list of scene_ids as a CTE for efficient joining
        WITH scene_ids AS (
        SELECT '{self.s1_scene.scene_id}' AS scene_id
        ),

        -- Step 2: Filter the match table by joining with scene_ids
        filtered_matches AS (
        SELECT match.*
        FROM `world-fishing-827.pipe_sar_v1_published.detect_scene_match_pipe_v3` AS match
        INNER JOIN scene_ids
            ON match.scene_id = scene_ids.scene_id
        WHERE match.score < .01 -- either no match or low confidence match
        ),

        -- Step 3: Optimize the unique_infra CTE with pre-filtering using a bounding box
        unique_infra AS (
        SELECT
            infra.*,
            ROW_NUMBER() OVER (
            PARTITION BY infra.structure_id
            ORDER BY infra.label_confidence DESC  -- Assuming you want the highest confidence
            ) AS rn
        FROM `world-fishing-827.pipe_sar_v1_published.published_infrastructure` AS infra
        INNER JOIN filtered_matches AS match
            -- Define a rough bounding box around detection points to limit infra records
            ON ABS(infra.lat - match.detect_lat) < 0.001  -- Approx ~100 meters latitude
            AND ABS(infra.lon - match.detect_lon) < 0.001  -- Approx ~100 meters longitude
        )

        -- Step 4: Final SELECT with optimized joins and distance calculation
        SELECT
            match.scene_id AS scene_id,
            match.ssvid AS ssvid,
            infra.structure_id AS structure_id,
            pred.presence AS detection_probability,
            match.detect_lat AS detect_lat,
            match.detect_lon AS detect_lon,
            pred.length_m AS length_m,
        FROM `world-fishing-827.pipe_sar_v1_published.detect_scene_pred` AS pred
        INNER JOIN filtered_matches AS match
            ON pred.detect_id = match.detect_id
        LEFT JOIN unique_infra AS infra
            ON infra.rn = 1
            AND ST_DISTANCE(
                ST_GEOGPOINT(match.detect_lon, match.detect_lat),
                ST_GEOGPOINT(infra.lon, infra.lat)
                ) < 100  -- Distance in meters
        WHERE pred.length_m > 30 -- only keep detections with length > 30m
        AND pred.presence > 0.99 -- only keep detections with high confidence
        AND infra.structure_id IS NULL -- ignore infra detections because they are captured by the infrastructure analyzer
        """

        df = pandas_gbq.read_gbq(
            sql,
            project_id=self.gfw_project_id,
            credentials=self.credentials,
        )

        df["geometry"] = df.apply(
            lambda row: Point(row["detect_lon"], row["detect_lat"]),
            axis=1,
        )

        # XXX need a better solution for a unique name here.
        # This code chooses the ssvid, or if that is null, the structure_id, or if that is null, the length_m.
        def make_unique_id(row):
            # if pd.notna(row["ssvid"]):
            #     return "V" + str(row["ssvid"])
            # elif pd.notna(row["structure_id"]):
            #     return "I" + str(row["structure_id"])
            # else:
            return "D" + str(row["length_m"])

        df["st_name"] = df.apply(make_unique_id, axis=1)
        df["ext_id"] = df["st_name"]
        df["type"] = self.source_type

        self.dark_objects_gdf = gpd.GeoDataFrame(df, crs="4326").reset_index(drop=True)

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Computes coincidence scores for specified points.
        """

        start_time = time.time()
        self.coincidence_scores = np.zeros(len(self.dark_objects_gdf))
        self.slick_gdf = slick_gdf

        self.combined_geometry, _, _ = self.process_slicks()

        filtered_dark_objects = self.filter_points(
            self.combined_geometry, self.dark_objects_gdf
        )

        if filtered_dark_objects.empty:
            print(
                "No dark objects within the radius of interest. No coincidence scores edited."
            )
            return

        # Collect extremity points and compute weights
        self.load_slick_centerlines()

        extrema, secondary_points, weights = self.calc_cl_extrema_and_weights()

        # Build KD-Tree and compute confidence scores
        coincidence_filtered = self.calc_points_to_extrema_scores(
            filtered_dark_objects, extrema, weights, secondary_points
        )

        self.coincidence_scores[filtered_dark_objects.index] = coincidence_filtered

        # Return a DataFrame with geojson, coincidence_scores, and collated_score
        results = self.dark_objects_gdf.copy()
        results["coincidence_score"] = self.coincidence_scores
        results = results[results["coincidence_score"] > 0]
        results["geojson_fc"] = results["geometry"].apply(self.make_geojson_feature)
        results["collated_score"] = results["coincidence_score"].apply(self.collate)

        self.results = results

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")
        return self.results


class NaturalAnalyzer(SourceAnalyzer):
    """
    Analyzer for natural seeps.
    Currently a placeholder for future implementation.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the NaturalAnalyzer.
        """
        super().__init__(s1_scene, **kwargs)
        self.source_type = 4
        # Initialize attributes specific to natural seep analysis


ASA_MAPPING = {
    1: AISAnalyzer,
    2: InfrastructureAnalyzer,
    3: DarkAnalyzer,
    4: NaturalAnalyzer,
}
