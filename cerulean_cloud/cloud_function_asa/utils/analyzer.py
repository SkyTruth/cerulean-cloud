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
import movingpandas as mpd
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
    mapping,
)

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
        self.ais_buffer = kwargs.get("ais_buffer", c.AIS_BUFFER)
        self.num_timesteps = kwargs.get("num_timesteps", c.NUM_TIMESTEPS)
        self.ais_project_id = kwargs.get("ais_project_id", c.AIS_PROJECT_ID)
        self.w_temporal = kwargs.get("w_temporal", c.W_TEMPORAL)
        self.w_proximity = kwargs.get("w_proximity", c.W_PROXIMITY)
        self.w_parity = kwargs.get("w_parity", c.W_PARITY)
        self.sensitivity_parity = kwargs.get("sensitivity_parity", c.SENSITIVITY_PARITY)
        self.ais_ref_time_over = kwargs.get("ais_ref_time_over", c.AIS_REF_TIME_OVER)
        self.ais_ref_time_under = kwargs.get("ais_ref_time_under", c.AIS_REF_TIME_UNDER)
        self.spread_rate = kwargs.get("spread_rate", c.SPREAD_RATE)
        self.coinc_mean = kwargs.get("coinc_mean", c.VESSEL_MEAN)
        self.coinc_std = kwargs.get("coinc_std", c.VESSEL_STD)
        self.ais_trajectories = kwargs.get("ais_trajectories", None)

        # Calculated values
        self.ais_start_time = self.s1_scene.start_time - timedelta(
            hours=self.hours_before
        )
        self.ais_end_time = self.s1_scene.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.ais_start_time,
            end=self.ais_end_time,
            periods=self.num_timesteps,
        )
        self.s1_env = gpd.GeoDataFrame(
            {"geometry": [to_shape(self.s1_scene.geometry)]}, crs="4326"
        )
        self.ais_envelope = (
            self.s1_env.to_crs(self.crs_meters).buffer(self.ais_buffer).to_crs("4326")
        )
        self.credentials = Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        )

        # Initialize other attributes
        self.sql = None
        self.slick_centerlines = None
        self.ais_gdf = None
        self.ais_filtered = None
        self.results = gpd.GeoDataFrame()

    def retrieve_ais_data(self):
        """
        Retrieves AIS data from BigQuery.
        """
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
                `world-fishing-827.gfw_research.pipe_v20201001` as seg
            LEFT JOIN
                `world-fishing-827.gfw_research.vi_ssvid_v20230801` as ves
                ON seg.ssvid = ves.ssvid
            WHERE
                seg._PARTITIONTIME between '{datetime.strftime(self.ais_start_time, c.D_FORMAT)}' AND '{datetime.strftime(self.ais_end_time, c.D_FORMAT)}'
                AND seg.timestamp between '{datetime.strftime(self.ais_start_time, c.T_FORMAT)}' AND '{datetime.strftime(self.ais_end_time, c.T_FORMAT)}'
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{self.ais_envelope[0]}'))
            """
        df = pandas_gbq.read_gbq(
            sql,
            project_id=self.ais_project_id,
            credentials=self.credentials,
        )
        df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
        self.ais_gdf = (
            gpd.GeoDataFrame(df, crs="4326")
            .sort_values(by=["ssvid", "timestamp"])
            .reset_index(drop=True)
        )

    def build_trajectories(self):
        """
        Builds trajectories from AIS data.

        - Creates a fully interpolated trajectory (with datetime timestamps) for scoring.
        - Generates a truncated version (with ISO-formatted timestamps) for display.
        """
        # print("Building trajectories")
        # Convert the entire timestamp column.
        self.ais_filtered = self.ais_filtered.sort_values("timestamp")
        self.ais_filtered["timestamp"] = (
            self.ais_filtered["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        )

        ais_trajectories = []
        s1_time = self.s1_scene.start_time

        # Group the filtered AIS data by ship identifier (ssvid)
        for st_name, group in self.ais_filtered.groupby("ssvid"):
            group = group.copy()  # avoid chained assignment issues

            # If only one point is present, we cannot interpolate.
            if len(group) == 1:
                print(f"Trajectory {st_name} has only one point, cannot interpolate")
                continue

            # Create a trajectory object
            traj = mpd.Trajectory(df=group, traj_id=st_name, t="timestamp")
            traj.ext_name, traj.ext_shiptype, traj.flag = group.iloc[0][
                ["shipname", "best_shiptype", "flag"]
            ]

            # Get the first and last timestamp of the AIS data for this trajectory.
            first_ais_tstamp = group["timestamp"].min()
            last_ais_tstamp = group["timestamp"].max()

            # Interpolate to times in time_vec
            interp_times = self.time_vec[
                (self.time_vec >= first_ais_tstamp) & (self.time_vec <= last_ais_tstamp)
            ]
            # Add three critical times: first, s1_time(if in range), last.
            pos = interp_times.searchsorted(first_ais_tstamp)
            interp_times = interp_times.insert(pos, first_ais_tstamp)
            if first_ais_tstamp < s1_time < last_ais_tstamp:
                pos = interp_times.searchsorted(s1_time)
                interp_times = interp_times.insert(pos, s1_time)
            pos = interp_times.searchsorted(last_ais_tstamp)
            interp_times = interp_times.insert(pos, last_ais_tstamp)

            # Interpolate positions at the required times.
            positions = self.vectorized_interpolate_positions(group, interp_times)

            # Build a full GeoDataFrame for scoring (timestamps remain as datetime objects).
            interp_gdf = gpd.GeoDataFrame(
                {"timestamp": interp_times, "geometry": positions}, crs="4326"
            ).set_index("timestamp")
            traj.df = interp_gdf

            # Create a truncated GeoDataFrame for display (only points before s1_time).
            display_gdf = group[group["timestamp"] < s1_time].copy()
            # Use vectorized formatting to convert timestamps to ISO strings.
            display_gdf["timestamp"] = display_gdf["timestamp"].dt.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            traj.geojson_fc = {
                "type": "FeatureCollection",
                "features": json.loads(display_gdf.to_json())["features"],
            }

            ais_trajectories.append(traj)

        self.ais_trajectories = mpd.TrajectoryCollection(ais_trajectories)

    def vectorized_interpolate_positions(self, group, interp_times):
        """
        Given a sorted group (by timestamp) with a "geometry" column (shapely Points),
        and a Series of interpolation times, perform linear interpolation on the x and y
        coordinates using numpy.interp.

        Returns a list of shapely Point objects corresponding to the interpolated positions.
        """
        # Convert timestamps to numeric values (nanoseconds since epoch)
        # It is important that both arrays use the same numeric units.
        t_orig = group["timestamp"].astype("datetime64[s]").astype("int64").values
        t_interp = interp_times.astype("datetime64[s]").astype("int64").values

        # Extract x and y coordinates from the geometry column.
        xs = group["geometry"].x.values
        ys = group["geometry"].y.values

        # Use NumPy's vectorized linear interpolation.
        x_interp = np.interp(t_interp, t_orig, xs)
        y_interp = np.interp(t_interp, t_orig, ys)

        # Reconstruct shapely Points from the interpolated x and y.
        positions = [Point(x, y) for x, y in zip(x_interp, y_interp)]
        return positions

    def filter_ais_data(self):
        """
        Prune AIS data to only include trajectories that are within the AIS buffer.
        """
        search_area = (
            self.slick_gdf.geometry.to_crs(self.crs_meters)
            .buffer(self.ais_buffer)
            .to_crs("4326")
        )

        # Query the spatial index of ais_gdf using the bounds of the search area
        candidate_indices = list(
            self.ais_gdf.sindex.intersection(search_area.total_bounds)
        )

        # Retrieve candidate AIS points
        candidate_points = self.ais_gdf.iloc[candidate_indices]

        # Further filter to ensure actual intersection with the buffered area
        candidate_points = candidate_points[
            candidate_points.geometry.intersects(search_area.iloc[0])
        ]

        # Extract unique ssvid values from the candidate points
        ssvids_of_interest = candidate_points["ssvid"].unique()
        self.ais_filtered = self.ais_gdf[self.ais_gdf["ssvid"].isin(ssvids_of_interest)]

    def score_trajectories(self):
        """
        Measure association by computing multiple metrics between AIS trajectories and slicks

        Returns:
            GeoDataFrame of slick associations
        """
        # print("Scoring trajectories")

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

        centerlines = self.slick_centerlines.sort_values("length", ascending=False)
        longest_centerline = centerlines.to_crs(self.crs_meters).iloc[0]["geometry"]

        # Iterate over filtered trajectories
        for traj in self.ais_trajectories:
            traj_gdf = (
                traj.to_point_gdf()
                .sort_values(by="timestamp", ascending=False)
                .set_crs("4326")
                .to_crs(self.crs_meters)
            )

            slick_to_traj_mapping = self.get_closest_centerline_points(
                traj_gdf, longest_centerline
            )

            temporal_score = compute_temporal_score(
                self.s1_scene.start_time,
                self.ais_ref_time_over,
                self.ais_ref_time_under,
                slick_to_traj_mapping,
            )

            slick_to_traj_mapping = self.get_closest_centerline_points(
                traj_gdf, longest_centerline, self.s1_scene.start_time
            )

            proximity_score = compute_proximity_score(
                traj_gdf,
                self.spread_rate,
                self.s1_scene.start_time,
                slick_to_traj_mapping,
            )
            parity_score = compute_parity_score(
                traj_gdf,
                longest_centerline,
                self.sensitivity_parity,
                slick_to_traj_mapping,
            )

            # Compute total score from these three metrics
            coincidence_score = vessel_compute_total_score(
                temporal_score,
                proximity_score,
                parity_score,
                self.w_temporal,
                self.w_proximity,
                self.w_parity,
            )

            print(
                f"st_name {traj.id}: coincidence_score ({round(coincidence_score, 2)}), "
                f"temporal_score ({round(temporal_score, 2)}), "
                f"proximity_score ({round(proximity_score, 2)}), "
                f"parity_score ({round(parity_score, 2)})"
            )

            entry = {
                "st_name": traj.id,
                "ext_id": str(traj.id),
                "geometry": LineString([p.coords[0] for p in traj.df["geometry"]]),
                "coincidence_score": coincidence_score,
                "type": self.source_type,
                "ext_name": traj.ext_name,
                "ext_shiptype": traj.ext_shiptype,
                "flag": traj.flag,
                "geojson_fc": traj.geojson_fc,
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
    ) -> tuple[pd.Timestamp, Point, float, pd.Timestamp, Point, float]:
        """
        Returns the timestamp and distance of the closest points on the centerline to the vessel at the given image_timestamp.
        """
        # Create centerline endpoints
        cl_A = Point(longest_centerline.coords[0])
        cl_B = Point(longest_centerline.coords[-1])

        # Find nearest trajectory point indices for each endpoint
        traj_idx_A = self.get_closest_point_before_timestamp(cl_A, traj_gdf, t_image)
        traj_idx_B = self.get_closest_point_before_timestamp(cl_B, traj_gdf, t_image)

        # Create tuples for each endpoint: (timestamp, centerline_point, distance)
        ends = [
            (traj_idx_A, cl_A, traj_gdf.loc[traj_idx_A]["geometry"].distance(cl_A)),
            (traj_idx_B, cl_B, traj_gdf.loc[traj_idx_B]["geometry"].distance(cl_B)),
        ]

        # Sort the pairs by timestamp to determine head and tail
        (t_tail, cl_tail, d_tail), (t_head, cl_head, d_head) = sorted(
            ends, key=lambda x: x[0]
        )
        return (t_tail, cl_tail, d_tail, t_head, cl_head, d_head)

    def get_closest_point_before_timestamp(
        self,
        reference_point: Point,
        traj_gdf: gpd.GeoDataFrame,
        t_image: datetime = None,
    ) -> pd.Timestamp:
        """
        Returns the trajectory row that is closest to the reference_point,
        using a turning-point heuristic if an image_timestamp is provided.

        When image_timestamp is None, the function returns the row (from the entire trajectory)
        with the minimum distance to the reference point.

        When image_timestamp is provided, only rows with index <= image_timestamp
        are considered. The function then assumes that the distance from the trajectory
        points to the reference point first decreases and then increases. It returns the last
        row before the first increase in distance. If no such turning point is detected
        (i.e. distances are monotonically decreasing), the function returns the last row
        in the valid time range.

        Parameters:
            reference_point (shapely.geometry.Point): The point to compare distances to.
            traj_gdf (geopandas.GeoDataFrame): A GeoDataFrame with a datetime-like index and a 'geometry' column.
            image_timestamp (datetime, optional): A timestamp to restrict the trajectory.
                Only rows with index <= image_timestamp will be considered.

        Returns:
            pd.Timestamp: The index corresponding to the selected trajectory row.

        """
        # Sort the trajectory by index in descending order, so the latest broadcasted point is first.
        traj_gdf = traj_gdf.sort_index(ascending=False)

        if t_image is None:
            # Find shortest distance, irrespective of any timing
            return traj_gdf.geometry.distance(reference_point).idxmin()

        valid_gdf = traj_gdf.loc[t_image:]
        if valid_gdf.empty:
            # If no points are at or before image_timestamp return the earliest point.
            return traj_gdf.index[-1]

        distances = valid_gdf.geometry.distance(reference_point)

        # Compute differences between consecutive distance values. To identify where the distance increases
        diff_values = distances.diff().iloc[1:]
        increase_mask = diff_values > 0

        if not increase_mask.any():
            # If the distances are monotonically decreasing, return the last row.
            return valid_gdf.index[-1]

        # Find the first occurrence where distance increases.
        return valid_gdf.index[np.argmax(increase_mask.values)]

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Associates AIS trajectories with slicks.
        """
        self.results = gpd.GeoDataFrame()

        self.ais_filtered = None
        self.slick_gdf = slick_gdf
        if self.slick_centerlines is None:
            self.load_slick_centerlines()
        if self.ais_gdf is None:
            self.retrieve_ais_data()
        if self.ais_gdf.empty:
            return pd.DataFrame()
        if self.ais_filtered is None:
            self.filter_ais_data()
        if self.ais_trajectories is None:
            if self.ais_gdf is None:
                self.retrieve_ais_data()
            if self.ais_gdf.empty:
                return pd.DataFrame()
            if self.ais_filtered is None:
                self.filter_ais_data()
            if self.ais_trajectories is None:
                self.build_trajectories()
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
