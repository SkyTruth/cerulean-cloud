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
    MultiPolygon,
    Point,
    Polygon,
    mapping,
)
from shapely.ops import unary_union

from . import constants as c
from .scoring import (
    compute_distance_score,
    compute_overlap_score,
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
        ais_buffered (GeoDataFrame): Buffered geometries of trajectories.
        ais_weighted (list): List of weighted geometries for each trajectory.
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
        self.buf_vec = kwargs.get("buf_vec", c.BUF_VEC)
        self.weight_vec = kwargs.get("weight_vec", c.WEIGHT_VEC)
        self.ais_project_id = kwargs.get("ais_project_id", c.AIS_PROJECT_ID)
        self.w_temporal = kwargs.get("w_temporal", c.VESSEL_W_TEMPORAL)
        self.w_overlap = kwargs.get("w_overlap", c.VESSEL_W_OVERLAP)
        self.w_distance = kwargs.get("w_distance", c.VESSEL_W_DISTANCE)
        self.ais_ref_dist = kwargs.get("ais_ref_dist", c.VESSEL_REF_DIST)
        self.coinc_mean = kwargs.get("coinc_mean", c.VESSEL_MEAN)
        self.coinc_std = kwargs.get("coinc_std", c.VESSEL_STD)

        # Calculated values
        self.ais_start_time = self.s1_scene.start_time - timedelta(
            hours=self.hours_before
        )
        self.ais_end_time = self.s1_scene.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.ais_start_time,
            end=self.s1_scene.start_time,
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
        self.ais_trajectories = None
        self.ais_buffered = None
        self.ais_weighted = None
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
        """
        # print("Building trajectories")
        ais_trajectories = list()
        for st_name, group in self.ais_gdf.groupby("ssvid"):
            # Duplicate the row if there's only one point
            if len(group) == 1:
                group = pd.concat([group] * 2).reset_index(drop=True)

            # Build trajectory
            group["timestamp"] = (
                group["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            )
            traj = mpd.Trajectory(df=group, traj_id=st_name, t="timestamp")

            # Interpolate to times in time_vec
            times = list()
            positions = list()
            for t in self.time_vec:
                pos = traj.interpolate_position_at(t)
                times.append(t)
                positions.append(pos)
            gdf = gpd.GeoDataFrame(
                {"timestamp": times, "geometry": positions}, crs="4326"
            )

            # Store as trajectory
            interpolated_traj = mpd.Trajectory(
                gdf,
                traj_id=st_name,
                t="timestamp",
            )
            gdf["timestamp"] = gdf["timestamp"].apply(lambda x: x.isoformat())
            interpolated_traj.ext_name = group.iloc[0]["shipname"]
            interpolated_traj.ext_shiptype = group.iloc[0]["best_shiptype"]
            interpolated_traj.flag = group.iloc[0]["flag"]

            # Calculate display feature collection
            s1_time = pd.Timestamp(times[-1])
            display_gdf = group[group["timestamp"] <= s1_time].copy()
            display_gdf["timestamp"] = display_gdf["timestamp"].apply(
                lambda x: x.isoformat()
            )
            if group["timestamp"].iloc[-1] > s1_time:
                display_gdf = pd.concat(
                    [display_gdf, gdf.iloc[[-1]]], ignore_index=True
                )

            interpolated_traj.geojson_fc = {
                "type": "FeatureCollection",
                "features": json.loads(display_gdf.to_json())["features"],
            }

            ais_trajectories.append(interpolated_traj)

        self.ais_trajectories = mpd.TrajectoryCollection(ais_trajectories)

    def buffer_trajectories(self):
        """
        Buffers trajectories.
        """
        # print("Buffering trajectories")
        ais_buf = list()
        ais_weighted = list()
        for traj in self.ais_trajectories:
            # Grab points
            points = (
                traj.to_point_gdf()
                .sort_values(by="timestamp", ascending=False)
                .to_crs(self.crs_meters)
                .reset_index()
            )

            # Create buffered circles at points
            ps = (
                points.apply(
                    lambda row: row.geometry.buffer(self.buf_vec[row.name]), axis=1
                )
                .set_crs(self.crs_meters)
                .to_crs("4326")
            )

            # Create convex hulls from sequential circles
            convex_hulls = [
                MultiPolygon([a, b]).convex_hull for a, b in zip(ps[:-1], ps[1:])
            ]

            # Weight convex hulls
            weighted = gpd.GeoDataFrame(
                {"geometry": convex_hulls, "weight": self.weight_vec[:-1]},
                crs="4326",
            )
            ais_weighted.append(weighted)

            # Create connected polygon from hulls
            ais_buf.append({"geometry": unary_union(convex_hulls), "st_name": traj.id})

        self.ais_buffered = gpd.GeoDataFrame(ais_buf, crs="4326")
        self.ais_weighted = ais_weighted

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

        # Create a GeoDataFrame of buffered trajectories
        buffered_trajectories_gdf = self.ais_buffered.copy()
        buffered_trajectories_gdf["id"] = [t.id for t in self.ais_trajectories]
        buffered_trajectories_gdf.set_index("id", inplace=True)

        # Perform a spatial join between buffered trajectories and slick geometries
        matches = gpd.sjoin(
            buffered_trajectories_gdf,
            self.slick_gdf,
            how="inner",
            predicate="intersects",
        )

        if matches.empty:
            print("No trajectories intersect the slicks.")
            self.results = gpd.GeoDataFrame(columns=columns, crs="4326")
            return self.results

        # Get unique trajectory IDs that intersect slicks
        intersecting_traj_ids = matches.index.unique()

        # Filter trajectories and weights based on intersecting IDs
        ais_filt = [t for t in self.ais_trajectories if t.id in intersecting_traj_ids]
        weighted_filt = [
            self.ais_weighted[idx]
            for idx, t in enumerate(self.ais_trajectories)
            if t.id in intersecting_traj_ids
        ]
        buffered_filt = [buffered_trajectories_gdf.loc[[t.id]] for t in ais_filt]

        entries = []
        # Skip the loop if weighted_filt is empty
        if weighted_filt:
            # Create a trajectory collection from filtered trajectories
            ais_filt = mpd.TrajectoryCollection(ais_filt)

            # Iterate over filtered trajectories
            for t, w, b in zip(ais_filt, weighted_filt, buffered_filt):
                # Compute temporal score
                temporal_score = compute_temporal_score(w, self.slick_gdf)

                # Compute overlap score
                overlap_score = compute_overlap_score(
                    b, self.slick_gdf, self.crs_meters
                )

                # Compute distance score between trajectory and slick centerline
                distance_score = compute_distance_score(
                    t, self.slick_centerlines, self.crs_meters, self.ais_ref_dist
                )

                # Compute total score from these three metrics
                coincidence_score = vessel_compute_total_score(
                    temporal_score,
                    overlap_score,
                    distance_score,
                    self.w_temporal,
                    self.w_overlap,
                    self.w_distance,
                )

                print(
                    f"st_name {t.id}: coincidence_score ({round(coincidence_score, 2)}) = "
                    f"({self.w_overlap} * overlap_score ({round(overlap_score, 2)}) + "
                    f"{self.w_temporal} * temporal_score ({round(temporal_score, 2)}) + "
                    f"{self.w_distance} * distance_score ({round(distance_score, 2)})) / "
                    f"({self.w_overlap + self.w_temporal + self.w_distance})"
                )

                entry = {
                    "st_name": t.id,
                    "ext_id": str(t.id),
                    "geometry": LineString([p.coords[0] for p in t.df["geometry"]]),
                    "coincidence_score": coincidence_score,
                    "type": self.source_type,
                    "ext_name": t.ext_name,
                    "ext_shiptype": t.ext_shiptype,
                    "flag": t.flag,
                    "geojson_fc": t.geojson_fc,
                }
                entries.append(entry)

        sources = gpd.GeoDataFrame(entries, columns=columns, crs="4326")
        self.results = sources[sources["coincidence_score"] > 0]
        return self.results

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Associates AIS trajectories with slicks.
        """
        self.results = gpd.GeoDataFrame()

        self.slick_gdf = slick_gdf
        if self.slick_centerlines is None:
            self.load_slick_centerlines()
        if self.ais_gdf is None:
            self.retrieve_ais_data()
        if self.ais_gdf.empty:
            return pd.DataFrame()
        if self.ais_trajectories is None:
            self.build_trajectories()
        if self.ais_buffered is None:
            self.buffer_trajectories()
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

    def process_slicks(self, slick_gdf: gpd.GeoDataFrame):
        """
        Processes slicks and returns combined geometry, overall centroid, polygons, and largest polygon area.
        """
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

    def calc_score_extremities_to_points(
        self,
        points_gdf: gpd.GeoDataFrame,
        extremity_tree: cKDTree,
        all_extremity_points: np.ndarray,
        all_weights: np.ndarray,
        center_points: np.ndarray = None,
    ) -> np.ndarray:
        """
        Computes confidence scores for points based on their proximity to extremity points.
        """
        points_coords = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])
        extremity_indices = extremity_tree.query_ball_point(
            points_coords, r=self.cutoff_radius
        )
        coincidence_scores = np.zeros(len(points_coords))

        for i, neighbors in enumerate(extremity_indices):
            if neighbors:
                neighbor_weights = all_weights[neighbors]
                neighbor_points = all_extremity_points[neighbors]
                center_points_reduced = center_points[neighbors]
                if self.decay_theta > 0 and center_points is not None:
                    scaled_angles = self.scaled_inner_angles(
                        points_coords[i], center_points_reduced, neighbor_points
                    )
                else:
                    scaled_angles = np.zeros(len(neighbor_points))
                dists = np.linalg.norm(neighbor_points - points_coords[i], axis=1)

                C_i = (
                    neighbor_weights
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

    def get_both_endpoints_with_secondary(
        self, line: LineString, gap: float, offset: float
    ):
        """
        Select endpoints from a line from a specified offset and gap in meters
        """
        total_length = line.length

        if gap + offset > total_length:
            raise ValueError("The given distance exceeds the total length of the line.")

        # For the start of the line:
        start_point = line.interpolate(offset)  # Start point is the first point.
        secondary_start_point = line.interpolate(
            offset + gap
        )  # Move forward from the start.

        # For the end of the line:
        end_point = line.interpolate(total_length - offset)  # End point of the line.
        secondary_end_point = line.interpolate(
            total_length - offset - gap
        )  # Move backward from the end.

        return ((start_point, secondary_start_point), (end_point, secondary_end_point))

    def select_endpoints_from_centerlines(
        self, lines, center_point, offset=0.05, gap=0.05
    ):
        """
        Given a list of LineString geometries and a center point, this function:
        - Extracts endpoints (extrema) from each LineString.
        - Extracts associated delta points (using the second and second-to-last vertices).
        - Selects the extrema point farthest from the center_point.
        - Then selects, from all extrema, the point farthest from that first selected extrema.
        Returns:
        selected_extrema: a numpy array containing the selected endpoints.
        selected_delta: a numpy array containing the corresponding delta points.
        weights: a numpy array of ones.

        Parameters:
        lines (list): List of LineString geometries.
        center_point (np.ndarray or shapely Point): The center point (if a shapely Point, it will be converted to np.array).
        """
        # If center_point is a shapely Point, convert it to a NumPy array.
        if isinstance(center_point, Point):
            center_point = np.array([center_point.x, center_point.y])

        extrema_points = []
        delta_points = []
        length_fractions = []

        for line in lines:
            coords = list(line.coords)
            if len(coords) < (offset + gap):
                continue

            start, end = self.get_both_endpoints_with_secondary(
                line, gap=gap * line.length, offset=offset * line.length
            )

            extrema_points.append(start[0].coords[0])
            extrema_points.append(end[0].coords[0])
            length_fractions.append(line.length)  # Add weights for front extrema
            length_fractions.append(line.length)  # Add weights for back extrema

            # If there are enough vertices, grab the “delta” points.
            # (Here we use the second vertex and the second-to-last vertex.)

            delta_points.append(np.array(start[1].coords[0]))
            delta_points.append(np.array(end[1].coords[0]))

        # Convert lists to numpy arrays.
        length_fractions = np.array(length_fractions)
        length_fractions = length_fractions / max(length_fractions)

        extrema_points = np.array(extrema_points)

        # Only keep delta_points if we have one per extrema.
        if len(delta_points) == len(extrema_points):
            delta_points = np.array(delta_points)
        else:
            delta_points = None

        all_extrema = np.vstack(extrema_points)
        all_length_fractions = np.array(length_fractions)

        # Calculate distances from centroid
        distances_sq = np.sum((all_extrema - center_point) ** 2, axis=1)
        # Scale weights by area fraction
        scaled_weights = distances_sq * all_length_fractions
        # Normalize weights to ensure the maximum weight is 1
        max_weight = scaled_weights.max()
        if max_weight != 0:
            all_weights = scaled_weights / max_weight
        else:
            all_weights = np.ones_like(scaled_weights)

        return extrema_points, delta_points, all_weights


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
            df = df[df["label"] == "oil"]
        df["st_name"] = df["structure_id"].apply(str)
        df["ext_id"] = df["structure_id"].apply(str)
        df["type"] = self.source_type

        datetime_fields = ["structure_start_date", "structure_end_date"]
        for field in datetime_fields:
            if field in df.columns:
                df.loc[df[field] == "", field] = str(int(time.time() * 1000))
                df[field] = pd.to_numeric(df[field], errors="coerce")
                df[field] = pd.to_datetime(df[field], unit="ms", errors="coerce")
        df.reset_index(drop=True, inplace=True)
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

        combined_geometry, polygons, largest_polygon_area = self.process_slicks(
            self.slick_gdf
        )
        filtered_infra = self.filter_points(combined_geometry, self.infra_gdf)

        if filtered_infra.empty:
            print(
                "No infrastructure within the dates / radius of interest. No coincidence scores edited."
            )
            return

        # Collect extremity points and compute weights
        all_extrema, all_weights = self.aggregate_extrema_and_area_fractions(
            polygons, combined_geometry, largest_polygon_area
        )

        point = np.array(combined_geometry.centroid.coords[0])
        delta_points = np.tile(point, (len(all_extrema), 1))

        # Build KD-Tree and compute confidence scores
        extremity_tree = cKDTree(all_extrema)
        coincidence_filtered = self.calc_score_extremities_to_points(
            filtered_infra,
            extremity_tree,
            all_extrema,
            all_weights,
            # XXX HACK OVERALL_CENTROID -- should remove when we switch to using spines
            delta_points,
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

        combined_geometry, polygons, largest_polygon_area = self.process_slicks(
            self.slick_gdf
        )

        filtered_dark_objects = self.filter_points(
            combined_geometry, self.dark_objects_gdf
        )

        if filtered_dark_objects.empty:
            print(
                "No dark objects within the radius of interest. No coincidence scores edited."
            )
            return

        # Collect extremity points and compute weights
        if self.slick_centerlines is None:
            self.load_slick_centerlines()

        (
            all_extrema,
            delta_points,
            all_weights,
        ) = self.select_endpoints_from_centerlines(
            self.slick_centerlines.to_crs(self.crs_meters).geometry.values,
            np.array([combined_geometry.centroid.coords[0]]),
            offset=self.endpoints_offset,
            gap=self.endpoints_gap,
        )

        # Build KD-Tree and compute confidence scores
        extremity_tree = cKDTree(all_extrema)
        coincidence_filtered = self.calc_score_extremities_to_points(
            filtered_dark_objects,
            extremity_tree,
            all_extrema,
            all_weights,
            delta_points,
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
