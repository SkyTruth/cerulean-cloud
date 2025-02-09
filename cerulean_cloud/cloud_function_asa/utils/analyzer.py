"""
Unified Source Analysis Module
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import centerline.geometry
import geopandas as gpd
import mapbox_vector_tile
import morecantile
import movingpandas as mpd
import numpy as np
import pandas as pd
import pandas_gbq
import requests
import scipy.interpolate
import scipy.spatial.distance
import shapely
from geoalchemy2.shape import to_shape
from google.oauth2.service_account import Credentials
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping

from .constants import (
    AIS_BUFFER,
    AIS_PROJECT_ID,
    AIS_REF_DIST,
    BUF_VEC,
    CLOSING_BUFFER,
    D_FORMAT,
    DARK_MEAN,
    DARK_REF_DIST,
    DARK_STD,
    DECAY_FACTOR,
    HOURS_AFTER,
    HOURS_BEFORE,
    INFRA_MEAN,
    INFRA_REF_DIST,
    INFRA_STD,
    MIN_AREA_THRESHOLD,
    NUM_TIMESTEPS,
    NUM_VERTICES,
    T_FORMAT,
    THETA_DECAY,
    VESSEL_MEAN,
    VESSEL_STD,
    W_ARF_DARK,
    W_ARF_VESSEL,
    W_DISTANCE,
    W_OVERLAP,
    W_TEMPORAL,
    WEIGHT_VEC,
)
from .scoring import (
    compute_aspect_ratio_factor,
    compute_distance_score,
    compute_overlap_score,
    compute_temporal_score,
    compute_total_score,
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

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Placeholder method to be overridden
        """
        pass

    def apply_closing_buffer(self, geo_df: gpd.GeoDataFrame, closing_buffer: float):
        """
        Applies a closing buffer to geometries in the GeoDataFrame.
        """
        geo_df["geometry"] = (
            geo_df["geometry"].buffer(closing_buffer).buffer(-closing_buffer)
        )
        return geo_df


class InfrastructureAnalyzer(SourceAnalyzer):
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
        self.num_vertices = kwargs.get("num_vertices", NUM_VERTICES)
        self.closing_buffer = kwargs.get("closing_buffer", CLOSING_BUFFER)
        self.cutoff_radius = kwargs.get("cutoff_radius", INFRA_REF_DIST)
        self.decay_factor = kwargs.get("decay_factor", DECAY_FACTOR)
        self.min_area_threshold = kwargs.get("min_area_threshold", MIN_AREA_THRESHOLD)
        self.coinc_mean = kwargs.get("coinc_mean", INFRA_MEAN)
        self.coinc_std = kwargs.get("coinc_std", INFRA_STD)

        self.infra_gdf = kwargs.get("infra_gdf", None)

        if self.infra_gdf is None:
            self.infra_api_token = os.getenv("INFRA_API_TOKEN")
            self.infra_gdf = self.load_infrastructure_data_api()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))

    def load_infrastructure_data_csv(self, only_oil=True):
        """
        Loads infrastructure data from a CSV file.
        """
        df = pd.read_csv("SAR Fixed Infrastructure 202407 DENOISED UNIQUE.csv")
        df["st_name"] = df["structure_id"].apply(str)
        df["ext_id"] = df["structure_id"].apply(str)
        df["type"] = 2  # infra
        if only_oil:
            df = df[df["label"] == "oil"]

        # This code isn't working because of a bug in how GFW records the first and last date.
        # df["structure_start_date"] = pd.to_datetime(df["structure_start_date"])

        # df.loc[df["structure_end_date"].isna(), "structure_end_date"] = (
        #     datetime.now().strftime("%Y-%m-%d")
        # )
        # df["structure_end_date"] = pd.to_datetime(df["structure_end_date"])

        df.reset_index(drop=True, inplace=True)
        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="epsg:4326"
        )

    def load_infrastructure_data_api(self, only_oil=True):
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
        df["type"] = 2  # infra

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

    def extract_polygons(self, geometry):
        """
        Extracts individual polygons from a geometry.
        """
        return (
            [geom for geom in geometry.geoms if isinstance(geom, Polygon)]
            if isinstance(geometry, (MultiPolygon, GeometryCollection))
            else [geometry]
        )

    def select_extreme_points(
        self, polygon: Polygon, num_vertices: int, reference_points: List[np.ndarray]
    ) -> np.ndarray:
        """
        Selects N extremity points from the polygon based on their distance from reference points.
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

    def collect_extremity_points(
        self,
        polygons: List[Polygon],
        num_vertices: int,
        overall_centroid: np.ndarray,
        largest_polygon_area: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collects extremity points and their scaled area fractions from all polygons.
        """
        extremity_points_list = []
        area_fractions_list = []

        for polygon in polygons:
            if polygon.area < self.min_area_threshold * largest_polygon_area:
                continue  # Skip small polygons

            # Select N extremity points for the current polygon
            selected_points = self.select_extreme_points(
                polygon, num_vertices, [overall_centroid]
            )
            extremity_points_list.append(selected_points)

            # Compute scaled area fraction for weighting
            area_fraction = polygon.area / largest_polygon_area
            scaled_area_fraction = np.sqrt(
                area_fraction
            )  # More sensitive to small areas
            area_fractions_list.extend([scaled_area_fraction] * num_vertices)

        if not extremity_points_list:
            raise ValueError("No extremity points collected from polygons.")

        all_extremity_points = np.vstack(extremity_points_list)
        all_area_fractions = np.array(area_fractions_list)

        return all_extremity_points, all_area_fractions

    def compute_weights(
        self, all_extremity_points, overall_centroid, all_area_fractions
    ):
        """
        Computes normalized weights based on distances from the centroid and area fractions.
        """
        distances_sq = np.sum((all_extremity_points - overall_centroid) ** 2, axis=1)
        scaled_weights = distances_sq * all_area_fractions

        max_weight = scaled_weights.max()

        return (
            scaled_weights / max_weight
            if max_weight != 0
            else np.ones_like(scaled_weights)
        )

    def compute_coincidence_scores_for_infra(
        self,
        infra_gdf: gpd.GeoDataFrame,
        extremity_tree: cKDTree,
        all_extremity_points: np.ndarray,
        all_weights: np.ndarray,
        cutoff_radius: float,
        decay_factor: float,
    ) -> np.ndarray:
        """
        Computes confidence scores for infrastructure points based on proximity to extremity points.
        """
        infra_coords = np.array([(geom.x, geom.y) for geom in infra_gdf.geometry])
        extremity_indices = extremity_tree.query_ball_point(
            infra_coords, r=cutoff_radius
        )
        coincidence_scores = np.zeros(len(infra_coords))

        for i, neighbors in enumerate(extremity_indices):
            if neighbors:
                neighbor_points = all_extremity_points[neighbors]
                neighbor_weights = all_weights[neighbors]
                dists = np.linalg.norm(neighbor_points - infra_coords[i], axis=1)
                C_i = neighbor_weights * np.exp(-decay_factor * dists / cutoff_radius)
                coincidence_scores[i] = np.clip(C_i.max(), 0, 1)

        return coincidence_scores

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Computes coincidence scores for infrastructure points.
        """
        start_time = time.time()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))
        self.slick_gdf = slick_gdf

        slick_gdf = self.slick_gdf.to_crs(self.crs_meters)
        infra_gdf = self.infra_gdf.to_crs(self.crs_meters)

        # Generate closed polygons
        slick_gdf = self.apply_closing_buffer(slick_gdf, self.closing_buffer)
        combined_geometry = slick_gdf.unary_union
        polygons = self.extract_polygons(combined_geometry)

        # Filter based on scene date and radius of interest
        # scene_date = pd.to_datetime(self.s1_scene.scene_id[17:25], format="%Y%m%d") # XXX Not working because of a bug in how GFW records the first and last date.
        slick_buffered = combined_geometry.buffer(self.cutoff_radius)
        filtered_infra = infra_gdf[
            infra_gdf.geometry.within(slick_buffered)
            # & (infra_gdf["structure_start_date"] < scene_date)
            # & (infra_gdf["structure_end_date"] > scene_date)
        ]

        if filtered_infra.empty:
            print(
                "No infrastructure within the dates / radius of interest."
                "No coincidence scores edited."
            )
            return

        # Compute largest area and overall centroid
        largest_polygon_area = max(polygon.area for polygon in polygons)
        overall_centroid = np.array(combined_geometry.centroid.coords[0])

        # Collect extremity points and compute weights
        all_extremity_points, all_area_fractions = self.collect_extremity_points(
            polygons, self.num_vertices, overall_centroid, largest_polygon_area
        )
        all_weights = self.compute_weights(
            all_extremity_points, overall_centroid, all_area_fractions
        )

        # Build KD-Tree and compute confidence scores
        extremity_tree = cKDTree(all_extremity_points)
        confidence_filtered = self.compute_coincidence_scores_for_infra(
            filtered_infra,
            extremity_tree,
            all_extremity_points,
            all_weights,
            self.cutoff_radius,
            self.decay_factor,
        )

        self.coincidence_scores[filtered_infra.index] = confidence_filtered
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")

        # Return a DataFrame with infra_gdf and coincidence_scores
        self.results = self.infra_gdf.copy()
        self.results["coincidence_score"] = self.coincidence_scores

        self.results["geojson_fc"] = self.infra_gdf["geometry"].apply(
            lambda geom: {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "0",
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {},  # XXX Add properties as they become available (like first/last date)
                    }
                ],
            }
        )
        self.results = self.results[self.results["coincidence_score"] > 0]
        self.results["collated_score"] = (
            self.results["coincidence_score"] - self.coinc_mean
        ) / self.coinc_std
        return self.results


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
        self.hours_before = kwargs.get("hours_before", HOURS_BEFORE)
        self.hours_after = kwargs.get("hours_after", HOURS_AFTER)
        self.ais_buffer = kwargs.get("ais_buffer", AIS_BUFFER)
        self.num_timesteps = kwargs.get("num_timesteps", NUM_TIMESTEPS)
        self.buf_vec = kwargs.get("buf_vec", BUF_VEC)
        self.weight_vec = kwargs.get("weight_vec", WEIGHT_VEC)
        self.ais_project_id = kwargs.get("ais_project_id", AIS_PROJECT_ID)
        self.w_temporal = kwargs.get("w_temporal", W_TEMPORAL)
        self.w_overlap = kwargs.get("w_overlap", W_OVERLAP)
        self.w_distance = kwargs.get("w_distance", W_DISTANCE)
        self.w_aspect_ratio_factor = kwargs.get("w_aspect_ratio_factor", W_ARF_VESSEL)
        self.ais_ref_dist = kwargs.get("ais_ref_dist", AIS_REF_DIST)
        self.coinc_mean = kwargs.get("coinc_mean", VESSEL_MEAN)
        self.coinc_std = kwargs.get("coinc_std", VESSEL_STD)

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
        self.slick_curves = None
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
                seg._PARTITIONTIME between '{datetime.strftime(self.ais_start_time, D_FORMAT)}' AND '{datetime.strftime(self.ais_end_time, D_FORMAT)}'
                AND seg.timestamp between '{datetime.strftime(self.ais_start_time, T_FORMAT)}' AND '{datetime.strftime(self.ais_end_time, T_FORMAT)}'
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{self.ais_envelope[0]}'))
            """
        df = pandas_gbq.read_gbq(
            sql,
            project_id=self.ais_project_id,
            credentials=self.credentials,
        )
        df["geometry"] = df.apply(
            lambda row: shapely.geometry.Point(row["lon"], row["lat"]), axis=1
        )
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
                shapely.geometry.MultiPolygon([a, b]).convex_hull
                for a, b in zip(ps[:-1], ps[1:])
            ]

            # Weight convex hulls
            weighted = gpd.GeoDataFrame(
                {"geometry": convex_hulls, "weight": self.weight_vec[:-1]},
                crs="4326",
            )
            ais_weighted.append(weighted)

            # Create connected polygon from hulls
            ais_buf.append(
                {"geometry": shapely.ops.unary_union(convex_hulls), "st_name": traj.id}
            )

        self.ais_buffered = gpd.GeoDataFrame(ais_buf, crs="4326")
        self.ais_weighted = ais_weighted

    def slick_to_curves(
        self,
        buf_size: int = 2000,
        smoothing_factor: float = 1e10,
    ):
        """
        From a set of oil slick detections, estimate curves that go through the detections
        This process transforms a set of slick detections into LineStrings for each detection

        Inputs:
            buf_size: buffer size for cleaning up slick detections
            smoothing_factor: smoothing factor for smoothing centerline
        Returns:
            GeoDataFrame of slick curves
        """
        # print("Creating slick curves")
        # clean up the slick detections by dilation followed by erosion
        # this process can merge some polygons but not others, depending on proximity
        slick_clean: gpd.GeoDataFrame = self.slick_gdf.copy()
        slick_clean = self.apply_closing_buffer(
            slick_clean.to_crs(self.crs_meters), buf_size
        )

        # split slicks into individual polygons
        slick_clean = slick_clean.explode(ignore_index=True, index_parts=False)

        # find a centerline through detections
        slick_curves = list()
        for _, row in slick_clean.iterrows():
            # create centerline -> MultiLineString
            polygon_perimeter = row.geometry.length  # Perimeter of the polygon
            interp_dist = min(
                100, polygon_perimeter / 1000
            )  # Use a minimum of 1000 points for voronoi calculation
            cl = centerline.geometry.Centerline(
                row.geometry, interpolation_distance=interp_dist
            )

            # grab coordinates from centerline
            x = list()
            y = list()
            if isinstance(cl.geometry, shapely.geometry.MultiLineString):
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
                    x_fit_sort_x,
                    y_fit_sort_x,
                    k=3,
                    s=smoothing_factor,
                    full_output=True,
                )
                tck_sort_y, fp_sort_y, _, _ = scipy.interpolate.splrep(
                    y_fit_sort_y,
                    x_fit_sort_y,
                    k=3,
                    s=smoothing_factor,
                    full_output=True,
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

        slick_clean["areas"] = slick_clean.geometry.area
        self.slick_clean = slick_clean.to_crs(self.slick_gdf.crs)
        slick_curves_gdf = gpd.GeoDataFrame(geometry=slick_curves, crs=self.crs_meters)
        slick_curves_gdf["length"] = slick_curves_gdf.geometry.length
        slick_curves_gdf = slick_curves_gdf.to_crs("4326")

        self.slick_curves = slick_curves_gdf

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

                # Compute distance score between trajectory and slick curve
                distance_score = compute_distance_score(
                    t, self.slick_curves, self.crs_meters, self.ais_ref_dist
                )

                aspect_ratio_factor_score = compute_aspect_ratio_factor(
                    self.slick_curves, self.slick_clean
                )

                # Compute total score from these three metrics
                coincidence_score = compute_total_score(
                    temporal_score,
                    overlap_score,
                    distance_score,
                    aspect_ratio_factor_score,
                    self.w_temporal,
                    self.w_overlap,
                    self.w_distance,
                    self.w_aspect_ratio_factor,
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
                    "geometry": shapely.geometry.LineString(
                        [p.coords[0] for p in t.df["geometry"]]
                    ),
                    "coincidence_score": coincidence_score,
                    "type": 1,  # Vessel
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

        self.slick_curves = None
        self.slick_gdf = slick_gdf

        if self.ais_gdf is None:
            self.retrieve_ais_data()
        if self.ais_gdf.empty:
            return pd.DataFrame()

        self.slick_to_curves()
        if self.ais_trajectories is None:
            self.build_trajectories()
        if self.ais_buffered is None:
            self.buffer_trajectories()
        self.score_trajectories()
        self.results["collated_score"] = (
            self.results["coincidence_score"] - self.coinc_mean
        ) / self.coinc_std
        return self.results


class DarkAnalyzer(InfrastructureAnalyzer):
    """
    Analyzer for dark vessels (non-AIS broadcasting vessels).
    Currently inherits functionality from Infrastructure Analyzer.
    """

    def __init__(self, s1_scene, **kwargs):
        """
        Initialize the DarkAnalyzer.
        """

        self.credentials = Credentials.from_service_account_info(
            json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        )
        self.gfw_project_id = "world-fishing-827"
        self.s1_scene = s1_scene
        self.infra_gdf = None
        self.dark_vessels_gdf = kwargs.get("dark_vessels_gdf", None)
        if self.dark_vessels_gdf is None:
            self.retrieve_sar_detections()
        self.infra_gdf = self.dark_vessels_gdf
        kwargs["infra_gdf"] = self.infra_gdf

        super().__init__(s1_scene, **kwargs)
        self.source_type = 3
        self.theta_decay = kwargs.get("theta_decay", THETA_DECAY)
        self.coinc_mean = kwargs.get("coinc_mean", DARK_MEAN)
        self.coinc_std = kwargs.get("coinc_std", DARK_STD)
        self.cutoff_radius = kwargs.get("cutoff_radius", DARK_REF_DIST)
        self.w_aspect_ratio_factor = kwargs.get("w_aspect_ratio_factor", W_ARF_DARK)

    def scaled_inner_angles(self, a, b, c_set):
        """
        Calculate scaled inner angles at vertex A for a triangle formed by points A, B, and each point in C.

        Parameters:
            a: Tuple representing the vertex A (x1, y1).
            b: Tuple representing the vertex B (x2, y2).
            c_set: List or array of tuples representing the points in C (x3, y3).

        Returns:
            A NumPy array of scaled angles at vertex A for each point in C.
        """
        # Convert A and B to numpy arrays
        a = np.array(a)
        b = np.array(b)

        # Convert C set to a NumPy array
        c_set = np.array(c_set)

        # Vectors AB and AC
        ab = b - a  # Vector AB (constant for all C points)
        ac = c_set - a  # Vector AC for each point in C

        # Dot products and magnitudes
        dot_products = np.sum(ac * ab, axis=1)  # Dot product of AB and each AC
        magnitude_ab = np.linalg.norm(ab)  # Magnitude of AB (constant)
        magnitudes_ac = np.linalg.norm(ac, axis=1)  # Magnitudes of all AC vectors

        # Calculate angles in radians
        angles_radians = np.arccos(dot_products / (magnitude_ab * magnitudes_ac))

        # Scale angles to range 0-1
        scaled_angles = angles_radians / np.pi

        return np.array(scaled_angles)

    def retrieve_sar_detections(self):
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
        ),

        -- Step 3: Optimize the unique_infra CTE with pre-filtering using a bounding box
        unique_infra AS (
        SELECT
            infra.*,
            ROW_NUMBER() OVER (
            PARTITION BY infra.structure_id
            ORDER BY infra.label_confidence DESC  -- Assuming you want the highest confidence
            ) AS rn
        FROM
            `world-fishing-827.pipe_sar_v1_published.published_infrastructure` AS infra
        INNER JOIN filtered_matches AS match
            -- Define a rough bounding box around detection points to limit infra records
            ON ABS(infra.lat - match.detect_lat) < 0.001  -- Approx ~100 meters latitude
            AND ABS(infra.lon - match.detect_lon) < 0.001  -- Approx ~100 meters longitude
        )

        -- Step 4: Final SELECT with optimized joins and distance calculation
        SELECT
        match.scene_id,
        match.ssvid,
        infra.structure_id,
        match.detect_id,
        pred.presence AS dt_pres_score,
        match.detect_lat,
        match.detect_lon,
        match.score AS match_score,
        pred.length_m,
        infra.label AS infra_label,
        infra.label_confidence AS infra_label_confidence
        FROM
        `world-fishing-827.pipe_sar_v1_published.detect_scene_pred` AS pred
        INNER JOIN
        filtered_matches AS match
            ON pred.detect_id = match.detect_id
        LEFT JOIN
        unique_infra AS infra
            ON infra.rn = 1
            AND ST_DISTANCE(
                ST_GEOGPOINT(match.detect_lon, match.detect_lat),
                ST_GEOGPOINT(infra.lon, infra.lat)
                ) < 100  -- Distance in meters
        """

        df = pandas_gbq.read_gbq(
            sql,
            project_id=self.gfw_project_id,
            credentials=self.credentials,
        )

        df = df[df["structure_id"].isna()]
        df = df[df["ssvid"].isna()]

        df["geometry"] = df.apply(
            lambda row: shapely.geometry.Point(row["detect_lon"], row["detect_lat"]),
            axis=1,
        )

        self.dark_vessels_gdf = gpd.GeoDataFrame(df, crs="4326").reset_index(drop=True)

    def decayed_points_to_poly_score(
        self,
        infra_gdf: gpd.GeoDataFrame,
        extremity_tree: cKDTree,
        all_extremity_points: np.ndarray,
        all_weights: np.ndarray,
        cutoff_radius: float,
        decay_factor: float,
        center_point: np.ndarray,
        theta_decay: float,
    ) -> np.ndarray:
        """
        Computes confidence scores for infrastructure points based on proximity to extremity points.
        """
        self.all_extremity_points = all_extremity_points
        infra_coords = np.array([(geom.x, geom.y) for geom in infra_gdf.geometry])
        extremity_indices = extremity_tree.query_ball_point(
            infra_coords, r=cutoff_radius
        )
        coincidence_scores = np.zeros(len(infra_coords))

        for i, neighbors in enumerate(extremity_indices):
            if neighbors:
                neighbor_points = all_extremity_points[neighbors]

                scaled_angles = self.scaled_inner_angles(
                    infra_coords[i], center_point, neighbor_points
                )
                neighbor_weights = all_weights[neighbors]
                dists = np.linalg.norm(neighbor_points - infra_coords[i], axis=1)

                C_i = (
                    np.exp(-theta_decay * scaled_angles)
                    * neighbor_weights
                    * np.exp(-decay_factor * dists / cutoff_radius)
                )

                coincidence_scores[i] = np.clip(C_i.max(), 0, 1)

        return coincidence_scores

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Computes coincidence scores for infrastructure points.
        """
        start_time = time.time()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))
        self.slick_gdf = slick_gdf

        slick_gdf = self.slick_gdf.to_crs(self.crs_meters)
        infra_gdf = self.infra_gdf.to_crs(self.crs_meters)

        # Generate closed polygons
        slick_gdf = self.apply_closing_buffer(slick_gdf, self.closing_buffer)
        combined_geometry = slick_gdf.unary_union
        polygons = self.extract_polygons(combined_geometry)

        # Filter based on scene date and radius of interest
        # scene_date = pd.to_datetime(self.s1_scene.scene_id[17:25], format="%Y%m%d") # XXX Not working because of a bug in how GFW records the first and last date.
        slick_buffered = combined_geometry.buffer(self.cutoff_radius)
        filtered_infra = infra_gdf[
            infra_gdf.geometry.within(slick_buffered)
            # & (infra_gdf["structure_start_date"] < scene_date)
            # & (infra_gdf["structure_end_date"] > scene_date)
        ]

        if filtered_infra.empty:
            print(
                "No dark vessels within the dates / radius of interest."
                "No coincidence scores edited."
            )
            return

        # Compute largest area and overall centroid
        largest_polygon_area = max(polygon.area for polygon in polygons)
        overall_centroid = np.array(combined_geometry.centroid.coords[0])
        # self.overall_centroid = overall_centroid
        # Collect extremity points and compute weights
        all_extremity_points, all_area_fractions = self.collect_extremity_points(
            polygons, self.num_vertices, overall_centroid, largest_polygon_area
        )
        all_weights = self.compute_weights(
            all_extremity_points, overall_centroid, all_area_fractions
        )

        # Build KD-Tree and compute confidence scores
        extremity_tree = cKDTree(all_extremity_points)
        confidence_filtered = self.decayed_points_to_poly_score(
            filtered_infra,
            extremity_tree,
            all_extremity_points,
            all_weights,
            self.cutoff_radius,
            self.decay_factor,
            overall_centroid,
            self.theta_decay,
        )

        self.coincidence_scores[filtered_infra.index] = confidence_filtered
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")

        # Return a DataFrame with infra_gdf and coincidence_scores
        self.results = self.infra_gdf.copy()
        self.results["coincidence_score"] = self.coincidence_scores

        self.results["geojson_fc"] = self.infra_gdf["geometry"].apply(
            lambda geom: {
                "type": "FeatureCollection",
                "features": [
                    {
                        "id": "0",
                        "type": "Feature",
                        "geometry": mapping(geom),
                        "properties": {},  # XXX Add properties as they become available (like first/last date)
                    }
                ],
            }
        )

        aspect_ratio_factor = compute_aspect_ratio_factor(
            self.slick_curves, self.slick_clean
        )

        self.results = self.results[self.results["coincidence_score"] > 0]

        self.results["coincidence_score"] = (
            self.w_aspect_ratio_factor * aspect_ratio_factor
            + self.results["coincidence_score"]
        ) / (1 + self.w_aspect_ratio_factor)

        self.results["collated_score"] = (
            self.results["coincidence_score"] - self.coinc_mean
        ) / self.coinc_std
        return self.results

    def slick_to_curves(
        self,
        buf_size: int = 2000,
        smoothing_factor: float = 1e10,
    ):
        """
        From a set of oil slick detections, estimate curves that go through the detections
        This process transforms a set of slick detections into LineStrings for each detection

        Inputs:
            buf_size: buffer size for cleaning up slick detections
            smoothing_factor: smoothing factor for smoothing centerline
        Returns:
            GeoDataFrame of slick curves
        """
        # print("Creating slick curves")
        # clean up the slick detections by dilation followed by erosion
        # this process can merge some polygons but not others, depending on proximity
        slick_clean = self.slick_gdf.copy()
        slick_clean = self.apply_closing_buffer(
            slick_clean.to_crs(self.crs_meters), buf_size
        )

        # split slicks into individual polygons
        slick_clean = slick_clean.explode(ignore_index=True, index_parts=False)
        self.slick_clean = slick_clean.to_crs(self.slick_gdf.crs)
        # find a centerline through detections
        slick_curves = list()
        for _, row in slick_clean.iterrows():

            # create centerline -> MultiLineString
            polygon_perimeter = row.geometry.length  # Perimeter of the polygon
            interp_dist = min(
                100, polygon_perimeter / 1000
            )  # Use a minimum of 1000 points for voronoi calculation
            cl = centerline.geometry.Centerline(
                row.geometry, interpolation_distance=interp_dist
            )

            # grab coordinates from centerline
            x = list()
            y = list()
            if isinstance(cl.geometry, shapely.geometry.MultiLineString):
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
                    x_fit_sort_x,
                    y_fit_sort_x,
                    k=3,
                    s=smoothing_factor,
                    full_output=True,
                )
                tck_sort_y, fp_sort_y, _, _ = scipy.interpolate.splrep(
                    y_fit_sort_y,
                    x_fit_sort_y,
                    k=3,
                    s=smoothing_factor,
                    full_output=True,
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

        self.slick_clean["areas"] = slick_clean.geometry.area
        self.slick_clean = self.slick_clean.to_crs(self.slick_gdf.crs)
        slick_curves_gdf = gpd.GeoDataFrame(geometry=slick_curves, crs=self.crs_meters)
        slick_curves_gdf["length"] = slick_curves_gdf.geometry.length
        slick_curves_gdf = slick_curves_gdf.to_crs("4326")

        self.slick_curves = slick_curves_gdf


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

    def compute_coincidence_scores(self, slick_gdf: gpd.GeoDataFrame):
        """
        Implement the analysis logic for natural seeps.
        """
        self.slick_gdf = slick_gdf
        pass


ASA_MAPPING = {
    1: AISAnalyzer,
    2: InfrastructureAnalyzer,
    3: DarkAnalyzer,
    4: NaturalAnalyzer,
}
