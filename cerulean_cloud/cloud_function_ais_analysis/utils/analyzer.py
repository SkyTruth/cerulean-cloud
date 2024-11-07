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
import shapely
from geoalchemy2.shape import to_shape
from pyproj import CRS
from scipy.spatial import cKDTree
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon

# Constants (adjust as needed)
AIS_BUFFER = 50000  # Meters
BUF_VEC = [5000, 4000, 3000, 2000, 1000]
WEIGHT_VEC = [1.0, 0.8, 0.6, 0.4, 0.2]
HOURS_BEFORE = 24
HOURS_AFTER = 24
NUM_TIMESTEPS = 5
D_FORMAT = "%Y-%m-%d"
T_FORMAT = "%Y-%m-%d %H:%M:%S"


class SourceAnalyzer:
    """
    Base class for source analysis.

    Attributes:
        slick_gdf (GeoDataFrame): GeoDataFrame containing the slick geometries.
    """

    def __init__(self, slick_gdf: gpd.GeoDataFrame, scene_id: str, **kwargs):
        """
        Initialize the SourceAnalyzer.
        """
        self.slick_gdf = slick_gdf
        self.scene_id = scene_id
        # Any common initializations

    def load_sources(self):
        """
        Placeholder method to be overridden
        """
        pass

    def compute_coincidence_scores(self):
        """
        Placeholder method to be overridden
        """
        pass

    def associate_sources_to_slicks(self):
        """
        Placeholder method to be overridden
        """
        pass

    def estimate_utm_crs(self, geometry):
        """
        Estimates an appropriate UTM CRS based on the centroid of the geometry.
        """
        return CRS.from_dict(
            {
                "proj": "utm",
                "zone": int((geometry.centroid.x + 180) / 6) + 1,
                "south": geometry.centroid.y < 0,
            }
        )


class InfrastructureAnalyzer(SourceAnalyzer):
    """
    Analyzer for fixed infrastructure sources.

    Attributes:
        infra_gdf (GeoDataFrame): GeoDataFrame containing infrastructure points.
        coincidence_scores (np.ndarray): Computed confidence scores.
    """

    def __init__(self, slick_gdf, scene_id, **kwargs):
        """
        Initialize the InfrastructureAnalyzer.
        """
        super().__init__(slick_gdf, scene_id, **kwargs)
        self.num_vertices = kwargs.get("N", 10)
        self.closing_buffer = kwargs.get("closing_buffer", 500)
        self.radius_of_interest = kwargs.get("radius_of_interest", 3000)
        self.min_area_threshold = kwargs.get("min_area_threshold", 0.1)

        self.crs_meters = self.estimate_utm_crs(self.slick_gdf.unary_union)
        self.infra_api_token = os.getenv("infra_api_token")
        self.infra_gdf = self.load_infrastructure_data()
        self.coincidence_scores = np.zeros(len(self.infra_gdf))

    def load_infrastructure_data(self, only_oil=True):
        """
        Loads infrastructure data from a CSV file.

        Parameters:
            infra_api_token (str): API token for infrastructure data.

        Returns:
            GeoDataFrame: GeoDataFrame containing infrastructure points.
        """

        # zxy = self.select_enveloping_tile() # Something's wrong with this code. Ex. [3105854, 'S1A_IW_GRDH_1SDV_20230806T221833_20230806T221858_049761_05FBD2_577C"] should have 2 nearby infras
        mvt_data = self.download_mvt_tile()

        df = pd.DataFrame([d["properties"] for d in mvt_data["main"]["features"]])

        datetime_fields = ["structure_start_date", "structure_end_date"]
        for field in datetime_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field])
                df[field] = pd.to_datetime(df[field], unit="ms")
        df["source_type"] = "infra"
        if only_oil:
            df = df[df["label"] == "oil"]

        return gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="epsg:4326"
        )

    def select_enveloping_tile(self, max_zoom=20):
        """
        Determine the minimal zoom level and tile coordinates (x, y, z)
        that cover the area of interest (slick_gdf buffered by radius_of_interest)
        in a single tile.
        """

        buffered_slick_gdf = (
            self.slick_gdf.to_crs(self.crs_meters)
            .envelope.buffer(self.radius_of_interest)
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
            return {}

    def apply_closing_buffer(self, geo_df, closing_buffer):
        """
        Applies a closing buffer to geometries in the GeoDataFrame.
        """
        geo_df["geometry"] = (
            geo_df["geometry"].buffer(closing_buffer).buffer(-closing_buffer)
        )
        return geo_df

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
        radius_of_interest: float,
    ) -> np.ndarray:
        """
        Computes confidence scores for infrastructure points based on proximity to extremity points.
        """
        infra_coords = np.array([(geom.x, geom.y) for geom in infra_gdf.geometry])
        extremity_indices = extremity_tree.query_ball_point(
            infra_coords, r=radius_of_interest
        )
        coincidence_scores = np.zeros(len(infra_coords))

        for i, neighbors in enumerate(extremity_indices):
            if neighbors:
                neighbor_points = all_extremity_points[neighbors]
                neighbor_weights = all_weights[neighbors]
                dists = np.linalg.norm(neighbor_points - infra_coords[i], axis=1)
                C_i = neighbor_weights - dists / radius_of_interest
                coincidence_scores[i] = np.clip(C_i.max(), 0, 1)

        return coincidence_scores

    def compute_coincidence_scores(self):
        """
        Computes coincidence scores for infrastructure points.
        """
        start_time = time.time()

        slick_gdf = self.slick_gdf.to_crs(self.crs_meters)
        infra_gdf = self.infra_gdf.to_crs(self.crs_meters)

        # Apply closing buffer and project slick_gdf
        slick_gdf = self.apply_closing_buffer(slick_gdf, self.closing_buffer)

        # Combine geometries and extract polygons
        combined_geometry = slick_gdf.unary_union
        polygons = self.extract_polygons(combined_geometry)

        slick_buffered = combined_geometry.buffer(self.radius_of_interest)

        # Filter based on scene date
        scene_date = pd.to_datetime(self.scene_id[17:25], format="%Y%m%d")
        filtered_infra = infra_gdf[infra_gdf["structure_start_date"] < scene_date]
        filtered_infra = filtered_infra[
            (infra_gdf["structure_end_date"] > scene_date)
            | (infra_gdf["structure_end_date"].isna())
        ]

        # Filter based on radius of interest
        filtered_infra = filtered_infra[filtered_infra.geometry.within(slick_buffered)]

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
            self.radius_of_interest,
        )

        self.coincidence_scores[filtered_infra.index] = confidence_filtered
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")

    def associate_sources_to_slicks(self):
        """
        Associates infrastructure sources with slicks.
        """
        self.compute_coincidence_scores()

        # Return a DataFrame with infra_gdf and coincidence_scores
        self.infra_gdf["coincidence_score"] = self.coincidence_scores
        return self.infra_gdf[self.infra_gdf["coincidence_score"] > 0]  # Filter out 0s


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

    def __init__(self, slick_gdf, scene_id, s1_scene, **kwargs):
        """
        Initialize the AISAnalyzer.
        """
        super().__init__(slick_gdf, scene_id, **kwargs)
        self.s1_scene = s1_scene
        # Default parameters
        self.hours_before = kwargs.get("hours_before", HOURS_BEFORE)
        self.hours_after = kwargs.get("hours_after", HOURS_AFTER)
        self.ais_buffer = kwargs.get("ais_buffer", AIS_BUFFER)
        self.num_timesteps = kwargs.get("num_timesteps", NUM_TIMESTEPS)
        self.buf_vec = kwargs.get("buf_vec", BUF_VEC)
        self.weight_vec = kwargs.get("weight_vec", WEIGHT_VEC)
        # Initialize other attributes
        self.ais_gdf = None
        self.ais_trajectories = None
        self.ais_buffered = None
        self.ais_weighted = None
        self.results = None
        self.crs_meters = None
        self.s1_env = None
        # Additional initializations
        self.initialize_scene()

    def initialize_scene(self):
        """
        Initializes the scene parameters.
        """
        self.start_time = self.s1_scene.start_time - timedelta(hours=self.hours_before)
        self.end_time = self.s1_scene.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.start_time,
            end=self.s1_scene.start_time,
            periods=self.num_timesteps,
        )
        self.s1_env = gpd.GeoDataFrame(
            {"geometry": [to_shape(self.s1_scene.geometry)]}, crs="4326"
        )
        self.crs_meters = self.estimate_utm_crs(self.slick_gdf.unary_union)
        self.envelope = (
            self.s1_env.to_crs(self.crs_meters).buffer(self.ais_buffer).to_crs("4326")
        )

    def retrieve_ais_data(self):
        """
        Retrieves AIS data from BigQuery.
        """
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
                seg._PARTITIONTIME between '{datetime.strftime(self.start_time, D_FORMAT)}' AND '{datetime.strftime(self.end_time, D_FORMAT)}'
                AND seg.timestamp between '{datetime.strftime(self.start_time, T_FORMAT)}' AND '{datetime.strftime(self.end_time, T_FORMAT)}'
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{self.envelope.iloc[0].geometry.wkt}'))
            """
        df = pandas_gbq.read_gbq(
            sql,
            project_id="world-fishing-827",  # credentials=credentials
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
        ais_trajectories = list()
        for st_name, group in self.ais_gdf.groupby("ssvid"):
            # Duplicate the row if there's only one point
            if len(group) == 1:
                group = pd.concat([group] * 2).reset_index(drop=True)

            # Build trajectory
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
            s1_time = pd.Timestamp(times[-1]).tz_localize("UTC")
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

    def slick_to_curves(self, slick_gdf, crs_meters):
        """
        Converts slick polygons to curves for analysis.
        """
        slick_gdf_meters = slick_gdf.to_crs(crs_meters)
        slick_curves = []
        for geom in slick_gdf_meters.geometry:
            if isinstance(geom, (Polygon, MultiPolygon)):
                coords = np.array(geom.exterior.coords)
                curve = LineString(coords)
                slick_curves.append(curve)
        return slick_gdf_meters, slick_curves

    def associate_ais_with_slick(
        self,
        ais_trajectories,
        ais_buffered,
        ais_weighted,
        slick_gdf,
        slick_curves,
        crs_meters,
    ):
        """
        Associates AIS trajectories with slicks based on spatial relationships.
        """
        associations = []
        for traj, buffered, weighted in zip(
            ais_trajectories, ais_buffered.geometry, self.ais_weighted
        ):
            # Check intersection with slick
            if buffered.intersects(slick_gdf.unary_union):
                # Compute coincidence score
                coincidence_score = weighted["weight"].sum()
                associations.append(
                    {
                        "st_name": traj.id,
                        "coincidence_score": coincidence_score,
                        "geometry": traj.to_line_gdf().geometry.iloc[0],
                        "geojson_fc": traj.geojson_fc,
                    }
                )
        return pd.DataFrame(associations)

    def compute_coincidence_scores(self):
        """
        Computes coincidence scores for AIS trajectories.
        """
        self.results = self.associate_ais_to_slick()
        return self.results

    def associate_ais_to_slick(self):
        """
        Associates AIS trajectories with slicks.
        """
        # Transform slick_gdf to appropriate CRS
        slick_gdf = self.slick_gdf.to_crs("4326")
        _, slick_curves = self.slick_to_curves(slick_gdf, self.crs_meters)

        ais_associations = self.associate_ais_with_slick(
            self.ais_trajectories,
            self.ais_buffered,
            self.ais_weighted,
            slick_gdf,
            slick_curves,
            self.crs_meters,
        )

        return ais_associations

    def associate_sources_to_slicks(self):
        """
        Associates AIS trajectories with slicks.
        """
        self.retrieve_ais_data()
        if not self.ais_gdf.empty:
            self.build_trajectories()
            self.buffer_trajectories()
            self.compute_coincidence_scores()
            # Return the results
            return self.results
        else:
            # No AIS data
            return pd.DataFrame()  # Empty DataFrame


class DarkAnalyzer(SourceAnalyzer):
    """
    Analyzer for dark vessels (non-AIS broadcasting vessels).
    Currently a placeholder for future implementation.
    """

    def __init__(self, slick_gdf, s1, **kwargs):
        """
        Initialize the DarkAnalyzer.
        """
        super().__init__(slick_gdf, s1, **kwargs)
        # Initialize attributes specific to dark vessel analysis

    def compute_coincidence_scores(self):
        """
        Implement the analysis logic for dark vessels.
        """
        pass

    def associate_sources_to_slicks(self):
        """
        Associates dark vessels with slicks.
        """
        # Return associations
        pass
