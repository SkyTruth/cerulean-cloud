"""
Utilities and helper functions for the AIS class
"""
import json
import os
from datetime import datetime, timedelta

import geopandas as gpd
import movingpandas as mpd
import pandas as pd
import pandas_gbq
import shapely
from geoalchemy2.shape import to_shape
from google.oauth2.service_account import Credentials

from .constants import (
    AIS_BUFFER,
    BUF_VEC,
    D_FORMAT,
    HOURS_AFTER,
    HOURS_BEFORE,
    NUM_TIMESTEPS,
    T_FORMAT,
    WEIGHT_VEC,
)

credentials = Credentials.from_service_account_info(
    json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
)


class AISConstructor:
    """
    A class for constructing AIS (Automatic Identification System) data for maritime trajectories.

    Attributes:
        s1 (object): Initial S1 scene with start time and geometry.
        hours_before (int): Number of hours before the start time to consider.
        hours_after (int): Number of hours after the start time to consider.
        ais_buffer (float): Buffer radius for the S1 scene.
        num_timesteps (int): Number of timesteps for interpolation/extrapolation.
        buf_vec (list): Vector of buffer radii for trajectory points.
        weight_vec (list): Vector of weights for trajectory points.
        poly (object): Buffered geometry of the S1 scene.
        start_time (datetime): Start time for the analysis.
        end_time (datetime): End time for the analysis.
        time_vec (list): Vector of datetime objects for interpolation/extrapolation.
        ais_gdf (DataFrame): AIS data retrieved from the database.
        ais_trajectories (list): List of interpolated/extrapolated trajectories.
        ais_buffered (DataFrame): Dataframe containing buffered trajectories.
        ais_weighted (list): List of weighted geometries for each trajectory.
    """

    def __init__(
        self,
        s1,
        hours_before=HOURS_BEFORE,
        hours_after=HOURS_AFTER,
        ais_buffer=AIS_BUFFER,
        num_timesteps=NUM_TIMESTEPS,
        buf_vec=BUF_VEC,
        weight_vec=WEIGHT_VEC,
    ):
        """
        Initialize an AISTrajectoryAnalysis object.

        Parameters:
            s1 (object): Initial spatial object with start time and geometry.
            hours_before (int, optional): Number of hours before the start time to consider. Defaults to HOURS_BEFORE.
            hours_after (int, optional): Number of hours after the start time to consider. Defaults to HOURS_AFTER.
            ais_buffer (float, optional): Buffer radius for the spatial object. Defaults to AIS_BUFFER.
            num_timesteps (int, optional): Number of timesteps for interpolation/extrapolation. Defaults to NUM_TIMESTEPS.
            buf_vec (list, optional): Vector of buffer radii for trajectory points. Defaults to BUF_VEC.
            weight_vec (list, optional): Vector of weights for trajectory points. Defaults to WEIGHT_VEC.
        """
        # Default values
        self.s1 = s1
        self.hours_before = hours_before
        self.hours_after = hours_after
        self.ais_buffer = ais_buffer
        self.num_timesteps = num_timesteps
        self.buf_vec = buf_vec
        self.weight_vec = weight_vec

        # Calculated values
        self.start_time = self.s1.start_time - timedelta(hours=self.hours_before)
        self.end_time = self.s1.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.start_time, end=self.s1.start_time, periods=self.num_timesteps
        )
        self.s1_env = gpd.GeoDataFrame(
            {"geometry": [to_shape(self.s1.geometry)]}, crs="4326"
        )
        self.crs_meters = self.s1_env.estimate_utm_crs()
        self.envelope = (
            self.s1_env.to_crs(self.crs_meters).buffer(self.ais_buffer).to_crs("4326")
        )

        # Placeholder values
        self.sql = None
        self.ais_gdf = None
        self.ais_trajectories = None
        self.ais_buffered = None
        self.ais_weighted = None
        self.infra_gdf = None

    def retrieve_ais(self):
        """
        Retrieve AIS data from Google BigQuery database.

        The function constructs a SQL query and fetches AIS data based on time and spatial constraints.
        The retrieved data is stored in the ais_gdf attribute.
        """
        self.sql = f"""
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
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{self.envelope[0]}'))
            """
        df = pandas_gbq.read_gbq(
            self.sql, project_id="world-fishing-827", credentials=credentials
        )
        df["geometry"] = df.apply(
            lambda row: shapely.geometry.Point(row["lon"], row["lat"]), axis=1
        )
        self.ais_gdf = (
            gpd.GeoDataFrame(df, crs="4326")
            .sort_values(by=["ssvid", "timestamp"])
            .reset_index(drop=True)
        )

    def load_infra(self, filepath):
        """Load global infrastructure CSV"""
        infra_gdf = gpd.GeoDataFrame.from_file(filepath)

        # Convert infrastructure data to Point geometries using longitude and latitude columns
        infra_gdf["geometry"] = infra_gdf.apply(
            lambda row: shapely.geometry.Point(
                row["clust_centr_lon"], row["clust_centr_lat"]
            ),
            axis=1,
        )

        # Set CRS for infrastructure data to WGS 84 and transform it to match the CRS of the 'slick' GeoDataFrame
        self.infra_gdf = infra_gdf.set_crs("4326")

    def build_trajectories(self):
        """
        Build maritime trajectories based on the AIS data.

        The function groups AIS data by ssvid (ship identifier) and constructs trajectories.
        It then interpolates or extrapolates the trajectories to match the time vector.
        The resulting trajectories are stored in the ais_trajectories attribute.
        """
        ais_trajectories = list()
        for st_name, group in self.ais_gdf.groupby("ssvid"):
            # Duplicate the row if there's only one point
            if len(group) == 1:
                group = pd.concat([group] * 2).reset_index(drop=True)

            # build trajectory
            traj = mpd.Trajectory(df=group, traj_id=st_name, t="timestamp")

            # interpolate to times in time_vec
            times = list()
            positions = list()
            for t in self.time_vec:
                pos = traj.interpolate_position_at(t)
                times.append(t)
                positions.append(pos)
            gdf = gpd.GeoDataFrame(
                {"timestamp": times, "geometry": positions}, crs="4326"
            )

            # store as trajectory
            interpolated_traj = mpd.Trajectory(
                gdf,
                traj_id=st_name,
                t="timestamp",
            )
            gdf["timestamp"] = gdf["timestamp"].apply(lambda x: x.isoformat())
            interpolated_traj.ext_name = group.iloc[0]["shipname"]
            interpolated_traj.ext_shiptype = group.iloc[0]["best_shiptype"]
            interpolated_traj.flag = group.iloc[0]["flag"]

            # calculate ideal display feature collection, which includes all the original datapoints, plus the time the image was taken (interpolation), minus everything after that point
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
        Create buffered geometries around the trajectories.

        The function iterates over each trajectory, buffers individual points, and then constructs
        convex hulls around these points. Weighted geometries are also created based on the buffer radii.
        The resulting buffered and weighted geometries are stored in the ais_buffered and ais_weighted attributes.
        """
        ais_buf = list()
        ais_weighted = list()
        for traj in self.ais_trajectories:
            # grab points
            points = (
                traj.to_point_gdf()
                .sort_values(by="timestamp", ascending=False)
                .to_crs(self.crs_meters)
                .reset_index()
            )

            # create buffered circles at points
            ps = (
                points.apply(
                    lambda row: row.geometry.buffer(self.buf_vec[row.name]), axis=1
                )
                .set_crs(self.crs_meters)
                .to_crs("4326")
            )

            # create convex hulls from sequential circles
            convex_hulls = [
                shapely.geometry.MultiPolygon([a, b]).convex_hull
                for a, b in zip(ps[:-1], ps[1:])
            ]

            # weight convex hulls
            weighted = gpd.GeoDataFrame(
                {"geometry": convex_hulls, "weight": self.weight_vec[:-1]},
                crs="4326",
            )
            ais_weighted.append(weighted)

            # create connected polygon from hulls
            ais_buf.append(
                {"geometry": shapely.ops.unary_union(convex_hulls), "st_name": traj.id}
            )

        self.ais_buffered = gpd.GeoDataFrame(ais_buf, crs="4326")
        self.ais_weighted = ais_weighted
