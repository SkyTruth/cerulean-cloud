"""
Utilities and helper functions for the AIS class
"""
import os
from datetime import datetime, timedelta

import geopandas as gpd
import movingpandas as mpd
import pandas as pd
import pandas_gbq
import shapely

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
        ais_df (DataFrame): AIS data retrieved from the database.
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
        self.poly = self.s1.geometry.buffer(self.ais_buffer)
        self.start_time = self.s1.start_time - timedelta(hours=self.hours_before)
        self.end_time = self.s1.start_time + timedelta(hours=self.hours_after)
        self.time_vec = pd.date_range(
            start=self.start_time, end=self.end_time, periods=self.num_timesteps
        )

        # Placeholder values
        self.ais_df = None
        self.ais_trajectories = None
        self.ais_buffered = None
        self.ais_weighted = None

    def retrieve_ais(self):
        """
        Retrieve AIS data from Google BigQuery database.

        The function constructs a SQL query and fetches AIS data based on time and spatial constraints.
        The retrieved data is stored in the ais_df attribute.
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
                AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{self.poly}'))
            """
        self.ais_df = pandas_gbq.read_gbq(sql, project_id=os.getenv("BQ_PROJECT_ID"))

    def build_trajectories(self):
        """
        Build maritime trajectories based on the AIS data.

        The function groups AIS data by ssvid (ship identifier) and constructs trajectories.
        It then interpolates or extrapolates the trajectories to match the time vector.
        The resulting trajectories are stored in the ais_trajectories attribute.
        """
        ais_trajectories = list()
        for ssvid, group in self.ais_df.groupby("ssvid"):
            if (
                len(group) > 1
            ):  # ignore single points # XXX Should NOT ignore single points!
                # build trajectory
                traj = mpd.Trajectory(df=group, traj_id=ssvid, t="timestamp")

                # interpolate/extrapolate to times in time_vec
                times = list()
                positions = list()
                for t in self.time_vec:
                    pos = traj.interpolate_position_at(t)
                    times.append(t)
                    positions.append(pos)

                # store as trajectory
                interpolated_traj = mpd.Trajectory(
                    df=gpd.GeoDataFrame(
                        {"timestamp": times, "geometry": positions}, crs=self.ais_df.crs
                    ),
                    traj_id=ssvid,
                    t="timestamp",
                )

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
        for traj in self.ais_df:
            # grab points
            points = traj.to_point_gdf()
            points = points.sort_values(by="timestamp", ascending=False)

            # create buffered circles at points
            ps = list()
            for idx, buffer in enumerate(self.buf_vec):
                ps.append(points.iloc[idx].geometry.buffer(buffer))

            # create convex hulls from circles
            n = range(len(ps) - 1)
            convex_hulls = [
                shapely.geometry.MultiPolygon([ps[i], ps[i + 1]]).convex_hull for i in n
            ]

            # weight convex hulls
            weighted = list()
            for cidx, c in enumerate(convex_hulls):
                entry = dict()
                entry["geometry"] = c
                entry["weight"] = 1.0 / (cidx + 1)  # weight is the inverse of the index
                weighted.append(entry)
            weighted = gpd.GeoDataFrame(weighted, crs=traj.crs)
            ais_weighted.append(weighted)

            # create polygon from hulls
            poly = shapely.ops.unary_union(convex_hulls)
            ais_buf.append(poly)

        self.ais_buffered = gpd.GeoDataFrame(geometry=ais_buf, crs=traj.crs)
        self.ais_weighted = ais_weighted