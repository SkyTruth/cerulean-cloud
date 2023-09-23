"""
Miscellaneous collection of helper functions
"""

import datetime

import geopandas as gpd
import pandas as pd
import pyproj


def build_time_vec(
    collect_time: datetime.datetime,
    hours_before: int,
    hours_after: int,
    num_timesteps: int,
) -> pd.DatetimeIndex:
    """
    Build a vector of times, starting at a given time and going back in time

    Inputs:
        collect_time: datetime object for the time of collection
        hours_before: number of hours before the collection time to start the vector
        hours_after: number of hours after the collection time to end the vector
        num_timesteps: number of timesteps in the vector
    Returns:
        DatetimeIndex of times
    """
    start_time = collect_time - datetime.timedelta(hours=hours_before)
    end_time = collect_time + datetime.timedelta(hours=hours_after)
    return pd.date_range(start=start_time, end=end_time, periods=num_timesteps)


def get_utm_zone(gdf: gpd.GeoDataFrame) -> pyproj.crs.crs.CRS:
    """
    Get the UTM zone for a GeoDataFrame

    Inputs:
        gdf: GeoDataFrame to get the UTM zone for
    Returns:
        UTM zone as a pyproj CRS object
    """
    return gdf.estimate_utm_crs()
