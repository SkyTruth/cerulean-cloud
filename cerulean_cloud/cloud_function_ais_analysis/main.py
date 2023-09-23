"""cloud function AIS analysis handler
"""

import asyncio
import json
import os
from datetime import datetime, timedelta

import pandas_gbq

# import shapely.geometry as sh  # https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

from cerulean_cloud.cloud_function_ais_analysis.utils.associate import (
    associate_ais_to_slicks,
    slicks_to_curves,
)
from cerulean_cloud.cloud_function_ais_analysis.utils.constants import (
    AIS_BUFFER,
    BUF_VEC,
    D_FORMAT,
    HOURS_AFTER,
    HOURS_BEFORE,
    NUM_TIMESTEPS,
    T_FORMAT,
    WEIGHT_VEC,
)
from cerulean_cloud.cloud_function_ais_analysis.utils.misc import (
    build_time_vec,
    get_utm_zone,
)
from cerulean_cloud.cloud_function_ais_analysis.utils.trajectory import (
    ais_points_to_trajectories,
    buffer_trajectories,
)
from cerulean_cloud.database_client import DatabaseClient, get_engine

# mypy: ignore-errors


def main(request):
    """
    Entry point for handling HTTP requests.

    Args:
        request (flask.Request): The incoming HTTP request object.

    Returns:
        Any: The response object or any value that can be converted to a
        Response object using Flask's `make_response`.

    Notes:
        - This function sets up an asyncio event loop and delegates the actual
          request handling to `handle_aaa_request`.
        - It's important to set up a new event loop if the function is running
          in a context where the default event loop is not available (e.g., in some WSGI servers).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(handle_aaa_request(request))
    return res


async def handle_aaa_request(request):
    """
    Asynchronously handles the main logic of Automatic AIS Analysis (AAA) for a given scene.

    Args:
        request (flask.Request): The incoming HTTP request object containing scene information.

    Returns:
        str: A string "Success!" upon successful completion.

    Notes:
        - The function gets scene information from the request.
        - It uses the `DatabaseClient` for database operations.
        - It calls `get_ais` and `automatic_ais_analysis` for AIS data retrieval and analysis.
    """
    request_json = request.get_json()
    if not request_json.get("dry_run"):
        scene_id = request_json.get("scene_id")
        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1 = await db_client.get_scene_from_id(scene_id)
                slicks_without_sources = (
                    await db_client.get_slicks_without_sources_from_scene_id(scene_id)
                )
                if len(slicks_without_sources) > 0:
                    ais_traj, ais_buffered, ais_weighted, utm_zone = get_ais(s1)
                    if ais_traj:
                        for slick in slicks_without_sources:
                            automatic_ais_analysis(
                                slick, ais_traj, ais_buffered, ais_weighted, utm_zone
                            )
    return "Success!"


async def get_ais(
    s1, hours_before=HOURS_BEFORE, hours_after=HOURS_AFTER, ais_buffer=AIS_BUFFER
):
    """
    Asynchronously fetches and processes AIS data.

    Args:
        s1 (Scene Object): The scene object for which AIS data is needed.
        hours_before (int): The number of hours before the scene time to consider for AIS data.
        hours_after (int): The number of hours after the scene time to consider for AIS data.
        ais_buffer (float): The buffer distance around the scene geometry.

    Returns:
        tuple: A tuple containing ais_traj, ais_buffered, ais_weighted, and utm_zone.

    Notes:
        - AIS data is downloaded and then transformed into trajectories.
        - The function also buffers and weighs the AIS trajectories.
    """
    grd_buff = s1.geometry.buffer(ais_buffer)
    ais = download_ais(s1.start_time, hours_before, hours_after, grd_buff)
    time_vec = build_time_vec(s1.start_time, hours_before, hours_after, NUM_TIMESTEPS)
    utm_zone = get_utm_zone(ais)
    ais_traj = ais_points_to_trajectories(ais, time_vec)
    ais_buffered, ais_weighted = buffer_trajectories(ais_traj, BUF_VEC, WEIGHT_VEC)

    return ais_traj, ais_buffered, ais_weighted, utm_zone


def download_ais(
    t_stamp,
    hours_before,
    hours_after,
    poly,
):
    """
    Downloads AIS data using a SQL query on BigQuery.

    Args:
        t_stamp (datetime): The timestamp for which AIS data is needed.
        hours_before (int): The number of hours before the timestamp to consider for AIS data.
        hours_after (int): The number of hours after the timestamp to consider for AIS data.
        poly (str): The polygon geometry in WKT format to filter AIS data spatially.

    Returns:
        DataFrame: A Pandas DataFrame containing the downloaded AIS data.

    Notes:
        - The function uses Google's BigQuery Python client `pandas_gbq` to execute the SQL query.
        - Make sure that the BigQuery project ID is set in the environment variable "BQ_PROJECT_ID".
    """
    sql = f"""
        SELECT * FROM(
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
        seg._PARTITIONTIME between '{datetime.strftime(t_stamp-timedelta(hours=hours_before), D_FORMAT)}' AND '{datetime.strftime(t_stamp+timedelta(hours=hours_after), D_FORMAT)}'
        AND seg.timestamp between '{datetime.strftime(t_stamp-timedelta(hours=hours_before), T_FORMAT)}' AND '{datetime.strftime(t_stamp+timedelta(hours=hours_after), T_FORMAT)}'
        AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), ST_GeogFromText('{poly}'))
        ORDER BY
        seg.timestamp DESC
        )
        ORDER BY
        ssvid, timestamp
        """
    return pandas_gbq.read_gbq(sql, project_id=os.getenv("BQ_PROJECT_ID"))


def automatic_ais_analysis(slick, ais_traj, ais_buffered, ais_weighted, utm_zone):
    """
    Performs automatic AIS analysis for a given slick and AIS data.

    Args:
        slick (GeoDataFrame): The oil slick geometry.
        ais_traj (GeoDataFrame): The AIS trajectories.
        ais_buffered (GeoDataFrame): The buffered AIS trajectories.
        ais_weighted (GeoDataFrame): The weighted AIS trajectories.
        utm_zone (str): The UTM zone for coordinate transformation.

    Returns:
        DataFrame: A Pandas DataFrame containing the AIS analysis results sorted by slick index and score.

    Notes:
        - The function performs geometry transformations and data association.
        - It uses the helper functions `slicks_to_curves` and `associate_ais_to_slicks`.
    """
    slicks = slick.to_crs(utm_zone)
    slicks_clean, slicks_curves = slicks_to_curves(slicks)
    slick_ais = associate_ais_to_slicks(
        ais_traj, ais_buffered, ais_weighted, slicks_clean, slicks_curves
    )
    results = slick_ais.sort_values(
        ["slick_index", "slick_size", "total_score"], ascending=[True, False, False]
    ).groupby("slick_index")
    return results


def add_to_aaa_queue(scene_id):
    """
    Adds a new task to Google Cloud Tasks for automatic AIS analysis.

    Args:
        scene_id (str): The ID of the scene for which AIS analysis is needed.

    Returns:
        google.cloud.tasks_v2.types.Task: The created Task object.

    Notes:
        - The function uses Google Cloud Tasks API to schedule the AIS analysis.
        - Multiple retries are scheduled with different delays.
    """
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION")
    queue = os.getenv("QUEUE")
    url = os.getenv("FUNCTION_URL")
    dry_run = bool(os.getenv("IS_DRY_RUN"))

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    payload = {"sceneid": scene_id, "dry_run": dry_run}

    task = {
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,  # The url path that the task will be sent to.
        },
        "headers": {
            "Content-type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
        },
        "body": json.dumps(payload).encode(),
    }

    # Number of days that the Automatic AIS Analysis should be run after
    # Each entry is another retry
    ais_delays = [0, 3, 7]  # TODO Magic number >>> Where should this live?
    for delay in ais_delays:
        d = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
            days=delay
        )

        # Create Timestamp protobuf.
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(d)

        # Add the timestamp to the tasks.
        task["schedule_time"] = timestamp

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print("Created task {}".format(response.name))
    return response
