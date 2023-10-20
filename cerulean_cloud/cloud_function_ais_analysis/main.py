"""cloud function AIS analysis handler
"""

import asyncio
import os

import geopandas as gpd
from shapely import wkt
from utils.ais import AISConstructor
from utils.associate import associate_ais_to_slick, slick_to_curves

from cerulean_cloud.database_client import DatabaseClient, get_engine


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
                    ais_constructor = AISConstructor(s1)
                    ais_constructor.retrieve_ais()
                    if (
                        ais_constructor.ais_gdf is not None
                        and not ais_constructor.ais_gdf.empty
                    ):
                        ais_constructor.build_trajectories()
                        ais_constructor.buffer_trajectories()
                        for slick in slicks_without_sources:
                            ordered_trajs = automatic_ais_analysis(
                                ais_constructor, slick
                            )
                            for idx, traj in (
                                ordered_trajs.get_group(0).iloc[:5].iterrows()
                            ):  # XXX Magic number 5 = number of sources to record for each slick
                                # Insert into SlickToSource:::
                                # slick = slick
                                # source = get_or_insert(traj["traj_id"])
                                # coincidence_score = traj["total_score"]
                                # rank = idx
                                # hitl_confirmed = False
                                # geojson_fc from
                                # geometry from
                                pass

    return "Success!"


def automatic_ais_analysis(ais_constructor, slick):
    """
    Perform automatic analysis to associate AIS trajectories with slicks.

    Parameters:
        ais_constructor (AISTrajectoryAnalysis): An instance of the AISTrajectoryAnalysis class.
        slick (GeoDataFrame): A GeoDataFrame containing the slick geometries.

    Returns:
        GroupBy object: The AIS-slick associations sorted and grouped by slick index.
    """
    slick_gdf = gpd.GeoDataFrame(
        {"geometry": [wkt.loads(str(slick.geometry))]}, crs=ais_constructor.crs_degrees
    ).to_crs(ais_constructor.crs_meters)
    slick_clean, slick_curves = slick_to_curves(slick_gdf)
    slick_ais = associate_ais_to_slick(
        ais_constructor.ais_trajectories,
        ais_constructor.ais_buffered,
        ais_constructor.ais_weighted,
        slick_clean.to_crs(ais_constructor.crs_degrees),
        slick_curves.to_crs(ais_constructor.crs_degrees),
    )
    results = slick_ais.sort_values(
        ["slick_index", "slick_size", "total_score"], ascending=[True, False, False]
    ).groupby("slick_index")
    return results
