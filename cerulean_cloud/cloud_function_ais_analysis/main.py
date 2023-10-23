"""cloud function AIS analysis handler
"""

import asyncio
import os

import geopandas as gpd
from shapely import wkb
from shapely.geometry import LineString
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
                            ais_associations = automatic_ais_analysis(
                                ais_constructor, slick
                            )
                            if len(ais_associations) > 0:
                                # XXX What to do if len(ordered_ass)==0 and no sources are associated?
                                # Then it will trigger another round of this process later! (unnecessary computation)
                                for idx, traj in (
                                    ais_associations.get_group(
                                        0  # XXX Magic number 0 = first polygon in the slick
                                    )
                                    .iloc[
                                        :5  # XXX Magic number 5 = number of sources to record for each slick
                                    ]
                                    .iterrows()
                                ):
                                    single_track = (
                                        ais_constructor.ais_gdf[
                                            ais_constructor.ais_gdf["ssvid"]
                                            == traj["st_name"]
                                        ]
                                        .to_crs(ais_constructor.crs_degrees)
                                        .sort_values(by="timestamp")
                                        .assign(
                                            timestamp=lambda x: x["timestamp"].astype(
                                                str
                                            )
                                        )
                                    )
                                    source = await db_client.get_source(
                                        st_name=traj["st_name"]
                                    )
                                    if source is None:
                                        source = await db_client.insert_source(
                                            st_name=traj["st_name"],
                                            source_type=1,  # XXX This will need to be dynamic for SSS
                                            # XXX This is where we would pass in the kwargs for this source SSS
                                        )
                                    await db_client.insert_slick_to_source(
                                        source=source.id,
                                        slick=slick.id,
                                        coincidence_score=traj["total_score"],
                                        rank=idx,
                                        geojson_fc=single_track.to_json(),
                                        geometry=LineString(
                                            single_track["geometry"]
                                        ).wkt,
                                    )

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
        {"geometry": [wkb.loads(str(slick.geometry))]}, crs=ais_constructor.crs_degrees
    ).to_crs(ais_constructor.crs_meters)
    slick_clean, slick_curves = slick_to_curves(
        slick_gdf
    )  # XXX This splits the gdf into single parts! BAD
    associations = associate_ais_to_slick(
        ais_constructor.ais_trajectories,
        ais_constructor.ais_buffered,
        ais_constructor.ais_weighted,
        slick_clean,
        slick_curves,
    )
    results = associations.sort_values(
        ["poly_index", "poly_size", "total_score"], ascending=[True, False, False]
    ).groupby("poly_index")
    return results
