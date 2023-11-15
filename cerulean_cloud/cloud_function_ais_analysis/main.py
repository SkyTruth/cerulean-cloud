"""cloud function AIS analysis handler
"""

import asyncio
import os

import geopandas as gpd
import pandas as pd
from shapely import wkb
from utils.ais import AISConstructor
from utils.associate import (
    associate_ais_to_slick,
    associate_infra_to_slick,
    slick_to_curves,
)

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
        print(f"Running AAA on scene_id: {scene_id}")
        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1 = await db_client.get_scene_from_id(scene_id)
                AAA_CONFIDENCE_THRESHOLD = 0.5
                slicks_without_sources = (
                    await db_client.get_slicks_without_sources_from_scene_id(
                        scene_id, AAA_CONFIDENCE_THRESHOLD
                    )
                )
                print(f"# Slicks found: {len(slicks_without_sources)}")
                if len(slicks_without_sources) > 0:
                    aisc = AISConstructor(s1)
                    aisc.retrieve_ais()
                    print("AIS retrieved")
                    if not aisc.ais_gdf.empty:
                        aisc.build_trajectories()
                        aisc.buffer_trajectories()
                        # We only load infra AFTER we've confirmed there are AIS points, otherwise get_slicks_without_sources_from_scene_id would prevent AIS tracks from being processed later
                        aisc.load_infra("20231103_all_infrastructure_v20231103.csv")
                        for slick in slicks_without_sources:
                            source_associations = automatic_source_analysis(aisc, slick)
                            print(
                                f"{len(source_associations)} found for Slick ID: {slick.id}"
                            )
                            if len(source_associations) > 0:
                                # XXX What to do if len(ais_associations)==0 and no sources are associated?
                                # Then it will trigger another round of this process later! (unnecessary computation)
                                RECORD_NUM_SOURCES = 5  # XXX Magic number 5 = number of sources to record for each slick
                                for idx, traj in source_associations.iloc[
                                    :RECORD_NUM_SOURCES
                                ].iterrows():
                                    source = await db_client.get_source(
                                        st_name=traj["st_name"]
                                    )
                                    if source is None:
                                        source = (
                                            await db_client.insert_source_from_traj(
                                                traj
                                            )
                                        )
                                    await db_client.session.flush()

                                    print(
                                        f'{type(traj["geometry"])} : type(traj["geometry"]) {scene_id} : {slick.id} : {source.id}'
                                    )
                                    print(
                                        f'{traj["geometry"]} : traj["geometry"] {scene_id} : {slick.id} : {source.id}'
                                    )

                                    await db_client.insert_slick_to_source(
                                        source=source.id,
                                        slick=slick.id,
                                        coincidence_score=traj["coincidence_score"],
                                        rank=idx + 1,
                                        geojson_fc=traj["geojson_fc"],
                                        geometry=traj["geometry"],
                                    )

    return "Success!"


def automatic_source_analysis(aisc, slick):
    """
    Perform automatic analysis to associate AIS trajectories with slicks.

    Parameters:
        ais (ais_constructor): An instance of the ais_constructor class.
        slick (GeoDataFrame): A GeoDataFrame containing the slick geometries.

    Returns:
        GroupBy object: The AIS-slick associations sorted and grouped by slick index.
    """
    slick_gdf = gpd.GeoDataFrame(
        {"geometry": [wkb.loads(str(slick.geometry)).buffer(0)]}, crs="4326"
    )
    _, slick_curves = slick_to_curves(slick_gdf, aisc.crs_meters)

    ais_associations = associate_ais_to_slick(
        aisc.ais_trajectories,
        aisc.ais_buffered,
        aisc.ais_weighted,
        slick_gdf,
        slick_curves,
        aisc.crs_meters,
    )

    infra_associations = associate_infra_to_slick(
        aisc.infra_gdf, slick_gdf, aisc.crs_meters
    )

    all_associations = pd.concat(
        [ais_associations, infra_associations], ignore_index=True
    )

    results = all_associations.sort_values(
        "coincidence_score", ascending=False
    ).reset_index(drop=True)
    return results
