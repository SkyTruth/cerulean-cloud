"""cloud function Automatic Source Association handler"""

import asyncio
import os
import random

import geopandas as gpd
import pandas as pd
from flask import abort
from shapely import wkb
from utils.analyzer import ASA_MAPPING

from cerulean_cloud.database_client import DatabaseClient, get_engine


def verify_api_key(request):
    """Function to verify API key"""
    expected_api_key = os.getenv("API_KEY")
    auth_header = request.headers.get("Authorization")

    # Check if the Authorization header is present
    if not auth_header:
        abort(401, description="Unauthorized: No Authorization header")

    # Split the header into 'Bearer' and the token part
    parts = auth_header.split()

    # Check if the header is formed correctly
    if parts[0].lower() != "bearer" or len(parts) != 2:
        abort(401, description="Unauthorized: Invalid Authorization header format")

    request_api_key = parts[1]

    # Compare the token part with your expected API key
    if request_api_key != expected_api_key:
        abort(401, description="Unauthorized: Invalid API key")


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
          request handling to `handle_asa_request`.
        - It's important to set up a new event loop if the function is running
          in a context where the default event loop is not available (e.g., in some WSGI servers).
    """
    verify_api_key(request)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        res = loop.run_until_complete(handle_asa_request(request))
        return res
    finally:
        # Ensure all async generators finish and close the loop properly
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def handle_asa_request(request):
    """
    Asynchronously handles the main logic of Automatic Source Analysis (ASA) for a given scene.

    Args:
        request (flask.Request): The incoming HTTP request object containing scene information.

    Returns:
        str: A string "Success!" upon successful completion.
    """
    request_json = request.get_json()
    if not request_json.get("dry_run"):
        scene_id = request_json.get("scene_id")
        run_flags = request_json.get("run_flags")  # expects list of integers
        if not run_flags:
            run_flags = list(ASA_MAPPING.keys())
        elif any(run_flag not in ASA_MAPPING.keys() for run_flag in run_flags):
            raise ValueError(
                f"Invalid run_flag provided. {run_flags} not in {ASA_MAPPING.keys()}"
            )

        overwrite_previous = request_json.get("overwrite_previous", False)
        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1_scene = await db_client.get_scene_from_id(scene_id)
                slicks = await db_client.get_slicks_from_scene_id(scene_id)
                previous_asa = {}
                previous_collated_scores = {}
                for slick in slicks:
                    if overwrite_previous:
                        previous_asa[slick.id] = []
                        previous_collated_scores[slick.id] = []
                    else:
                        previous_asa[slick.id] = await db_client.get_previous_asa(
                            slick.id
                        )
                        previous_collated_scores[
                            slick.id
                        ] = await db_client.get_id_collated_score_pairs(slick.id)

            print(f"Running ASA ({run_flags}) on scene_id: {scene_id}")
            print(f"{len(slicks)} slicks in scene {scene_id}: {[s.id for s in slicks]}")
            if len(slicks) > 0:
                analyzers = [
                    ASA_MAPPING[source_type](s1_scene) for source_type in run_flags
                ]
                random.shuffle(slicks)  # Allows rerunning a scene to skip bugs
                for slick in slicks:
                    analyzers_to_run = [
                        analyzer
                        for analyzer in analyzers
                        if analyzer.source_type not in previous_asa[slick.id]
                    ]
                    if len(analyzers_to_run) == 0:
                        continue

                    # Convert slick geometry to GeoDataFrame
                    slick_geom = wkb.loads(str(slick.geometry)).buffer(0)
                    slick_gdf = gpd.GeoDataFrame(
                        {"geometry": [slick_geom], "centerlines": [slick.centerlines]},
                        crs="4326",
                    )
                    fresh_ranked_sources = pd.DataFrame()

                    for analyzer in analyzers_to_run:
                        res = analyzer.compute_coincidence_scores(slick_gdf)
                        fresh_ranked_sources = pd.concat(
                            [fresh_ranked_sources, res], ignore_index=True
                        )

                    print(
                        f"{len(fresh_ranked_sources)} sources found for Slick ID: {slick.id}"
                    )
                    if len(fresh_ranked_sources) > 0:
                        old_ranked_sources = pd.DataFrame(
                            previous_collated_scores[slick.id],
                            columns=["slick_to_source_id", "collated_score"],
                        )
                        combined_df = pd.concat(
                            [old_ranked_sources, fresh_ranked_sources],
                            ignore_index=True,
                        )
                        combined_df.sort_values(
                            "collated_score", ascending=False, inplace=True
                        )
                        combined_df.reset_index(drop=True, inplace=True)
                        combined_df["rank"] = combined_df.index + 1

                        # Might want to increase this to save the top 3 per Source Type
                        only_record_top = 2 * len(ASA_MAPPING)

                        async with db_client.session.begin():
                            if overwrite_previous:
                                print(f"Deactivating sources for slick {slick.id}")
                                await db_client.deactivate_sources_for_slick(slick.id)
                            for idx, source_row in combined_df.iterrows():
                                if pd.isna(source_row["slick_to_source_id"]):
                                    # Insert slick to source association
                                    if idx < only_record_top:
                                        source = (
                                            await db_client.get_or_insert_ranked_source(
                                                source_row
                                            )
                                        )

                                        await db_client.insert_slick_to_source(
                                            source=source.id,
                                            slick=slick.id,
                                            active=True,
                                            git_hash=os.getenv("GIT_HASH"),
                                            git_tag=os.getenv("GIT_TAG"),
                                            coincidence_score=source_row[
                                                "coincidence_score"
                                            ],
                                            collated_score=source_row["collated_score"],
                                            rank=source_row["rank"],
                                            geojson_fc=source_row.get("geojson_fc"),
                                            geometry=(
                                                source_row["geometry"]
                                                if isinstance(
                                                    source_row["geometry"], str
                                                )
                                                else source_row["geometry"].wkt
                                                # XXX HACK TODO Need to figure out WHY this is a string 95% of the time, but then sometimes a shapely.point and sometimes a shapely.linestring
                                            ),
                                        )
                                else:
                                    await db_client.update_slick_to_source(
                                        filter_kwargs={
                                            "id": source_row["slick_to_source_id"]
                                        },
                                        update_kwargs={
                                            "rank": source_row["rank"],
                                            "active": idx < only_record_top,
                                        },
                                    )
        # Dispose the engine after finishing all DB operations.
        await db_engine.dispose()

    return "Success!"
