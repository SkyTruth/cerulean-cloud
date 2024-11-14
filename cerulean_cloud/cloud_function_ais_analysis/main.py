"""cloud function AIS analysis handler
"""

import asyncio
import os

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
    res = loop.run_until_complete(handle_asa_request(request))
    return res


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
        run_flags = request_json.get("run_flags", ["ais", "infra", "dark"])
        overwrite_previous = request_json.get("overwrite_previous", False)
        print(f"Running ASA ({run_flags}) on scene_id: {scene_id}")
        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1_scene = await db_client.get_scene_from_id(scene_id)
                slicks = await db_client.get_slicks_from_scene_id(
                    scene_id, with_sources=overwrite_previous
                )

            print(f"# Slicks found: {len(slicks)}")
            if len(slicks) > 0:
                analyzers = [ASA_MAPPING[source](s1_scene) for source in run_flags]
                for slick in slicks:
                    # Convert slick geometry to GeoDataFrame
                    slick_geom = wkb.loads(str(slick.geometry)).buffer(0)
                    slick_gdf = gpd.GeoDataFrame({"geometry": [slick_geom]}, crs="4326")
                    ranked_sources = pd.DataFrame()

                    for analyzer in analyzers:
                        res = analyzer.compute_coincidence_scores(slick_gdf)
                        ranked_sources = pd.concat(
                            [ranked_sources, res], ignore_index=True
                        )

                    print(
                        f"{len(ranked_sources)} sources found for Slick ID: {slick.id}"
                    )
                    if len(ranked_sources) > 0:
                        ranked_sources = ranked_sources.sort_values(
                            "coincidence_score", ascending=False
                        ).reset_index(drop=True)
                        async with db_client.session.begin():
                            for idx, source_row in ranked_sources.iloc[:5].iterrows():
                                # Only record the the top 5 ranked sources

                                if source_row.get("source_type") == "ais":
                                    source = (
                                        await db_client.get_or_insert_source_from_ssvid(
                                            ssvid=source_row["st_name"],
                                            shipname=source_row.get("shipname"),
                                            shiptype=source_row.get("shiptype"),
                                        )
                                    )
                                elif source_row.get("source_type") == "infra":
                                    source = await db_client.get_or_insert_infra_source(
                                        infra_id=source_row["infra_id"],
                                        infra_name=source_row.get("infra_name"),
                                    )
                                elif source_row.get("source_type") == "dark":
                                    raise NotImplementedError(
                                        "Dark vessel source not implemented"
                                    )
                                else:
                                    raise ValueError(
                                        f"Unknown source type: {source_row.get('source_type')}"
                                    )

                                await db_client.session.flush()

                                # Insert slick to source association
                                await db_client.insert_slick_to_source(
                                    source=source.id,
                                    slick=slick.id,
                                    coincidence_score=source_row["coincidence_score"],
                                    rank=idx + 1,
                                    geojson_fc=source_row.get("geojson_fc"),
                                    geometry=source_row["geometry"].wkt,
                                )

    return "Success!"
