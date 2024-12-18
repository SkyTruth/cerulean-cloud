"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""

import gc
import json
import logging
import os
import sys
import urllib.parse as urlparse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import geopandas as gpd
import morecantile
import numpy as np
import supermercado
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from global_land_mask import globe
from shapely.geometry import shape

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_function_ais_analysis.queuer import add_to_asa_queue
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.schema import (
    OrchestratorInput,
    OrchestratorResult,
)
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.models import get_model
from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient
from cerulean_cloud.tiling import TMS, offset_bounds_from_base_tiles
from cerulean_cloud.titiler_client import TitilerClient

# Configure logger
logger = logging.getLogger("orchestrate")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def structured_log(message, **kwargs):
    """
    Create a structured log message in JSON format.

    Args:
        message (str): The main log message.
        **kwargs: Arbitrary keyword arguments representing additional log details.

    Returns:
        str: A JSON-formatted string containing the log message and metadata.
    """
    log_data = {"message": message}
    log_data.update(kwargs)
    return json.dumps(log_data)


def get_landmask_gdf():
    """
    Retrieves the GeoDataFrame representing the land mask.
    Returns:
        GeoDataFrame: The GeoDataFrame object representing the land mask, with CRS set to "EPSG:3857".
    """
    mask_path = "/app/cerulean_cloud/cloud_run_orchestrator/gadmLandMask_simplified/gadmLandMask_simplified.shp"
    landmask_gdf = gpd.read_file(mask_path).set_crs("4326")
    return landmask_gdf


def make_cloud_log_url(
    cloud_run_name: str, start_time, project_id: str, duration=2
) -> str:
    """Forges a cloud log url given a service name, a date and a project id

    Args:
        cloud_run_name (str): the cloud run service name
        start_time (datetime.datetime): the start time of the cloud run
        project_id (str): a project id
        duration (int, optional): Default duration of the logs to show (in min). Defaults to 2.

    Returns:
        str: A cloud log url.
    """
    base_url = "https://console.cloud.google.com/logs/query;query="
    query = f'resource.type = "cloud_run_revision" resource.labels.service_name = "{cloud_run_name}"'
    formatted_time = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_time = (start_time + timedelta(minutes=duration)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    t_range = f";timeRange={formatted_time}%2F{end_time}"
    t = f";cursorTimestamp={formatted_time}"
    project_id = f"?project={project_id}"
    url = base_url + urlparse.quote(query) + t_range + t + project_id
    return url


def offset_group_shape_from_base_tiles(
    tiles: List[morecantile.Tile], scale=2
) -> Tuple[int, int]:
    """from a list of tiles, get the expected shape of the image (of offset tiles, +1)"""
    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles])
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )
    hw = scale * 256
    width = (tilexmax - tilexmin + 2) * hw
    height = (tileymax - tileymin + 2) * hw

    return height, width


def group_bounds_from_list_of_bounds(bounds: List[List[float]]) -> List[float]:
    """from a list of bounds, get the merged bounds (min max)"""
    bounds_np = np.array([(b[0], b[1], b[2], b[3]) for b in bounds])
    minx, miny, maxx, maxy = (
        np.min(bounds_np[:, 0]),
        np.min(bounds_np[:, 1]),
        np.max(bounds_np[:, 2]),
        np.max(bounds_np[:, 3]),
    )
    return list((minx, miny, maxx, maxy))


def get_tiler():
    """get tiler"""
    return TMS


def get_titiler_client():
    """get titiler client"""
    return TitilerClient(url=os.getenv("TITILER_URL"))


def get_roda_sentinelhub_client():
    """get roda sentinelhub client"""
    return RodaSentinelHubClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """create an engine that persists for the lifetime of the app"""
    app.state.database_engine = get_engine()
    try:
        yield
    finally:
        await app.state.database_engine.dispose()


app = FastAPI(
    title="Cloud Run orchestrator",
    dependencies=[Depends(api_key_auth)],
    lifespan=lifespan,
)
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


@app.post(
    "/orchestrate",
    description="Run orchestration",
    tags=["Run orchestration"],
    response_model=OrchestratorResult,
)
async def orchestrate(
    payload: OrchestratorInput,
    request: Request,
    tiler=Depends(get_tiler),
    titiler_client=Depends(get_titiler_client),
    roda_sentinelhub_client=Depends(get_roda_sentinelhub_client),
) -> Dict:
    """orchestrate"""
    db_engine = request.app.state.database_engine
    try:
        return await _orchestrate(
            payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
        )
    except DatabaseError as db_err:
        # Handle database-related errors
        logger.exception(
            structured_log(
                "Database error during orchestration",
                scene_id=payload.sceneid,
                exception=str(db_err),
            )
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred while processing your request.",
        ) from db_err
    except ValidationError as val_err:
        # Handle payload validation errors
        logger.exception(
            structured_log(
                "Validation error", scene_id=payload.sceneid, exception=str(val_err)
            )
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input data provided.",
        ) from val_err
    except InferenceError as inf_err:
        # Handle inference-related errors
        logger.exception(
            structured_log(
                "Inference error", scene_id=payload.sceneid, exception=str(inf_err)
            )
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="An error occurred during the inference process.",
        ) from inf_err
    except Exception as e:
        # Handle unexpected errors
        logger.exception(
            structured_log(
                "Unexpected error during orchestration",
                scene_id=payload.sceneid,
                exception=str(e),
            )
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
        ) from e


def is_tile_over_water(tile_bounds: List[float]) -> bool:
    """are the tile bounds over water"""
    minx, miny, maxx, maxy = tile_bounds
    return any(globe.is_ocean([miny, maxy], [minx, maxx]))


async def _orchestrate(
    payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
):

    # Orchestrate inference
    start_time = datetime.now()

    logger.info(
        structured_log(
            "Initiating database client",
            scene_id=payload.sceneid,
            start_time=start_time.isoformat(),
        )
    )

    # WARNING: until this is resolved https://github.com/cogeotiff/rio-tiler-pds/issues/77
    # When scene traverses the anti-meridian, scene_bounds are nonsensical
    # Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7 >>> [-180.0, 61.06949078480844, 180.0, 62.88226850489882]
    try:
        scene_bounds = await titiler_client.get_bounds(payload.sceneid)
        scene_stats = await titiler_client.get_statistics(payload.sceneid, band="vv")
        scene_info = await roda_sentinelhub_client.get_product_info(payload.sceneid)
    except Exception as e:
        logger.error(
            structured_log(
                "TiTiler client error", scene_id=payload.sceneid, exception=str(e)
            )
        )
        return OrchestratorResult(status=str(e))

    try:
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                db_model = await db_client.get_db_model(os.getenv("MODEL"))
                model_dict = {
                    column.name: (
                        getattr(db_model, column.name).isoformat()
                        if isinstance(getattr(db_model, column.name), datetime)
                        else getattr(db_model, column.name)
                    )
                    for column in db_model.__table__.columns
                }
    except Exception as e:
        return OrchestratorResult(status=str(e))

    zoom = payload.zoom or model_dict["zoom_level"]
    scale = payload.scale or model_dict["scale"]

    if model_dict["zoom_level"] != zoom:
        logger.warning(
            structured_log(
                f"Model was trained on zoom level {model_dict['zoom_level']} but is being run on {zoom}",
                scene_id=payload.sceneid,
            )
        )
    if model_dict["tile_width_px"] != scale * 256:
        logger.warning(
            structured_log(
                f"Model was trained on image tile of resolution {model_dict['tile_width_px']} but is being run on {scale*256}",
                scene_id=payload.sceneid,
            )
        )

    logger.info(
        structured_log(
            f"Generating tilesets (zoom: {zoom} scale: {scale}, scene_bounds: {scene_bounds}, scene_stats: {scene_stats}, : scene_info: {scene_info})",
            scene_id=payload.sceneid,
        )
    )

    base_tiles = list(tiler.tiles(*scene_bounds, [zoom], truncate=False))
    n_basetiles = len(base_tiles)

    offset_amounts = [0.0, 0.33, 0.66]
    tileset_list = [
        offset_bounds_from_base_tiles(base_tiles, offset_amount=o_a)
        for o_a in offset_amounts
    ]

    n_offsettiles = len(tileset_list[0])
    tileset_hw_pixels = offset_group_shape_from_base_tiles(base_tiles, scale=scale)
    tileset_envelope_bounds = group_bounds_from_list_of_bounds(tileset_list[0])

    # Filter out land tiles
    logger.info(
        structured_log(
            f"Removing invalid tiles - tileset contains {n_offsettiles} tiles before landfilter",
            scene_id=payload.sceneid,
        )
    )
    try:
        tileset_list = [
            [b for b in tileset if is_tile_over_water(b)] for tileset in tileset_list
        ]
    except ValueError as e:
        # XXX BUG is_tile_over_water throws ValueError if the scene crosses or is close to the antimeridian. Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7
        logger.exception(
            structured_log(
                "FAILURE - scene touches antimeridian, and is_tile_over_water() failed!",
                scene_id=payload.sceneid,
            )
        )
        return OrchestratorResult(status=str(e))

    if not any(set for set in tileset_list):
        # There are actually no tiles to be processed! This is because the scene relevancy ocean mask is coarser than globe.is_ocean().
        # WARNING this will return success, but there will be not trace in the DB of your request (i.e. in S1 or Orchestrator tables)
        # XXX TODO
        logger.info(
            structured_log(
                "NO TILES TO BE PROCESSED (SUCCESS)", scene_id=payload.sceneid
            )
        )
        return OrchestratorResult(status="Success (no oceanic tiles)")

    if payload.dry_run:
        # Only tests code above this point, without actually adding any new data to the database, or running inference.
        logger.warning(structured_log("DRY RUN (SUCCESS)", scene_id=payload.scene_id))
        return OrchestratorResult(status="Success (dry run)")

    # write to DB
    try:
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                trigger = await db_client.get_trigger(trigger=payload.trigger)
                layers = [
                    await db_client.get_layer(layer) for layer in model_dict["layers"]
                ]
                sentinel1_grd = await db_client.get_sentinel1_grd(
                    payload.sceneid,
                    scene_info,
                    titiler_client.get_base_tile_url(
                        payload.sceneid,
                        rescale=(0, 255),
                    ),
                )
                stale_slick_count = (
                    await db_client.deactivate_stale_slicks_from_scene_id(
                        payload.sceneid
                    )
                )
                logger.info(
                    structured_log(
                        f"{start_time}: Deactivating {stale_slick_count} slicks from stale runs.",
                        scene_id=payload.sceneid,
                    )
                )
                orchestrator_run = await db_client.add_orchestrator(
                    start_time,
                    start_time,
                    n_basetiles,
                    n_offsettiles,
                    os.getenv("GIT_HASH"),
                    os.getenv("GIT_TAG"),
                    make_cloud_log_url(
                        os.getenv("CLOUD_RUN_NAME"),
                        start_time,
                        os.getenv("PROJECT_ID"),
                    ),
                    zoom,
                    scale,
                    scene_bounds,
                    trigger,
                    db_model,
                    sentinel1_grd,
                )
                orchestrator_run_id = orchestrator_run.id
    except Exception as e:
        logging.error("Failed to write to DB")
        return OrchestratorResult(status=str(e))

    success = True
    try:
        logger.info(
            structured_log("Instantiating inference client", scene_id=payload.sceneid)
        )
        cloud_run_inference = CloudRunInferenceClient(
            url=os.getenv("INFERENCE_URL"),
            titiler_client=titiler_client,
            sceneid=payload.sceneid,
            tileset_envelope_bounds=tileset_envelope_bounds,
            image_hw_pixels=tileset_hw_pixels,
            layers=layers,
            scale=scale,
            model_dict=model_dict,
        )

        # Perform inferences
        n_offsettiles_after = len(tileset_list[0])
        logger.info(
            structured_log(
                f"Inference starting - {n_offsettiles_after} tileset lists",
                scene_id=payload.sceneid,
            )
        )
        tileset_results_list = [
            await cloud_run_inference.run_parallel_inference(tileset)
            for tileset in tileset_list
        ]

        # Stitch inferences
        logger.info(structured_log("Stitching result", scene_id=payload.sceneid))
        model = get_model(model_dict)
        tileset_fc_list = [
            model.postprocess_tileset(
                tileset_results, [[b] for b in tileset_bounds]
            )  # extra square brackets needed because each stack only has one tile in it for now XXX HACK
            for (tileset_results, tileset_bounds) in zip(
                tileset_results_list, tileset_list
            )
        ]

        # Ensemble inferences
        logger.info(structured_log("Ensembling results", scene_id=payload.sceneid))
        final_ensemble = model.nms_feature_reduction(
            features=tileset_fc_list, min_overlaps_to_keep=1
        )

        LAND_MASK_BUFFER_M = 1000
        logger.info(
            structured_log(
                f"Removing all slicks within {LAND_MASK_BUFFER_M}m of land",
                scene_id=payload.sceneid,
            )
        )
        landmask_gdf = get_landmask_gdf()
        for feat in final_ensemble.get("features"):
            buffered_gdf = gpd.GeoDataFrame(
                geometry=[shape(feat["geometry"])], crs="4326"
            )
            crs_meters = buffered_gdf.estimate_utm_crs(datum_name="WGS 84")
            buffered_gdf["geometry"] = (
                buffered_gdf.to_crs(crs_meters)
                .buffer(LAND_MASK_BUFFER_M)
                .to_crs("4326")
            )
            intersecting_land = gpd.sjoin(
                landmask_gdf,
                buffered_gdf,
                how="inner",
                predicate="intersects",
            )
            if not intersecting_land.empty:
                feat["properties"]["inf_idx"] = model.background_class_idx

            del buffered_gdf, crs_meters, intersecting_land
            gc.collect()  # Force garbage collection

        # Removed all preprocessing of features from within the
        # database session to avoid holding locks on the
        # table while performing un-related calculations.
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                for feat in final_ensemble.get("features"):
                    slick = await db_client.add_slick(
                        orchestrator_run,
                        sentinel1_grd.start_time,
                        feat.get("geometry"),
                        feat.get("properties").get("inf_idx"),
                        feat.get("properties").get("machine_confidence"),
                    )
                    logger.info(
                        structured_log(
                            "Added slick", scene_id=payload.sceneid, slick=str(slick)
                        )
                    )

                logger.info(
                    structured_log(
                        "Queueing up Automatic AIS Analysis", scene_id=payload.sceneid
                    )
                )
                add_to_asa_queue(sentinel1_grd.scene_id)

        del (
            final_ensemble,
            tileset_fc_list,
            tileset_results_list,
            tileset_list,
            landmask_gdf,
        )
        gc.collect()  # Force garbage collection

    except Exception as e:
        success = False
        exc = e
        logger.exception(
            structured_log(
                "Error processing scene", scene_id=payload.sceneid, exception=str(e)
            )
        )
    async with DatabaseClient(db_engine) as db_client:
        async with db_client.session.begin():
            or_refreshed = await db_client.get_orchestrator(orchestrator_run_id)

            end_time = datetime.now()
            or_refreshed.success = success
            or_refreshed.inference_end_time = end_time
            logger.info(
                structured_log(
                    f"{start_time}: End time: {end_time}; Orchestration success: {success}",
                    scene_id=payload.sceneid,
                )
            )
    if success is False:
        raise exc
    return OrchestratorResult(status="Success")


class DatabaseError(Exception):
    """DatabaseError"""

    pass


class ValidationError(Exception):
    """ValidationError"""

    pass


class InferenceError(Exception):
    """InferenceError"""

    pass
