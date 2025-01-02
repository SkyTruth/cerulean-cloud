"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""

import json
import logging
import os
import sys
import traceback
import urllib.parse as urlparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import geopandas as gpd
import morecantile
import numpy as np
import supermercado
from fastapi import Depends, FastAPI
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
from cerulean_cloud.cloud_run_orchestrator.utils import structured_log
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

app = FastAPI(title="Cloud Run orchestrator", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

landmask_gdf = None


def get_landmask_gdf():
    """
    Retrieves the GeoDataFrame representing the land mask.
    This function uses lazy initialization to load the land mask data from a .shp file
    only upon the first call. Subsequent calls return the stored GeoDataFrame.
    Returns:
        GeoDataFrame: The GeoDataFrame object representing the land mask, with CRS set to "EPSG:3857".
    """
    global landmask_gdf
    if landmask_gdf is None:
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


def get_database_engine():
    """get database engine"""
    return get_engine(db_url=os.getenv("DB_URL"))


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
    tiler=Depends(get_tiler),
    titiler_client=Depends(get_titiler_client),
    roda_sentinelhub_client=Depends(get_roda_sentinelhub_client),
    db_engine=Depends(get_database_engine),
) -> Dict:
    """orchestrate"""
    return await _orchestrate(
        payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
    )


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
            "Initiating Orchestrator",
            severity="INFO",
            scene_id=payload.sceneid,
            start_time=start_time.isoformat() + "Z",
        )
    )

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
    zoom = payload.zoom or model_dict["zoom_level"]
    scale = payload.scale or model_dict["scale"]

    if model_dict["zoom_level"] != zoom:
        logger.warning(
            structured_log(
                "Model zoom level warning",
                severity="WARNING",
                scene_id=payload.sceneid,
                training_zoom=model_dict["zoom_level"],
                zoom=zoom,
            )
        )
    if model_dict["tile_width_px"] != scale * 256:
        logger.warning(
            structured_log(
                "Model resolution warning",
                severity="WARNING",
                scene_id=payload.sceneid,
                training_resolution=model_dict["tile_width_px"],
                resolution=scale * 256,
            )
        )

    scene_bounds = await titiler_client.get_bounds(payload.sceneid)
    scene_stats = await titiler_client.get_statistics(payload.sceneid, band="vv")
    scene_info = await roda_sentinelhub_client.get_product_info(payload.sceneid)

    logger.info(
        structured_log(
            "Generating tilesets",
            severity="INFO",
            scene_id=payload.sceneid,
            zoom=zoom,
            scale=scale,
            scene_bounds=json.dumps(scene_bounds),
            scene_stats=json.dumps(scene_stats),
            scene_info=json.dumps(scene_info),
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

    logger.info(
        structured_log(
            "Removing invalid tiles (land filter)",
            severity="INFO",
            scene_id=payload.sceneid,
            n_tiles=n_offsettiles,
        )
    )

    # Filter out land tiles
    try:
        tileset_list = [
            [b for b in tileset if is_tile_over_water(b)] for tileset in tileset_list
        ]
        logger.info(
            structured_log(
                "Removed tiles over land",
                severity="INFO",
                scene_id=payload.sceneid,
                n_tilesets=len(tileset_list[0]),
            )
        )
    except ValueError as e:
        # XXX BUG is_tile_over_water throws ValueError if the scene crosses or is close to the antimeridian. Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7

        logger.error(
            structured_log(
                "FAILURE - scene touches antimeridian, and is_tile_over_water() failed!",
                severity="ERROR",
                scene_id=payload.sceneid,
                traceback=traceback.format_exc(),
            )
        )
        return OrchestratorResult(status=str(e))

    if not any(set for set in tileset_list):
        # There are actually no tiles to be processed! This is because the scene relevancy ocean mask is coarser than globe.is_ocean().
        # WARNING this will return success, but there will be not trace in the DB of your request (i.e. in S1 or Orchestrator tables)
        # XXX TODO
        logger.info(
            structured_log(
                "NO TILES TO BE PROCESSED (SUCCESS)",
                severity="INFO",
                scene_id=payload.sceneid,
            )
        )
        return OrchestratorResult(status="Success (no oceanic tiles)")

    if payload.dry_run:
        # Only tests code above this point, without actually adding any new data to the database, or running inference.
        logger.info(
            structured_log(
                "DRY RUN (SUCCESS)", severity="INFO", scene_id=payload.scene_id
            )
        )
        return OrchestratorResult(status="Success (dry run)")

    # write to DB
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
            stale_slick_count = await db_client.deactivate_stale_slicks_from_scene_id(
                payload.sceneid
            )
            logger.info(
                structured_log(
                    "Deactivating slicks from stale runs.",
                    severity="INFO",
                    scene_id=payload.sceneid,
                    n_stale_slicks=stale_slick_count,
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

        success = True
        try:
            logger.info(
                structured_log(
                    "Instantiating inference client",
                    severity="INFO",
                    scene_id=payload.sceneid,
                )
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
            logger.info(
                structured_log(
                    "Inference starting",
                    severity="INFO",
                    scene_id=payload.sceneid,
                )
            )
            tileset_results_list = [
                await cloud_run_inference.run_parallel_inference(tileset)
                for tileset in tileset_list
            ]

            # Stitch inferences
            logger.info(
                structured_log(
                    "Initializing model",
                    severity="INFO",
                    scene_id=payload.sceneid,
                    model_type=model_dict["type"],
                )
            )
            model = get_model(model_dict)

            logger.info(
                structured_log(
                    "Stitching result", severity="INFO", scene_id=payload.sceneid
                )
            )
            tileset_fc_list = []
            for tileset_results, tileset_bounds in zip(
                tileset_results_list, tileset_list
            ):
                if tileset_results and tileset_bounds:
                    fc = model.postprocess_tileset(
                        tileset_results, [[b] for b in tileset_bounds]
                    )  # extra square brackets needed because each stack only has one tile in it for now XXX HACK
                    tileset_fc_list.append(fc)

            # Ensemble inferences
            logger.info(
                structured_log(
                    "Ensembling results", severity="INFO", scene_id=payload.sceneid
                )
            )
            final_ensemble = model.nms_feature_reduction(
                features=tileset_fc_list, min_overlaps_to_keep=1
            )
            features = final_ensemble.get("features", [])
            n_feats = len(features)

            n_background = 0
            if final_ensemble.get("features"):
                LAND_MASK_BUFFER_M = 1000
                logger.info(
                    structured_log(
                        "Removing all slicks near land",
                        severity="INFO",
                        scene_id=payload.sceneid,
                        n_features=n_feats,
                        land_buffer_m=LAND_MASK_BUFFER_M,
                    )
                )
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
                        get_landmask_gdf(),
                        buffered_gdf,
                        how="inner",
                        predicate="intersects",
                    )
                    if not intersecting_land.empty:
                        feat["properties"]["inf_idx"] = model.background_class_idx
                        n_background += 1

                logger.info(
                    structured_log(
                        "Adding slicks to database",
                        severity="INFO",
                        scene_id=payload.sceneid,
                        land_buffer_m=LAND_MASK_BUFFER_M,
                        n_background_slicks=n_background,
                        n_slicks=len(features) - n_background,
                    )
                )
                # Removed all preprocessing of features from within the
                # database session to avoid holidng locks on the
                # table while performing un-related calculations.
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
                                "Added slick",
                                severity="INFO",
                                scene_id=payload.sceneid,
                                slick=slick.id,  # TODO: this is null - is there a slick attribute to use?
                            )
                        )

                logger.info(
                    structured_log(
                        "Queueing up Automatic AIS Analysis",
                        severity="INFO",
                        scene_id=payload.sceneid,
                    )
                )
                add_to_asa_queue(sentinel1_grd.scene_id)

        except Exception as e:
            success = False
            exc = e
            logger.error(
                structured_log(
                    "Failed to process inference on scene",
                    severity="ERROR",
                    scene_id=payload.sceneid,
                    exception=str(e),
                    traceback=traceback.format_exc(),
                )
            )
        async with db_client.session.begin():
            end_time = datetime.now()
            orchestrator_run.success = success
            orchestrator_run.inference_end_time = end_time
        if success is False:
            raise exc

    dt = (end_time - start_time).total_seconds() / 60
    logger.info(
        structured_log(
            "Orchestration complete!",
            severity="INFO",
            timestamp=end_time.isoformat() + "Z",
            scene_id=payload.sceneid,
            success=success,
            duration_minutes=dt,
        )
    )
    return OrchestratorResult(status="Success")
