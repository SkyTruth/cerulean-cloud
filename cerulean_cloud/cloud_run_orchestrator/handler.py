"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""

import json
import os
import random
import signal
import time
import traceback
import urllib.parse as urlparse
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import geopandas as gpd
import httpx
import morecantile
import numpy as np
import supermercado
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from global_land_mask import globe
from shapely.geometry import shape

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.centerlines import calculate_centerlines
from cerulean_cloud.cloud_function_asa.queuer import add_to_asa_queue
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.schema import (
    OrchestratorInput,
    OrchestratorResult,
)
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.models import get_model
from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient
from cerulean_cloud.structured_logger import (
    configure_structured_logger,
    context_dict_var,
)
from cerulean_cloud.tiling import TMS, offset_bounds_from_base_tiles
from cerulean_cloud.titiler_client import TitilerClient

# Configure logger
logger = configure_structured_logger("cerulean_cloud")


app = FastAPI(title="Cloud Run orchestrator", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

landmask_gdf = None


def flush_logs():
    """
    Flush all logger handlers.
    """
    for handler in logger.handlers:
        handler.flush()


def handle_sigterm(signum, frame):
    """
    Handle the SIGTERM signal.
    """
    logger.error(
        {
            "message": "SIGTERM signal received",
            "file_name": frame.f_code.co_filename,
            "line_number": frame.f_lineno,
        }
    )
    flush_logs()


# Register SIGTERM handler
signal.signal(signal.SIGTERM, handle_sigterm)


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


def download_sea_ice_gdf(date: date) -> gpd.GeoDataFrame:
    """
    Download and load a MASIE sea ice mask GeoDataFrame from a zipped shapefile archive.
    """
    archive_url = (
        "https://noaadata.apps.nsidc.org/NOAA/G02186/shapefiles/4km/"
        f"{date.year}/masie_ice_r00_v01_{date:%Y%j}_4km.zip"
    )

    with TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "masie_4km.zip"

        with httpx.stream(
            "GET", archive_url, timeout=60, follow_redirects=True
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

        with ZipFile(archive_path) as archive:
            shp_names = [n for n in archive.namelist() if n.lower().endswith(".shp")]
            if len(shp_names) != 1:
                raise ValueError(f"Expected one shapefile, found {len(shp_names)}")
            gdf = gpd.read_file(f"zip://{archive_path}!{shp_names[0]}")

    return gdf


def is_retryable_sea_ice_mask_error(exc: Exception) -> bool:
    """
    Return whether an upstream sea-ice mask fetch failure is worth retrying.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, httpx.TransportError)


def get_retry_sleep_seconds(attempt: int, retry_backoff_seconds: float) -> float:
    """
    Return exponential backoff with jitter for retryable sea-ice mask failures.
    """
    exponential_backoff_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
    jitter_seconds = random.uniform(0, exponential_backoff_seconds)
    return exponential_backoff_seconds + jitter_seconds


def get_sea_ice_mask(
    scene_date: date,
    earliest_supported_date: date = date(2010, 1, 1),
    max_attempts_per_date: int = 5,
    retry_backoff_seconds: float = 0.5,
) -> Tuple[Optional[gpd.GeoDataFrame], Optional[date]]:
    """
    Return the most recent NOAA MASIE sea ice mask on or before scene_date.
    """
    current_date = scene_date

    while current_date >= earliest_supported_date:
        for attempt in range(1, max_attempts_per_date + 1):
            try:
                gdf = download_sea_ice_gdf(current_date)
                return gdf, current_date
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    current_date -= timedelta(days=1)
                    break
                if not is_retryable_sea_ice_mask_error(exc):
                    raise

                if attempt == max_attempts_per_date:
                    logger.warning(
                        {
                            "message": "Skipping sea ice mask after retryable upstream failures",
                            "mask_date": current_date.isoformat(),
                            "status_code": exc.response.status_code,
                            "attempts": attempt,
                        }
                    )
                    return None, None

                sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
                logger.warning(
                    {
                        "message": "Retrying sea ice mask download after upstream HTTP failure",
                        "mask_date": current_date.isoformat(),
                        "status_code": exc.response.status_code,
                        "attempt": attempt,
                        "sleep_seconds": sleep_seconds,
                    }
                )
                time.sleep(sleep_seconds)
            except httpx.TransportError as exc:
                if not is_retryable_sea_ice_mask_error(exc):
                    raise

                if attempt == max_attempts_per_date:
                    logger.warning(
                        {
                            "message": "Skipping sea ice mask after retryable transport failures",
                            "mask_date": current_date.isoformat(),
                            "error_type": type(exc).__name__,
                            "attempts": attempt,
                        }
                    )
                    return None, None

                sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
                logger.warning(
                    {
                        "message": "Retrying sea ice mask download after transport failure",
                        "mask_date": current_date.isoformat(),
                        "error_type": type(exc).__name__,
                        "attempt": attempt,
                        "sleep_seconds": sleep_seconds,
                    }
                )
                time.sleep(sleep_seconds)

    return None, None


def classify_geometry_background_mask(
    buffered_geometry,
    sea_ice_mask_gdf: Optional[gpd.GeoDataFrame],
    land_mask_gdf: gpd.GeoDataFrame,
) -> Optional[str]:
    """
    Returns the background class short name for a buffered slick, if any.
    """
    if len(land_mask_gdf.sindex.query(buffered_geometry, predicate="intersects")) > 0:
        return "LAND"
    if sea_ice_mask_gdf is not None and not sea_ice_mask_gdf.empty:
        sea_ice_query_geometry = buffered_geometry
        if sea_ice_mask_gdf.crs and sea_ice_mask_gdf.crs.to_epsg() != 4326:
            sea_ice_query_geometry = (
                gpd.GeoSeries([buffered_geometry], crs="EPSG:4326")
                .to_crs(sea_ice_mask_gdf.crs)
                .iloc[0]
            )
        if (
            len(
                sea_ice_mask_gdf.sindex.query(
                    sea_ice_query_geometry, predicate="intersects"
                )
            )
            > 0
        ):
            return "SEA_ICE"
    return None


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
    context_dict_var.set({"scene_id": payload.scene_id})  # Set the scene_id for logging

    # Orchestrate inference
    start_time = datetime.now()
    logger.info(
        {
            "message": "Initiating Orchestrator",
            "start_time": start_time.isoformat() + "Z",
        }
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
            {
                "message": "Model zoom level warning",
                "description": f"Model was trained on zoom level {model_dict['zoom_level']} but is being run on {zoom}",
            }
        )
    if model_dict["tile_width_px"] != scale * 256:
        logger.warning(
            {
                "message": "Model resolution warning",
                "description": f"Model was trained on image tile of resolution {model_dict['tile_width_px']} but is being run on {scale * 256}",
            }
        )

    # WARNING: until this is resolved https://github.com/cogeotiff/rio-tiler-pds/issues/77
    # When scene traverses the anti-meridian, scene_bounds are nonsensical
    # Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7 >>> [-180.0, 61.06949078480844, 180.0, 62.88226850489882]
    logger.info("Getting scene bounds")
    scene_bounds = await titiler_client.get_bounds(payload.scene_id)

    logger.info("Getting scene statistics")
    scene_stats = await titiler_client.get_statistics(payload.scene_id, band="vv")

    logger.info("Getting SentinalHUB product info")
    scene_info = await roda_sentinelhub_client.get_product_info(payload.scene_id)

    logger.info(
        {
            "message": "Generating tilesets",
            "zoom": zoom,
            "scale": scale,
            "scene_bounds": json.dumps(scene_bounds),
            "scene_stats": json.dumps(scene_stats),
            "scene_info": json.dumps(scene_info),
        }
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
        {
            "message": "Removing invalid tiles (land filter)",
            "n_tiles": n_offsettiles,
        }
    )

    # Filter out land tiles
    try:
        tileset_list = [
            [b for b in tileset if is_tile_over_water(b)] for tileset in tileset_list
        ]
        logger.info(
            {
                "message": "Removed tiles over land",
                "n_tilesets": len(tileset_list[0]),
            }
        )
    except ValueError as e:
        # XXX BUG is_tile_over_water throws ValueError if the scene crosses or is close to the antimeridian. Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7

        logger.error(
            {
                "message": "FAILURE - scene touches antimeridian, and is_tile_over_water() failed!",
                "traceback": traceback.format_exc(),
            }
        )
        return OrchestratorResult(status=str(e))

    if not any(set for set in tileset_list):
        # There are actually no tiles to be processed! This is because the scene relevancy ocean mask is coarser than globe.is_ocean().
        # WARNING this will return success, but there will be not trace in the DB of your request (i.e. in S1 or Orchestrator tables)
        # XXX TODO
        logger.info(
            "IS_OCEAN() MISMATCH: NO TILES TO BE PROCESSED (SUCCESS BUT NO DB ENTRY)"
        )
        return OrchestratorResult(status="Success (no oceanic tiles)")

    if payload.dry_run:
        # Only tests code above this point, without actually adding any new data to the database, or running inference.
        logger.info("DRY RUN (SUCCESS)")
        return OrchestratorResult(status="Success (dry run)")

    # write to DB
    async with DatabaseClient(db_engine) as db_client:
        async with db_client.session.begin():
            not_oil_cls_ids = await db_client.get_cls_subtree_ids("NOT_OIL")
            trigger = await db_client.get_trigger(trigger=payload.trigger)
            layers = [
                await db_client.get_layer(layer) for layer in model_dict["layers"]
            ]
            sentinel1_grd = await db_client.get_or_insert_sentinel1_grd(
                payload.scene_id,
                scene_info,
                titiler_client.get_base_tile_url(
                    payload.scene_id,
                    rescale=(0, 255),
                ),
            )
            stale_slick_count = await db_client.deactivate_stale_slicks_from_scene_id(
                payload.scene_id
            )
            logger.info(
                {
                    "message": "Deactivating slicks from stale runs.",
                    "n_stale_slicks": stale_slick_count,
                }
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
        sea_ice_date = None
        try:
            logger.info("Instantiating inference client")
            cloud_run_inference = CloudRunInferenceClient(
                url=os.getenv("INFERENCE_URL"),
                titiler_client=titiler_client,
                scene_id=payload.scene_id,
                tileset_envelope_bounds=tileset_envelope_bounds,
                image_hw_pixels=tileset_hw_pixels,
                layers=layers,
                scale=scale,
                model_dict=model_dict,
            )

            # Perform inferences
            logger.info("Inference starting")
            tileset_results_list = [
                await cloud_run_inference.run_parallel_inference(tileset)
                for tileset in tileset_list
            ]

            # Stitch inferences
            logger.info(
                {
                    "message": "Initializing model",
                    "model_type": model_dict["type"],
                }
            )
            model = get_model(model_dict)

            logger.info("Stitching result")
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
            logger.info("Ensembling results")
            final_ensemble = model.nms_feature_reduction(
                features=tileset_fc_list, min_overlaps_to_keep=1
            )
            features = final_ensemble.get("features", [])
            n_features = len(features)

            if features:
                reclass_counts = Counter()

                LAND_MASK_BUFFER_M = 1000
                land_mask_gdf = get_landmask_gdf()
                sea_ice_mask_gdf, sea_ice_date = get_sea_ice_mask(
                    sentinel1_grd.start_time.date()
                )
                logger.info(
                    {
                        "message": "Classifying slicks against not-oil masks",
                        "n_features": n_features,
                        "LAND_MASK_BUFFER_M": LAND_MASK_BUFFER_M,
                        "uses_sea_ice_mask": sea_ice_mask_gdf is not None,
                        "sea_ice_mask_date": (
                            sea_ice_date.isoformat() if sea_ice_date else None
                        ),
                    }
                )
                for feat in features:
                    slick_gdf = gpd.GeoDataFrame(
                        geometry=[shape(feat["geometry"])], crs="4326"
                    )
                    crs_meters = slick_gdf.estimate_utm_crs(datum_name="WGS 84")
                    buffered_geometry = (
                        slick_gdf.to_crs(crs_meters)
                        .buffer(LAND_MASK_BUFFER_M)
                        .to_crs("4326")
                        .iloc[0]
                    )

                    reclass_name = classify_geometry_background_mask(
                        buffered_geometry,
                        sea_ice_mask_gdf=sea_ice_mask_gdf,
                        land_mask_gdf=land_mask_gdf,
                    )
                    if reclass_name:
                        reclass_counts[reclass_name] += 1
                        feat["properties"]["cls_id_override"] = not_oil_cls_ids[
                            reclass_name
                        ]

                    logger.info("Calculating centerlines")
                    centerlines_geojson, aspect_ratio_factor = calculate_centerlines(
                        slick_gdf, crs_meters
                    )
                    logger.info(
                        {
                            "message": "Centerlines calculated",
                            "aspect_ratio_factor": aspect_ratio_factor,
                            "centerlines_geojson": centerlines_geojson,
                        }
                    )
                    feat["properties"]["centerlines"] = centerlines_geojson
                    feat["properties"]["aspect_ratio_factor"] = aspect_ratio_factor

                n_not_oil_slicks = sum(reclass_counts.values())
                logger.info(
                    {
                        "message": "Adding slicks to database",
                        "LAND_MASK_BUFFER_M": LAND_MASK_BUFFER_M,
                        "n_not_oil_slicks": n_not_oil_slicks,
                        "n_land_slicks": reclass_counts["LAND"],
                        "n_sea_ice_slicks": reclass_counts["SEA_ICE"],
                        "n_slicks": n_features - n_not_oil_slicks,
                    }
                )
                del features
                # Removed all preprocessing of features from within the
                # database session to avoid holidng locks on the
                # table while performing un-related calculations.
                async with db_client.session.begin():
                    for feat in final_ensemble.get("features"):
                        _ = await db_client.add_slick(
                            orchestrator_run,
                            sentinel1_grd.start_time,
                            feat.get("geometry"),
                            feat.get("properties").get("inf_idx"),
                            feat.get("properties").get("machine_confidence"),
                            feat.get("properties").get("centerlines"),
                            feat.get("properties").get("aspect_ratio_factor"),
                            cls_id=feat.get("properties").get("cls_id_override"),
                        )
                        logger.info("Added slick")

                logger.info("Queueing up Automatic Source Association")
                add_to_asa_queue(
                    sentinel1_grd.scene_id,
                    task_suffix=f"run-{orchestrator_run.id}"
                    if payload.manual_trigger
                    else None,
                )

        except Exception as e:
            success = False
            exc = e
            logger.error(
                {
                    "message": "Failed to process inference on scene",
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        async with db_client.session.begin():
            orchestrator_run.sea_ice_date = sea_ice_date
            end_time = datetime.now()
            orchestrator_run.success = success
            orchestrator_run.inference_end_time = end_time
            logger.info(
                {
                    "message": "Orchestration complete!",
                    "timestamp": end_time.isoformat() + "Z",
                    "success": success,
                    "duration_minutes": (end_time - start_time).total_seconds() / 60,
                }
            )
        if success is False:
            raise exc
    return OrchestratorResult(status="Success")
