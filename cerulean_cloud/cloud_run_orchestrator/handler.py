"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""

import os
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
from cerulean_cloud.cloud_function_ais_analysis.queuer import add_to_aaa_queue
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
    print(f"Start time: {start_time}")
    print(f"{start_time}: Orchestrating for sceneid {payload.sceneid}")

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
    print(f"{start_time}: zoom: {zoom}")
    print(f"{start_time}: scale: {scale}")

    if model_dict["zoom_level"] != zoom:
        print(
            f"{start_time}: WARNING: Model was trained on zoom level {model_dict['zoom_level']} but is being run on {zoom}"
        )
    if model_dict["tile_width_px"] != scale * 256:
        print(
            f"{start_time}: WARNING: Model was trained on image tile of resolution {model_dict['tile_width_px']} but is being run on {scale*256}"
        )

    # WARNING: until this is resolved https://github.com/cogeotiff/rio-tiler-pds/issues/77
    # When scene traverses the anti-meridian, scene_bounds are nonsensical
    # Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7 >>> [-180.0, 61.06949078480844, 180.0, 62.88226850489882]
    scene_bounds = await titiler_client.get_bounds(payload.sceneid)
    scene_stats = await titiler_client.get_statistics(payload.sceneid, band="vv")
    scene_info = await roda_sentinelhub_client.get_product_info(payload.sceneid)
    print(f"{start_time}: scene_bounds: {scene_bounds}")
    print(f"{start_time}: scene_stats: {scene_stats}")
    print(f"{start_time}: scene_info: {scene_info}")

    base_tiles = list(tiler.tiles(*scene_bounds, [zoom], truncate=False))

    base_tiles_bounds = offset_bounds_from_base_tiles(
        base_tiles, offset_amount=0.0
    )
    offset_1_tiles_bounds = offset_bounds_from_base_tiles(
        base_tiles, offset_amount=0.33
    )
    offset_2_tiles_bounds = offset_bounds_from_base_tiles(
        base_tiles, offset_amount=0.66
    )

    offset_group_shape = offset_group_shape_from_base_tiles(base_tiles, scale=scale)
    offset_1_group_bounds = group_bounds_from_list_of_bounds(offset_1_tiles_bounds)

    print(
        f"{start_time}: Original tiles are {len(base_tiles)}, {len(offset_1_tiles_bounds)}, {len(offset_2_tiles_bounds)}"
    )

    # Filter out land tiles
    # XXXBUG is_tile_over_water throws ValueError if the scene crosses or is close to the antimeridian. Example: S1A_IW_GRDH_1SDV_20230726T183302_20230726T183327_049598_05F6CA_31E7
    # XXXBUG is_tile_over_water throws IndexError if the scene touches the Caspian sea (globe says it is NOT ocean, whereas our cloud_function_scene_relevancy says it is). Example: S1A_IW_GRDH_1SDV_20230727T025332_20230727T025357_049603_05F6F2_AF3E

    base_tiles_bounds = [b for b in base_tiles_bounds if is_tile_over_water(b)]
    offset_1_tiles_bounds = [b for b in offset_1_tiles_bounds if is_tile_over_water(b)]
    offset_2_tiles_bounds = [b for b in offset_2_tiles_bounds if is_tile_over_water(b)]

    ntiles = len(base_tiles_bounds)
    noffsettiles = len(offset_1_tiles_bounds)
    print(f"{start_time}: Preparing {ntiles} base tiles (no land).")
    print(f"{start_time}: Preparing {noffsettiles} offset tiles (no land).")

    # write to DB
    async with DatabaseClient(db_engine) as db_client:
        if not payload.dry_run:
            try:
                async with db_client.session.begin():
                    trigger = await db_client.get_trigger(trigger=payload.trigger)
                    layers = [
                        await db_client.get_layer(layer)
                        for layer in model_dict["layers"]
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
                    print(
                        f"{start_time}: Deactivating {stale_slick_count} slicks from stale runs on {payload.sceneid}."
                    )
                    orchestrator_run = await db_client.add_orchestrator(
                        start_time,
                        start_time,
                        ntiles,
                        noffsettiles,
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
            except:  # noqa: E722
                await db_client.session.close()
                raise

            success = True
            try:
                print(f"{start_time}: Instantiating inference client.")
                cloud_run_inference = CloudRunInferenceClient(
                    url=os.getenv("INFERENCE_URL"),
                    titiler_client=titiler_client,
                    sceneid=payload.sceneid,
                    offset_bounds=offset_1_group_bounds,
                    offset_image_shape=offset_group_shape,
                    layers=layers,
                    scale=scale,
                    model_dict=model_dict,
                )

                tileset_bounds_list = [
                    base_tiles_bounds,
                    offset_1_tiles_bounds,
                    offset_2_tiles_bounds,
                ]

                # Perform inferences
                print(f"Inference starting: {start_time}")
                tileset_results_list = [
                    await cloud_run_inference.run_parallel_inference(tileset)
                    for tileset in tileset_bounds_list
                ]

                # Stitch inferences
                print(f"Stitching results: {start_time}")
                model = get_model(model_dict)
                tileset_fc_list = [
                    model.postprocess_tileset(tileset_results, tileset_bounds)
                    for (tileset_results, tileset_bounds) in zip(
                        tileset_results_list, tileset_bounds_list
                    )
                ]

                # Ensemble inferences
                print(f"Ensembling results: {start_time}")
                final_ensemble = model.nms_feature_reduction(
                    features=tileset_fc_list, min_overlaps_to_keep=1
                )

                if final_ensemble.get("features"):
                    LAND_MASK_BUFFER_M = 1000
                    print(
                        f"{start_time}: Removing all slicks within {LAND_MASK_BUFFER_M}m of land"
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
                            feat["properties"]["inf_idx"] = 0
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
                            print(f"{start_time}: Added slick: {slick}")

                    AAA_CONFIDENCE_THRESHOLD = 0.5
                    if any(
                        feat.get("properties").get("machine_confidence")
                        > AAA_CONFIDENCE_THRESHOLD
                        for feat in final_ensemble.get("features")
                    ):
                        print(f"{start_time}: Queueing up Automatic AIS Analysis")
                        add_to_aaa_queue(sentinel1_grd.scene_id)

            except Exception as e:
                success = False
                exc = e
                print(f"{start_time}: {e}")
            async with db_client.session.begin():
                end_time = datetime.now()
                orchestrator_run.success = success
                orchestrator_run.inference_end_time = end_time
                print(f"{start_time}: End time: {end_time}")
                print(f"{start_time}: Orchestration succes: {success}")
            if success is False:
                raise exc

        else:
            print(f"{start_time}: WARNING: Operating as a DRY RUN!!")
    return OrchestratorResult(status="Success")
