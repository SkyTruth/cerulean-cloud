"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""
import asyncio
import os
import urllib.parse as urlparse
from base64 import b64decode  # , b64encode
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import geojson
import morecantile
import numpy as np
import rasterio
import supermercado
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from global_land_mask import globe
from rasterio.io import MemoryFile
from rasterio.merge import merge

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceResult,
    InferenceResultStack,
)
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.merging import merge_inferences
from cerulean_cloud.cloud_run_orchestrator.schema import (
    OrchestratorInput,
    OrchestratorResult,
)
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient
from cerulean_cloud.tiling import TMS, from_base_tiles_create_offset_tiles
from cerulean_cloud.titiler_client import TitilerClient

app = FastAPI(title="Cloud Run orchestrator", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


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


def b64_image_to_array(image: str) -> np.ndarray:
    """convert input b64image to torch tensor"""
    # handle image
    img_bytes = b64decode(image)

    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()

    return np_img


def from_tiles_get_offset_shape(
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


def from_bounds_get_offset_bounds(bounds: List[List[float]]) -> List[float]:
    """from a list of bounds, get the merged bounds (min max)"""
    bounds_np = np.array([(b[0], b[1], b[2], b[3]) for b in bounds])
    minx, miny, maxx, maxy = (
        np.min(bounds_np[:, 0]),
        np.min(bounds_np[:, 1]),
        np.max(bounds_np[:, 2]),
        np.max(bounds_np[:, 3]),
    )
    return list((minx, miny, maxx, maxy))


def get_fc_from_raster(raster: MemoryFile) -> geojson.FeatureCollection:
    """create a geojson from an input raster with classification

    Args:
        raster (MemoryFile): input raster

    Returns:
        geojson.FeatureCollection: output feature collection
    """
    with raster.open() as dataset:
        shapes = rasterio.features.shapes(
            dataset.read(1).astype("uint8"), connectivity=8, transform=dataset.transform
        )
    out_fc = geojson.FeatureCollection(
        features=[
            geojson.Feature(
                geometry=geom, properties=dict(classification=classification)
            )
            for geom, classification in shapes
            if int(classification) != 0
        ]
    )
    return out_fc


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


def create_dataset_from_inference_result(
    inference_output: InferenceResult,
) -> rasterio.io.DatasetReader:
    """From inference result create a open rasterio dataset for merge"""
    classes_array = b64_image_to_array(inference_output.classes)
    conf_array = b64_image_to_array(inference_output.confidence)
    ar = np.concatenate([classes_array, conf_array])

    transform = rasterio.transform.from_bounds(
        *inference_output.bounds, width=ar.shape[1], height=ar.shape[2]
    )

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=ar.shape[1],
        width=ar.shape[2],
        count=ar.shape[0],
        dtype=ar.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(ar)
    return memfile.open()


def is_tile_over_water(tile_bounds: List[float]) -> bool:
    """are the tile bounds over water"""
    minx, miny, maxx, maxy = tile_bounds
    return any(globe.is_ocean([miny, maxy], [minx, maxx]))


def flatten_feature_list(
    stack_list: List[InferenceResultStack],
) -> List[geojson.Feature]:
    """flatten a feature list coming from inference"""
    flat_list: List[geojson.Feature] = []
    for r in stack_list:
        for i in r.stack:
            for f in i.features:
                flat_list.append(f)
    return flat_list


async def _orchestrate(
    payload, tiler, titiler_client, roda_sentinelhub_client, db_engine
):
    # Orchestrate inference
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    zoom = payload.zoom
    scale = payload.scale
    print(f"Orchestrating for sceneid {payload.sceneid}...")
    bounds = await titiler_client.get_bounds(payload.sceneid)
    stats = await titiler_client.get_statistics(payload.sceneid, band="vv")
    info = await roda_sentinelhub_client.get_product_info(payload.sceneid)
    print(info)
    base_tiles = list(tiler.tiles(*bounds, [zoom], truncate=False))
    offset_image_shape = from_tiles_get_offset_shape(base_tiles, scale=scale)
    offset_tiles_bounds = from_base_tiles_create_offset_tiles(base_tiles)
    offset_bounds = from_bounds_get_offset_bounds(offset_tiles_bounds)
    print(f"Original tiles are {len(base_tiles)}, {len(offset_tiles_bounds)}")

    # Filter out land tiles
    base_tiles = [t for t in base_tiles if is_tile_over_water(tiler.bounds(t))]
    offset_tiles_bounds = [b for b in offset_tiles_bounds if is_tile_over_water(b)]

    ntiles = len(base_tiles)
    noffsettiles = len(offset_tiles_bounds)

    print(f"Preparing {ntiles} base tiles (no land).")
    print(f"Preparing {noffsettiles} offset tiles (no land).")

    print(f"Scene bounds are {bounds}, stats are {stats}.")
    print(f"Offset image size is {offset_image_shape} with {offset_bounds} bounds.")

    aux_infra_distance = os.getenv("AUX_INFRA_DISTANCE")
    aux_datasets = [
        "ship_density",
        aux_infra_distance,
    ]  # XXXDB This should pull from the model layers instead

    # write to DB
    async with DatabaseClient(db_engine) as db_client:
        try:
            async with db_client.session.begin():
                trigger = await db_client.get_trigger(trigger=payload.trigger)
                model = await db_client.get_model(os.getenv("MODEL"))
                sentinel1_grd = await db_client.get_sentinel1_grd(
                    payload.sceneid,
                    info,
                    titiler_client.get_base_tile_url(
                        payload.sceneid, rescale=(stats["min"], stats["max"])
                    ),
                )
                db_client.session.add(sentinel1_grd)
                orchestrator_run = db_client.add_orchestrator(
                    start_time,
                    start_time,
                    ntiles,
                    noffsettiles,
                    os.getenv("GIT_HASH"),
                    os.getenv("GIT_TAG"),
                    make_cloud_log_url(
                        os.getenv("CLOUD_RUN_NAME"), start_time, os.getenv("PROJECT_ID")
                    ),
                    zoom,
                    scale,
                    bounds,
                    trigger,
                    model,
                    sentinel1_grd,
                )
                db_client.session.add(orchestrator_run)
        except:  # noqa: E722
            await db_client.session.close()
            raise

        if not payload.dry_run:
            print(
                f"Instantiating inference client with aux_dataset = {aux_datasets}..."
            )
            cloud_run_inference = CloudRunInferenceClient(
                url=os.getenv("INFERENCE_URL"),
                titiler_client=titiler_client,
                sceneid=payload.sceneid,
                offset_bounds=offset_bounds,
                offset_image_shape=offset_image_shape,
                aux_datasets=aux_datasets,
                scale=scale,
            )

            print("Inference on base tiles!")
            base_tile_semaphore = asyncio.Semaphore(value=20)
            base_tiles_inference = await asyncio.gather(
                *[
                    cloud_run_inference.get_base_tile_inference(
                        tile=base_tile,
                        rescale=(stats["min"], stats["max"]),
                        semaphore=base_tile_semaphore,
                    )
                    for base_tile in base_tiles
                ],
                return_exceptions=True,
            )

            print("Inference on offset tiles!")
            offset_tile_semaphore = asyncio.Semaphore(value=20)
            offset_tiles_inference = await asyncio.gather(
                *[
                    cloud_run_inference.get_offset_tile_inference(
                        bounds=offset_tile_bounds,
                        rescale=(stats["min"], stats["max"]),
                        semaphore=offset_tile_semaphore,
                    )
                    for offset_tile_bounds in offset_tiles_bounds
                ],
                return_exceptions=True,
            )

            if base_tiles_inference[0].stack[0].dict().get("classes"):
                print("Loading all tiles into memory for merge!")
                ds_base_tiles = []
                for base_tile_inference in base_tiles_inference:
                    ds_base_tiles.append(
                        *[
                            create_dataset_from_inference_result(b)
                            for b in base_tile_inference.stack
                        ]
                    )

                ds_offset_tiles = []
                for offset_tile_inference in offset_tiles_inference:
                    ds_offset_tiles.append(
                        *[
                            create_dataset_from_inference_result(b)
                            for b in offset_tile_inference.stack
                        ]
                    )

                print("Merging base tiles!")
                base_tile_inference_file = MemoryFile()
                ar, transform = merge(ds_base_tiles)
                with base_tile_inference_file.open(
                    driver="GTiff",
                    height=ar.shape[1],
                    width=ar.shape[2],
                    count=ar.shape[0],
                    dtype=ar.dtype,
                    transform=transform,
                    crs="EPSG:4326",
                ) as dst:
                    dst.write(ar)

                out_fc = get_fc_from_raster(base_tile_inference_file)

                print("Merging offset tiles!")
                offset_tile_inference_file = MemoryFile()
                ar, transform = merge(ds_offset_tiles)
                with offset_tile_inference_file.open(
                    driver="GTiff",
                    height=ar.shape[1],
                    width=ar.shape[2],
                    count=ar.shape[0],
                    dtype=ar.dtype,
                    transform=transform,
                    crs="EPSG:4326",
                ) as dst:
                    dst.write(ar)

                out_fc_offset = get_fc_from_raster(offset_tile_inference_file)

            else:
                out_fc = geojson.FeatureCollection(
                    features=flatten_feature_list(base_tiles_inference)
                )
                out_fc_offset = geojson.FeatureCollection(
                    features=flatten_feature_list(offset_tiles_inference)
                )

            merged_inferences = merge_inferences(out_fc, out_fc_offset)

            for feat in merged_inferences.get("features"):
                async with db_client.session.begin():
                    slick = db_client.add_slick(
                        orchestrator_run,
                        sentinel1_grd.start_time,
                        feat.get("geometry"),
                        feat.get("properties").get("classification"),
                        feat.get("properties").get("confidence"),
                    )
                    print(f"Added slick {slick}")

            end_time = datetime.now()
            print(f"End time: {end_time}")
            print("Returning results!")

            async with db_client.session.begin():
                orchestrator_run.success = True
                orchestrator_run.inference_end_time = end_time

            orchestrator_result = OrchestratorResult(
                classification_base=out_fc,
                classification_offset=out_fc_offset,
                classification_merged=merged_inferences,
                ntiles=ntiles,
                noffsettiles=noffsettiles,
            )
        else:
            print("DRY RUN!!")
            orchestrator_result = OrchestratorResult(
                classification_base=geojson.FeatureCollection(features=[]),
                classification_offset=geojson.FeatureCollection(features=[]),
                classification_merged=geojson.FeatureCollection(features=[]),
                ntiles=ntiles,
                noffsettiles=noffsettiles,
            )

    return orchestrator_result
