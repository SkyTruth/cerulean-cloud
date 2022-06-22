"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done


needs env vars:
- TITILER_URL
- INFERENCE_URL
"""
import os
from base64 import b64decode, b64encode
from typing import Dict, List, Tuple

import morecantile
import numpy as np
import rasterio
import supermercado
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rasterio.io import MemoryFile
from rasterio.merge import merge

from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceResult
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.schema import (
    OrchestratorInput,
    OrchestratorResult,
)
from cerulean_cloud.tiling import TMS, from_base_tiles_create_offset_tiles
from cerulean_cloud.titiler_client import TitilerClient

app = FastAPI(title="Cloud Run orchestratort")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


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


def get_tiler():
    """get tiler"""
    return TMS


def get_titiler_client():
    """get titiler client"""
    return TitilerClient(url=os.getenv("TITILER_URL"))


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
) -> Dict:
    """orchestrate"""
    return await _orchestrate(payload, tiler, titiler_client)


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


async def _orchestrate(payload, tiler, titiler_client):
    zoom = 9
    scale = 2
    print(f"Orchestrating for sceneid {payload.sceneid}...")
    bounds = titiler_client.get_bounds(payload.sceneid)
    stats = titiler_client.get_statistics(payload.sceneid, band="vv")
    base_tiles = list(tiler.tiles(*bounds, [zoom], truncate=False))
    offset_image_shape = from_tiles_get_offset_shape(base_tiles, scale=scale)
    offset_tiles_bounds = from_base_tiles_create_offset_tiles(base_tiles)
    offset_bounds = from_bounds_get_offset_bounds(offset_tiles_bounds)

    ntiles = len(base_tiles)
    noffsettiles = len(offset_tiles_bounds)

    print(f"Preparing {ntiles} base tiles.")
    print(f"Preparing {noffsettiles} offset tiles.")

    print(f"Scene bounds are {bounds}, stats are {stats}.")
    print(f"Offset image size is {offset_image_shape} with {offset_bounds} bounds.")

    aux_datasets = ["ship_density", os.getenv("AUX_INFRA_DISTANCE")]
    print(f"Instatiating inference client with aux_dataset = {aux_datasets}...")
    cloud_run_inference = CloudRunInferenceClient(
        url=os.getenv("INFERENCE_URL"),
        titiler_client=titiler_client,
        sceneid=payload.sceneid,
        offset_bounds=offset_bounds,
        offset_image_shape=offset_image_shape,
        aux_datasets=aux_datasets,
    )

    print("Inferencing base tiles!")
    base_tiles_inference = []
    for base_tile in base_tiles:
        base_tiles_inference.append(
            await cloud_run_inference.get_base_tile_inference(
                tile=base_tile,
                rescale=(stats["min"], stats["max"]),
            )
        )

    print("Inferencing offset tiles!")
    offset_tiles_inference = []
    for offset_tile_bounds in offset_tiles_bounds:
        offset_tiles_inference.append(
            await cloud_run_inference.get_offset_tile_inference(
                bounds=offset_tile_bounds,
                rescale=(stats["min"], stats["max"]),
            )
        )

    print("Loading all tiles into memory for merge!")
    ds_base_tiles = []
    for base_tile_inference in base_tiles_inference:
        ds_base_tiles.append(create_dataset_from_inference_result(base_tile_inference))

    ds_offset_tiles = []
    for offset_tile_inference in offset_tiles_inference:
        ds_offset_tiles.append(
            create_dataset_from_inference_result(offset_tile_inference)
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
        compress="JPEG",
        crs="EPSG:4326",
    ) as dst:
        dst.write(ar)

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
        compress="JPEG",
        crs="EPSG:4326",
    ) as dst:
        dst.write(ar)

    print("Encoding results!")
    base_inference = b64encode(base_tile_inference_file.read()).decode("ascii")
    offset_inference = b64encode(offset_tile_inference_file.read()).decode("ascii")

    print("Returning results!")
    return OrchestratorResult(
        base_inference=base_inference,
        offset_inference=offset_inference,
        ntiles=ntiles,
        noffsettiles=noffsettiles,
    )
