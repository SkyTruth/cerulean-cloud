"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done
"""
import os
from base64 import b64decode
from typing import Dict

import numpy as np
import rasterio
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


def get_tiler():
    """get tiler"""
    return TMS


def get_titiler_client():
    """get titiler client"""
    return TitilerClient(url=os.getenv("TITILER_URL"))


def get_cloud_run_inference_client():
    """get inference client"""
    return CloudRunInferenceClient(
        url=os.getenv("INFERENCE_URL"), titiler_client=get_titiler_client()
    )


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
def orchestrate(
    payload: OrchestratorInput,
    tiler=Depends(get_tiler),
    titiler_client=Depends(get_titiler_client),
    cloud_run_inference=Depends(get_cloud_run_inference_client),
) -> Dict:
    """orchestrate"""
    return _orchestrate(payload, tiler, titiler_client, cloud_run_inference)


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


def _orchestrate(payload, tiler, titiler_client, cloud_run_inference):
    bounds = titiler_client.get_bounds(payload.sceneid)
    stats = titiler_client.get_statistics(payload.sceneid)
    base_tiles = list(TMS.tiles(*bounds, [10], truncate=False))
    offset_tiles_bounds = from_base_tiles_create_offset_tiles(base_tiles)

    base_tiles_inference = []
    for base_tile in base_tiles:
        base_tiles_inference.append(
            cloud_run_inference.get_base_tile_inference(
                payload.sceneid,
                base_tile,
                rescale=(stats["vv"]["min"], stats["vv"]["max"]),
            )
        )

    offset_tiles_inference = []
    for offset_tile_bounds in offset_tiles_bounds:
        offset_tiles_inference.append(
            cloud_run_inference.get_offset_tile_inference(
                payload.sceneid,
                bounds=offset_tile_bounds,
                rescale=(stats["vv"]["min"], stats["vv"]["max"]),
            )
        )

    ds_base_tiles = []
    for base_tile_inference in base_tiles_inference:
        ds_base_tiles.append(create_dataset_from_inference_result(base_tile_inference))

    ds_offset_tiles = []
    for offset_tile_inference in offset_tiles_inference:
        ds_offset_tiles.append(
            create_dataset_from_inference_result(offset_tile_inference)
        )

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

    # merge(ds_offset_tiles, dst_path="offset.tiff")

    return OrchestratorResult(
        ntiles=len(base_tiles_inference), noffsettiles=len(offset_tiles_inference)
    )
