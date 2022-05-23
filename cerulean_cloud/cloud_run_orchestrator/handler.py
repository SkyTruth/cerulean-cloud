"""Cloud run handler for orchestration of inference
1. Generated base and offset tiles from a scene id
2. Asyncronously place requests to fetch tiles (base and offset), and get inference result
3. Send result of inference to merge tiles cloud run once done
"""
import os
from typing import Dict

from clients import CloudRunInferenceClient
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schema import OrchestratorInput, OrchestratorResult

from ..tiling import TMS, from_base_tiles_create_offset_tiles
from ..titiler_client import TitilerClient

app = FastAPI(title="Cloud Run orchestratort")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def get_tiler():
    """get tiler"""
    return TMS


def get_titiler_client():
    """get titiler client"""
    return TitilerClient(url=os.getenv("TITILER_URL"))


def get_cloud_run_inference_client():
    """get inference client"""
    return CloudRunInferenceClient(url=os.getenv("INFERENCE_URL"))


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
    bounds = titiler_client.get_bounds(payload.sceneid)
    stats = titiler_client.get_statistics(payload.sceneid)
    print(stats)
    base_tiles = list(TMS.tiles(*bounds, [10], truncate=False))
    offset_tiles_bounds = from_base_tiles_create_offset_tiles(base_tiles)
    print(offset_tiles_bounds)
    ntiles = 0
    return OrchestratorResult(ntiles=ntiles)
