"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""

from typing import Dict

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware, record_timing
from starlette.requests import Request

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceResultStack,
    PredictPayload,
)
from cerulean_cloud.models import get_model
from cerulean_cloud.utils import configure_structured_logger, context_dict_var

# mypy: ignore-errors

app = FastAPI(title="Cloud Run for offset tiles", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])
add_timing_middleware(app, prefix="app")

# Configure the logger once at startup
configure_structured_logger("cerulean_cloud")


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResultStack,
)
def predict(request: Request, payload: PredictPayload) -> InferenceResultStack:
    """Run prediction using the loaded model."""
    record_timing(request, note="Started")
    context_dict_var.set({"scene_id": payload.scene_id})  # Set the scene_id for logging

    model = get_model(payload.model_dict)
    record_timing(request, note="Model loaded")

    return model.predict(payload.inf_stack)
