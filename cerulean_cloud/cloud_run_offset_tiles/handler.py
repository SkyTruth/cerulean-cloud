"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from base64 import b64decode, b64encode
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware, record_timing
from rasterio.io import MemoryFile
from starlette.requests import Request

from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceInput, InferenceResult

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])
add_timing_middleware(app, prefix="app")


def load_tracing_model(savepath):
    """load tracing model"""
    tracing_model = torch.jit.load(savepath)
    return tracing_model


@lru_cache()
def get_model():
    """load model"""
    return load_tracing_model("cloud_run_offset_tiles/model/model.pt")


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits, dim=1)
    conf, classes = torch.max(probs, 1)
    return (conf, classes)


def b64_image_to_tensor(image: str) -> torch.Tensor:
    """convert input b64image to torch tensor"""
    # handle image
    img_bytes = b64decode(image)

    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()

    return torch.tensor(np_img)


def array_to_b64_image(np_array: np.ndarray) -> str:
    """convert input b64image to torch tensor"""
    np_array = np_array.astype("int8")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=np_array.shape[0],
            dtype=np_array.dtype,
            width=np_array.shape[1],
            height=np_array.shape[2],
        ) as dataset:
            dataset.write(np_array)
        img_bytes = memfile.read()

    return b64encode(img_bytes).decode("ascii")


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


def _predict(payload: InferenceInput, model) -> Tuple[np.ndarray, np.ndarray]:
    print("Loading tensor!")
    tensor = b64_image_to_tensor(payload.image)
    print(f"Original tensor has shape {tensor.shape}")
    tensor = tensor[None, :, :, :]
    tensor = tensor.float()
    print(f"Expanded tensor has shape {tensor.shape}")

    print("Running inference...")
    out_batch_logits = model(tensor)
    print("Finished inference, applying softmax")
    conf, classes = logits_to_classes(out_batch_logits)
    print(f"Output classes array is {classes.shape}")
    return classes.numpy(), conf.detach().numpy()


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResult,
)
def predict(
    request: Request, payload: InferenceInput, model=Depends(get_model)
) -> Dict:
    """predict"""
    record_timing(request, note="Started")
    classes, conf = _predict(payload, model)
    record_timing(request, note="Finished inference")
    enc_classes = array_to_b64_image(classes)
    enc_conf = array_to_b64_image(conf)
    record_timing(request, note="Returning")
    return InferenceResult(
        classes=enc_classes, confidence=enc_conf, bounds=payload.bounds
    )
