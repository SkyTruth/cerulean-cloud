"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from base64 import b64decode, b64encode
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware, record_timing
from rasterio.io import MemoryFile
from starlette.requests import Request

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInputStack,
    InferenceResult,
    InferenceResultStack,
)

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])
add_timing_middleware(app, prefix="app")


def load_tracing_model(savepath):
    """load tracing model. a tracing model must be applied to the same batch dimensions the model was trained on."""
    tracing_model = torch.jit.load(savepath)
    return tracing_model


@lru_cache()
def get_model():
    """load model"""
    return load_tracing_model("cerulean_cloud/cloud_run_offset_tiles/model/model.pt")


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids for a single tile of shape [classes, H, W].
    """
    probs = torch.nn.functional.softmax(out_batch_logits, dim=0)  # 0 is the class dim
    conf, classes = torch.max(probs, 0)
    return (conf, classes)


def apply_conf_threshold(conf, classes, conf_threshold):
    """Apply a confidence threshold to the output of logits_to_classes for a tile.

    Args:
        conf (np.ndarray): an array of shape [H, W] of max confidence scores for each pixel
        classes (np.ndarray): an array of shape [H, W] of class integers for the max confidence scores for each pixel
        conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category

    Returns:
        torch.Tensor: An array of shape [H,W] with the class ids that satisfy the confidence threshold. This can be vectorized.
    """
    high_conf_mask = torch.any(torch.where(conf > conf_threshold, 1, 0), axis=0)
    return torch.where(high_conf_mask, classes, 0)


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
    if len(np_array.shape) == 2:
        np_array = np.expand_dims(np_array, 0)
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


def _predict(
    payload: InferenceInputStack, model
) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    print("Loading tensor!")
    stack_tensors = []
    for inference_input in payload.stack:
        stack_tensors.append(b64_image_to_tensor(inference_input.image))

    print(f"Stack has {len(stack_tensors)} images")
    tensor = torch.stack(stack_tensors)

    print(f"Original tensor has shape {tensor.shape}")
    tensor = tensor.float()
    print(f"Expanded tensor has shape {tensor.shape}")

    print("Running inference...")
    out_batch_logits = model(tensor)
    print("Finished inference, applying softmax")

    res = []
    for i, inference_input in enumerate(payload.stack):
        conf, classes = logits_to_classes(out_batch_logits[i, :, :, :])
        print(f"Output classes array is {classes.shape}")
        res.append(
            (conf.detach().numpy(), classes.detach().numpy(), inference_input.bounds)
        )
    return res


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResultStack,
)
def predict(
    request: Request, payload: InferenceInputStack, model=Depends(get_model)
) -> Dict:
    """predict"""
    record_timing(request, note="Started")
    results = _predict(payload, model)
    record_timing(request, note="Finished inference")

    inference_result_stack = []
    for conf, classes, bounds in results:
        enc_classes = array_to_b64_image(classes)
        enc_conf = array_to_b64_image(conf)
        inference_result_stack.append(
            InferenceResult(classes=enc_classes, confidence=enc_conf, bounds=bounds)
        )
    record_timing(request, note="Returning")
    return InferenceResultStack(stack=inference_result_stack)
