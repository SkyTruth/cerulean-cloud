"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from base64 import b64decode, b64encode
from functools import lru_cache
from typing import Dict, List, Tuple

import geojson
import numpy as np
import rasterio
import torch
import torchvision  # noqa
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
    probs = torch.nn.functional.softmax(out_batch_logits, dim=0)
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
    print(f"tensor max is {torch.max(tensor)}")
    tensor = tensor.float() / 255
    print(f"tensor max is {torch.max(tensor)}")
    print(f"Expanded tensor has shape {tensor.shape}")

    print("Running inference...")
    model_type = "MASKRCNN"
    print(f"Model type is {model_type}")
    if model_type == "UNET":
        confidence_threshold = 0.9

        out_batch_logits = model(tensor)
        print("Finished inference, applying softmax")

        res = []
        for i, inference_input in enumerate(payload.stack):
            conf, _classes = logits_to_classes(out_batch_logits[i, :, :, :])
            classes = apply_conf_threshold(conf, _classes, confidence_threshold)
            print(f"Output classes array is {classes.shape}")
            res.append(
                (
                    conf.detach().numpy(),
                    classes.detach().numpy(),
                    inference_input.bounds,
                )
            )
    if model_type == "MASKRCNN":
        bbox_conf_threshold = 0.5
        mask_conf_threshold = 0.05
        size = 512

        res_list = model(torch.unbind(tensor))
        print("Finished inference, applying post-process, thresholding")

        res = []
        for i, inference_input in enumerate(payload.stack):
            pred_dict = apply_conf_threshold_instances(
                res_list[1][i], bbox_conf_threshold=bbox_conf_threshold
            )
            high_conf_classes = apply_conf_threshold_masks(
                pred_dict, mask_conf_threshold=mask_conf_threshold, size=size
            )
            print(f"Output classes array is {high_conf_classes.shape}")
            # for now pass classes as conf, since we don't have conf map
            res.append(
                (
                    high_conf_classes.detach().numpy(),
                    high_conf_classes.detach().numpy(),
                    inference_input.bounds,
                )
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


def apply_conf_threshold_instances(pred_dict, bbox_conf_threshold):
    """Apply a confidence threshold to the output of logits_to_classes for a tile.
    Args:
        pred_dict (dict): a dict with (for example):

        {'boxes': tensor([[  0.00000,  14.11488, 206.41418, 210.23907],
          [ 66.99806, 119.41994, 107.67549, 224.00000],
          [ 47.37723,  41.04019, 122.53947, 224.00000]], grad_fn=<StackBackward0>),
        'labels': tensor([2, 2, 2]),
        'scores': tensor([0.99992, 0.99763, 0.22231], grad_fn=<IndexBackward0>),
        'masks': tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],


                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]],


                [[[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]]]], grad_fn=<UnsqueezeBackward0>)}
        bbox_conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        dict: The confidence thresholded dict result, using the bbox conf threshold. Value sof dict are now list sinstead of tensors: {'boxes':[], 'labels':[], 'scores':[], 'masks':[]}
    """
    new_dict = {"boxes": [], "labels": [], "scores": [], "masks": []}
    for i, score in enumerate(pred_dict["scores"]):
        if score > bbox_conf_threshold:
            new_dict["boxes"].append(pred_dict["boxes"][i])
            new_dict["labels"].append(pred_dict["labels"][i])
            new_dict["scores"].append(pred_dict["scores"][i])
            new_dict["masks"].append(pred_dict["masks"][i])
    return new_dict


def apply_conf_threshold_masks(pred_dict, mask_conf_threshold, size):
    """Apply a confidence threshold to the output of apply_conf_threshold_instances on the masks to get class masks.

    Output is equivalent in shape and content to apply_conf_threshold.

    Args:
        pred_dict (dict): a dict with {'boxes':[], 'labels':[], 'scores':[], 'masks':[]}
        classes (np.ndarray): an array of shape [H, W] of class integers for the max confidence scores for each pixel
        conf_threshold (float): the threshold to use to determine whether a pixel is background or maximally confident category
    Returns:
        List[torch.Tensor]: A list of arrays of shape [H,W] with the class ids that satisfy the confidence threshold. These can be vectorized.
    """
    high_conf_classes = []
    if len(pred_dict["masks"]) > 0:
        for i, mask in enumerate(pred_dict["masks"]):
            classes = torch.ones_like(mask) * pred_dict["labels"][i]
            classes = classes.long().squeeze()
            high_conf_class_mask = torch.where(mask > mask_conf_threshold, 1, 0)
            high_conf_class_mask = torch.where(high_conf_class_mask.bool(), classes, 0)
            high_conf_classes.append(high_conf_class_mask.squeeze())
        return high_conf_classes
    else:
        return [torch.zeros(size, size).long()]


def vectorize_mask_instances(
    high_conf_classes: torch.Tensor, transform
) -> List[geojson.FeatureCollection]:
    """vectorize multiple mask instances"""
    geojson_fcs = []
    for mask_instance in high_conf_classes:
        geojson_fc = vectorize_mask_instance(mask_instance, transform)
        geojson_fcs.append(geojson_fc)
    return geojson_fcs


def vectorize_mask_instance(
    high_conf_mask: torch.Tensor, transform
) -> geojson.FeatureCollection:
    """From a hight conf mask generate a feature collection"""

    memfile = MemoryFile()
    high_conf_mask = high_conf_mask.detach().numpy().astype("int16")
    with memfile.open(
        driver="GTiff",
        height=high_conf_mask.shape[0],
        width=high_conf_mask.shape[1],
        count=1,
        dtype=high_conf_mask.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(high_conf_mask, 1)

    return get_fc_from_raster(memfile)


# taken from cloud run orchestrator
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
