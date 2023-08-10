"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from base64 import b64decode, b64encode
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import geojson
import numpy as np
import rasterio
import torch
import torchvision  # noqa
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware, record_timing
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, shape
from starlette.requests import Request

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceResult,
    InferenceResultStack,
    PredictPayload,
)

# mypy: ignore-errors

app = FastAPI(title="Cloud Run for offset tiles", dependencies=[Depends(api_key_auth)])
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])
add_timing_middleware(app, prefix="app")


def load_tracing_model(savepath):
    """load tracing model. a tracing model must be applied to the same batch dimensions the model was trained on."""
    tracing_model = torch.jit.load(savepath, map_location="cpu")
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
    inf_stack: List, model, inf_parms: Dict
) -> List[
    Union[
        Tuple[np.ndarray, np.ndarray, List[float]],
        Tuple[List[geojson.Feature], List[float]],
    ]
]:
    print("Initiating cloud_run_offset_tiles/_predict()")
    print(f"Model type is {inf_parms['model_type']}")
    print(f"Stack has {len(inf_stack)} images")

    stack_tensors = [b64_image_to_tensor(record.image) / 255 for record in inf_stack]
    bounds = [record.bounds for record in inf_stack]

    if inf_parms["model_type"] == "MASKRCNN":
        print(f"Images have shape {stack_tensors[0].shape}")

        raw_preds = model(stack_tensors)[1]
        print("Finished inference, applying post-process, thresholding")

        reduced_preds = reduce_preds(
            raw_preds,
            **inf_parms["thresholds"],
            bounds=bounds,
        )
        # returns List[Tuple[List[geojson.Feature], List[float]]]
        return [(inf["polys"], bounds[i]) for i, inf in enumerate(reduced_preds)]

    elif inf_parms["model_type"] == "UNET":
        # out_batch_logits = model(tensor)
        # print("Finished inference, applying softmax")

        # res: Tuple[np.ndarray, np.ndarray, List[float]] = []
        # for i, inference_input in enumerate(inf_stack):
        #     conf, _classes = logits_to_classes(out_batch_logits[i, :, :, :])
        #     classes = apply_conf_threshold(conf, _classes, confidence_threshold)
        #     print(f"Output classes array is {classes.shape}")
        #     res.append(
        #         (
        #             conf.detach().numpy(),
        #             classes.detach().numpy(),
        #             inference_input.bounds,
        #         )
        #     )
        raise NotImplementedError("UNET pathway isn't well defined")

    else:
        raise NotImplementedError("Model_type must be one of ['MASKRCNN', 'UNET']")


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResultStack,
)
def predict(
    request: Request, payload: PredictPayload, model=Depends(get_model)
) -> Dict:
    """predict"""
    record_timing(request, note="Started")
    results = _predict(payload.inf_stack, model, payload.inf_parms)
    record_timing(request, note="Finished inference")

    inference_result_stack = []
    if len(results[0]) == 2:
        for feats, bounds in results:
            inference_result_stack.append(
                InferenceResult(features=feats, bounds=bounds)
            )

    else:
        for conf, classes, bounds in results:
            enc_classes = array_to_b64_image(classes)
            enc_conf = array_to_b64_image(conf)
            inference_result_stack.append(
                InferenceResult(classes=enc_classes, confidence=enc_conf, bounds=bounds)
            )
    record_timing(request, note="Returning")
    return InferenceResultStack(stack=inference_result_stack)


def reduce_preds(
    pred_list,
    bbox_score_thresh=None,
    pixel_score_thresh=None,
    pixel_nms_thresh=None,
    poly_score_thresh=None,
    bounds=[],
    **kwargs,
):
    """
    Apply various post-processing steps to the predictions from an object detection model.

    The post-processing includes:
    1. Removal of instances with bounding boxes below a certain confidence score.
    2. Removal of instances with pixel scores below a certain threshold.
    3. Application of non-maximum suppression at the pixel level.
    4. Generation of vectorized polygons from the remaining predictions.

    Arguments:
    - pred_list: A list of dictionaries containing model predictions. The dictionary should contain tensors with keys: "boxes", "labels", "scores", "masks".
    - bbox_score_thresh: A float indicating the confidence threshold for bounding boxes.
    - pixel_score_thresh: A float indicating the confidence threshold for pixels.
    - pixel_nms_thresh: A float indicating the threshold for pixel-based non-maximum suppression.
    - poly_score_thresh: A float indicating the confidence threshold for polygons.
    - bounds: A 2-D Tensor representing a list of geographical bounds [west, south, east, north].
    - kwargs: Additional parameters.

    Returns:
    - torch.Tensor: A tensor of indices that were kept.
    - A dictionary containing the post-processed predictions.
    """
    bounds = bounds or [[0, -1, 1, 0]] * len(pred_list)
    reduced_preds = []
    for i, pred_dict in enumerate(pred_list):
        # remove instances with low scoring boxes
        if bbox_score_thresh is not None:
            keep = keep_by_bbox_score(pred_dict, bbox_score_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # remove instances with low scoring pixels
        if pixel_score_thresh is not None:
            keep = keep_by_pixel_score(pred_dict, pixel_score_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # non-maximum suppression, done globally, across classes, using pixels rather than bboxes
        if pixel_nms_thresh is not None:
            keep = keep_by_global_pixel_nms(pred_dict, pixel_nms_thresh)
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        # generate vectorized polygons from predictions
        # adds "polys" to pred_dict
        if poly_score_thresh is not None:
            pred_dict, keep = polygonize_pixel_segmentations(
                pred_dict, poly_score_thresh, bounds[i]
            )
            pred_dict = keep_boxes_by_idx(pred_dict, keep)

        reduced_preds.extend([pred_dict])
    return reduced_preds


def keep_boxes_by_idx(pred_dict, keep_idxs):
    """
    Filters the prediction dictionary to keep only the indices specified in keep_idxs.

    Args:
    pred_dict (dict): The prediction dictionary to be filtered.
    keep_idxs (list or torch.Tensor): A list or a tensor of indices to keep.

    Returns:
    dict: A new dictionary with the same keys as pred_dict, but with values filtered to include only those at the indices specified by keep_idxs.
    """
    if not len(keep_idxs):  # if keep_idxs is empty
        return {
            key: [] if isinstance(val, list) else torch.tensor([])
            for key, val in pred_dict.items()
        }
    else:
        keep_idxs = (
            torch.tensor(keep_idxs) if isinstance(keep_idxs, list) else keep_idxs
        )
        return {
            key: [val[i] for i in keep_idxs]
            if isinstance(val, list)
            else torch.index_select(val, 0, keep_idxs)
            for key, val in pred_dict.items()
        }


def keep_by_bbox_score(pred_dict, bbox_score_thresh):
    """
    Finds the indices of bounding box predictions that have a score greater than bbox_score_thresh.

    Args:
    pred_dict (dict): The prediction dictionary to be filtered.
    bbox_score_thresh (float): The minimum score for a bounding box prediction to be kept.

    Returns:
    torch.Tensor: A tensor of indices for bounding box predictions that meet the score threshold.
    """
    return torch.where(pred_dict["scores"] > bbox_score_thresh)[0]


def keep_by_pixel_score(pred_dict, pixel_score_thresh):
    """
    Given a dictionary of predictions and a pixel score threshold, returns the indices
    of those predictions that exceed the threshold.

    Args:
        pred_dict (dict): Dictionary of prediction data.
        pixel_score_thresh (float): Threshold for pixel scores.

    Returns:
        torch.Tensor: Indices of predictions exceeding the threshold.

    """
    mask_maxes = (
        torch.stack([m.max() for m in pred_dict["masks"]])
        if len(pred_dict["masks"])
        else torch.tensor([])
    )
    return torch.where(mask_maxes > pixel_score_thresh)[0]


def keep_by_global_pixel_nms(pred_dict, pixel_nms_thresh):
    """
    Apply non-maximum suppression (NMS) to a dictionary of predictions.

    This function iterates over a dictionary of predicted masks and calculates
    the Dice Coefficient to measure similarity between each pair of masks.
    If the coefficient exceeds a certain threshold, the mask is marked for removal.

    Args:
        pred_dict (dict): Dictionary with key "masks" containing a list of predicted masks.
        pixel_nms_thresh (float): The threshold above which two predictions are considered overlapping.

    Returns:
        list: List of indices of masks to keep.
    """
    masks = pred_dict["masks"]
    masks_to_remove = []

    for i, current_mask in enumerate(masks):  # Loop through all masks
        # Skip if the mask is already marked for removal
        if i in masks_to_remove:
            continue

        # Check similarity against all subsequent masks
        for j, comparison_mask in enumerate(masks[i + 1 :], start=i + 1):
            # Skip if the mask is already marked for removal
            if j in masks_to_remove:
                continue

            # Calculate Dice Coefficient; if the similarity is too high, mark mask for removal
            if (
                calculate_dice_coefficient(
                    current_mask.squeeze(), comparison_mask.squeeze()
                )
                > pixel_nms_thresh
            ):
                masks_to_remove.append(j)

    # Return a list of mask indices that are not marked for removal
    return [i for i in range(len(masks)) if i not in masks_to_remove]


def calculate_dice_coefficient(u, v):
    """
    Takes two pixel-confidence masks, and calculates how similar they are to each other
    Returns a value between 0 (no overlap) and 1 (identical)
    Utilizes an IoU style construction
    Can be used as NMS across classes for mutually-exclusive classifications
    """
    return 2 * torch.sum(torch.sqrt(torch.mul(u, v))) / (torch.sum(u + v))


def polygonize_pixel_segmentations(pred_dict, poly_score_thresh, bounds):
    """
    Given a dictionary of predictions, a polygon score threshold, and bounding coordinates,
    transforms pixel segmentations into polygons and updates the prediction dictionary
    to include these polygons.

    Args:
        pred_dict (dict): Dictionary of prediction data.
        poly_score_thresh (float): Threshold for polygon scores.
        bounds (tuple): Bounding coordinates.

    Returns:
        tuple: Updated prediction dictionary and indices of polygons.

    """
    high_conf_classes = [
        torch.where(mask > poly_score_thresh, pred_dict["labels"][i], 0)
        .squeeze()
        .long()
        for i, mask in enumerate(pred_dict["masks"])
    ]
    transform = (
        rasterio.transform.from_bounds(*bounds, *high_conf_classes[0].shape[:2])
        if high_conf_classes
        else None
    )
    pred_dict["polys"] = [
        vectorize_mask_instance(c, transform) for c in high_conf_classes
    ]
    keep_masks = [i for i, poly in enumerate(pred_dict["polys"]) if poly]
    return pred_dict, keep_masks


def extract_geometries(dataset):
    """Extracts the geometries from a raster dataset."""

    shps = shapes(
        dataset.read(1).astype("uint8"), connectivity=8, transform=dataset.transform
    )
    geoms = [shape(geom) for geom, value in shps if value != 0]
    return geoms


def vectorize_mask_instance(high_conf_mask: torch.Tensor, transform):
    """
    From a high confidence mask, generate a GeoJSON feature collection.

    Args:
        high_conf_mask (torch.Tensor): A tensor representing the high confidence mask.
        transform: A transformation to apply to the mask.

    Returns:
        geojson.Feature: A GeoJSON feature object.
    """

    memfile = create_memfile(high_conf_mask, transform)

    with memfile.open() as dataset:
        geoms = extract_geometries(dataset)
        multipoly = MultiPolygon(geoms)

    return (
        geojson.Feature(
            geometry=multipoly,
            properties=dict(),
        )
        if multipoly
        else None
    )


def create_memfile(high_conf_mask, transform):
    """Creates a raster in memory from a mask tensor."""

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
    ) as dataset:
        dataset.write(high_conf_mask, 1)

    return memfile
