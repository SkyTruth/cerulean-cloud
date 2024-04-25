"""
This module defines the base and specific model classes for handling various types
of machine learning models, particularly focusing on MASKRCNN and FASTAIUNET models.
These classes are designed to load models, make predictions, stack results,
and stitch together inference outputs for geospatial analysis.
"""

from base64 import b64decode, b64encode
from typing import List

import geojson
import numpy as np
import rasterio
import torch
import torchvision  # noqa
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import MultiPolygon, shape

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceResult,
    InferenceResultStack,
)


class BaseModel:
    """
    A base class for machine learning models that defines common interfaces for loading models,
    making predictions, stacking results, and stitching outputs.
    """

    def __init__(self, model_dict=None, model_path_local=None):
        """
        Initializes the BaseModel with a model path and inference parameters.

        Args:
            model_path_local (str, optional): The path to the model file.
            model_dict (dict, optional): A dictionary of inference parameters.
        """
        self.model = None
        self.model_dict = model_dict
        self.model_path_local = model_path_local
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        """
        Loads the model from the given path. This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def preprocess_stack(self, inf_stack: List[InferenceInput]):
        """
        Prepares inf_stack for inference.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, inf_stack: List[InferenceInput]):
        """
        Makes predictions on the given input stack. This method should be implemented by subclasses.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_stack(self, raw_preds, bounds):
        """
        Process and stack the raw_preds of predictions. This method should be implemented by subclasses.

        Args:
            raw_preds: The results to be processed and stacked.
            bounds: The bounds corresponding with each pred.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def stitch(self, inference_lists):
        """
        Stitches together inference results. This method should be implemented by subclasses.

        Args:
            inference_lists: The list of inference results to stitch together.
        """
        raise NotImplementedError("Subclasses should implement this method")


class MASKRCNNModel(BaseModel):
    """
    A class for handling MASKRCNN model operations including loading, predicting,
    stacking results, and stitching outputs for geospatial analysis.
    """

    def load(self):
        """
        Loads the MASKRCNN model
        """
        try:
            if self.model is None:
                self.model = torch.jit.load(self.model_path_local, map_location="cpu")
                self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_stack(self, inf_stack: List[InferenceInput]):
        """
        Converts a list of InferenceInput objects into a processed tensor batch for model prediction.
        Processes image data contained in InferenceInput and prepares them for MASKRCNN model inference.
        """
        stack_tensors = [
            b64_image_to_tensor(record.image) / 255 for record in inf_stack
        ]
        print(f"Images have shape {stack_tensors[0].shape}")
        return stack_tensors

    def predict(self, inf_stack: List[InferenceInput]):
        """
        Predicts using the MASKRCNN model on the given input stack.

        Args:
            inf_stack: The input data stack for inference.

        Returns:
            A list of prediction results including geometries and their associated scores.
        """
        print("Initiating cloud_run_offset_tiles/_predict()")
        print(f"Stack has {len(inf_stack)} images")

        self.load()  # Load model into memory
        stack_tensors = self.preprocess_stack(inf_stack)  # Preprocess imagery
        raw_preds = self.model(stack_tensors)[1]  # Run inference
        inference_results = self.postprocess_stack(  # Postprocess inference
            raw_preds, bounds=[record.bounds for record in inf_stack]
        )
        return inference_results

    def postprocess_stack(self, raw_preds, bounds):
        """
        Process and stack the raw_preds of MASKRCNN predictions.

        Args:
            raw_preds: The results to be processed and stacked.
            bounds: The bounds corresponding with each pred.
        """

        reduced_preds = reduce_preds(
            raw_preds,
            **self.model_dict["thresholds"],
            bounds=bounds,
        )

        inference_results = []
        for i, inf in enumerate(reduced_preds):
            inference_results.append(
                InferenceResult(features=inf["polys"], bounds=bounds[i])
            )
        return inference_results

    def stitch(self, inference_list):
        """
        Stitches together inference results from the MASKRCNN model into a geojson feature collection.

        Args:
            inference_list: The list of inference results to stitch together.

        Returns:
            A geojson.FeatureCollection of stitched inference results.
        """
        return geojson.FeatureCollection(features=flatten_feature_list(inference_list))


class FASTAIUNETModel(BaseModel):
    """
    A class for handling FASTAIUNET model operations, including model loading, prediction,
    result stacking, and output stitching.
    """

    def load(self):
        """
        Loads the FASTAIUNET model.

        Args:
            model_path (str): The path to the model file.
        """
        # if self.model is None:
        #     self.model = torch.jit.load(self.model_path_local, map_location="cpu")
        raise NotImplementedError("FASTAIUNET pathway isn't well defined")

    def predict(self, inf_stack):
        """
        Predicts using the FASTAIUNET model on the given input stack.

        Args:
            inf_stack: The input data stack for inference.
        """
        # self.load()

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
        raise NotImplementedError("FASTAIUNET pathway isn't well defined")

    def stack(self, results):
        """
        Stacks the results of FASTAIUNET predictions.

        Args:
            results: The prediction results to be stacked.
        """
        inference_result_stack = []
        for conf, classes, bounds in results:
            enc_classes = array_to_b64_image(classes)
            enc_conf = array_to_b64_image(conf)
            inference_result_stack.append(
                InferenceResult(classes=enc_classes, confidence=enc_conf, bounds=bounds)
            )
        return inference_result_stack

    def stitch(self, inference_list):
        """
        Stitches together inference results from the FASTAIUNET model.

        Args:
            inference_list: The list of inference results to stitch together.
        """
        # print("Loading all tiles into memory for stitch!")
        # ds_base_tiles = []
        # for base_tile_inference in base_tiles_inference:
        #     ds_base_tiles.append(
        #         *[
        #             create_dataset_from_inference_result(b)
        #             for b in base_tile_inference.stack
        #         ]
        #     )

        # ds_offset_tiles = []
        # for offset_tile_inference in offset_tiles_inference:
        #     ds_offset_tiles.append(
        #         *[
        #             create_dataset_from_inference_result(b)
        #             for b in offset_tile_inference.stack
        #         ]
        #     )

        # print("Merging base tiles!")
        # base_tile_inference_file = MemoryFile()
        # ar, transform = merge(ds_base_tiles)
        # with base_tile_inference_file.open(
        #     driver="GTiff",
        #     height=ar.shape[1],
        #     width=ar.shape[2],
        #     count=ar.shape[0],
        #     dtype=ar.dtype,
        #     transform=transform,
        #     crs="EPSG:4326",
        # ) as dst:
        #     dst.write(ar)

        # out_fc = get_fc_from_raster(base_tile_inference_file)

        # print("Merging offset tiles!")
        # offset_tile_inference_file = MemoryFile()
        # ar, transform = merge(ds_offset_tiles)
        # with offset_tile_inference_file.open(
        #     driver="GTiff",
        #     height=ar.shape[1],
        #     width=ar.shape[2],
        #     count=ar.shape[0],
        #     dtype=ar.dtype,
        #     transform=transform,
        #     crs="EPSG:4326",
        # ) as dst:
        #     dst.write(ar)

        # out_fc_offset = get_fc_from_raster(offset_tile_inference_file)
        raise NotImplementedError("FASTAIUNET pathway isn't well defined")


def get_model(
    model_dict,
    model_path_local="cerulean_cloud/cloud_run_offset_tiles/model/model.pt",
):
    """
    Factory function to get the appropriate model instance based on inference parameters.

    Args:
        model_dict (dict): Inference parameters including the model type.
        model_path (str, optional): Path to the model file.

    Returns:
        An instance of the appropriate model class.
    """
    model_type = model_dict["type"]
    print(f"Model type is {model_type}")

    if model_type == "MASKRCNN":
        return MASKRCNNModel(model_dict, model_path_local)
    elif model_type == "FASTAIUNET":
        return FASTAIUNETModel(model_dict, model_path_local)
    else:
        raise ValueError("Unsupported model type")


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

    Args:
        pred_list: A list of prediction dictionaries from the model.
        bbox_score_thresh (Optional[float]): Threshold for bounding box scores.
        pixel_score_thresh (Optional[float]): Threshold for pixel scores.
        pixel_nms_thresh (Optional[float]): Threshold for pixel-level NMS.
        poly_score_thresh (Optional[float]): Threshold for polygon scores.
        bounds (List): Geographic bounds for the predictions.

    Returns:
        A list of reduced and processed prediction dictionaries.
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
            key: (
                [val[i] for i in keep_idxs]
                if isinstance(val, list)
                else torch.index_select(val, 0, keep_idxs)
            )
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
        vectorize_mask_instance(c, pred_dict["scores"][i].detach().item(), transform)
        for i, c in enumerate(high_conf_classes)
    ]
    keep_masks = [i for i, poly in enumerate(pred_dict["polys"]) if poly]
    return pred_dict, keep_masks


def extract_geometry(dataset):
    """Extracts the geometries from a raster dataset."""

    shps = shapes(
        dataset.read(1).astype("uint8"), connectivity=8, transform=dataset.transform
    )
    geoms, inf_idxs = zip(
        *[s for s in shps if s[1] != 0]  # XXX HACK assumes inf_idx=0 is background
    )
    return MultiPolygon([shape(g) for g in geoms]), inf_idxs[0] if inf_idxs else 0


def vectorize_mask_instance(
    high_conf_mask: torch.Tensor, machine_confidence, transform
):
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
        multipoly, inf_idx = extract_geometry(dataset)

    return (
        geojson.Feature(
            geometry=multipoly,
            properties=dict(machine_confidence=machine_confidence, inf_idx=inf_idx),
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


def flatten_feature_list(
    stack_list: List[InferenceResultStack],
) -> List[geojson.Feature]:
    """
    Flattens a list of InferenceResultStack objects into a list of GeoJSON features.

    Args:
        stack_list (List[InferenceResultStack]): List of InferenceResultStack objects.

    Returns:
        List[geojson.Feature]: A list of GeoJSON features.
    """
    flat_list: List[geojson.Feature] = []
    for r in stack_list:
        for i in r.stack:
            for f in i.features:
                flat_list.append(f)
    return flat_list


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


def b64_image_to_array(image: str) -> np.ndarray:
    """convert input b64image to torch tensor"""
    # handle image
    img_bytes = b64decode(image)

    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()

    return np_img


def b64_image_to_tensor(image: str) -> torch.Tensor:
    """
    Converts a base64-encoded image string into a PyTorch tensor.

    Args:
        image (str): A base64-encoded image string.

    Returns:
        torch.Tensor: A tensor representation of the decoded image.
    """
    """convert input b64image to torch tensor"""
    img_bytes = b64decode(image)

    with MemoryFile(img_bytes) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()

    return torch.tensor(np_img)


def array_to_b64_image(np_array: np.ndarray) -> str:
    """
    Encodes a numpy array into a base64-encoded image string.

    Args:
        np_array (np.ndarray): The numpy array to encode.

    Returns:
        str: A base64-encoded image string.
    """
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
