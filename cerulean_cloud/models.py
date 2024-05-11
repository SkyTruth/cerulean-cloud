"""
This module defines the base and specific model classes for handling various types
of machine learning models, particularly focusing on MASKRCNN and FASTAIUNET models.
These classes are designed to load models, make predictions, stack results,
and stitch together inference outputs for geospatial analysis.
"""

import logging
from base64 import b64decode, b64encode
from typing import List

import geojson
import numpy as np
import rasterio
import torch
import torchvision  # noqa
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from scipy.ndimage import label
from shapely.geometry import MultiPolygon, shape

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceResult,
    InferenceResultStack,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        self.background_class_idx = next(
            (
                key
                for key, value in model_dict["cls_map"].items()
                if value == "BACKGROUND"
            ),
            None,
        )

    def load(self):
        """
        Loads the model from the given path.
        """
        try:
            if self.model is None:
                self.model = torch.jit.load(self.model_path_local, map_location="cpu")
                self.model.eval()
        except Exception as e:
            logging.error("Error loading model: %s", e, exc_info=True)
            raise

    def preprocess_tiles(self, inf_stack: List[InferenceInput]):
        """
        Prepares inf_stack for inference.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def predict(self, inf_stack: List[InferenceInput]) -> InferenceResultStack:
        """
        Makes predictions on the given input stack. This method should be implemented by subclasses.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_tiles(self, raw_preds, bounds) -> InferenceResultStack:
        """
        Process and stack the raw_preds of predictions. This method should be implemented by subclasses.

        Args:
            raw_preds: The results to be processed and stacked.
            bounds: The bounds corresponding with each pred.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_tiling(
        self, inference_result_stacks: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Process the InferenceResultStack into a scene and then return a FC. This method should be implemented by subclasses.

        Args:
            inference_result_stacks: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")


class MASKRCNNModel(BaseModel):
    """
    A class for handling MASKRCNN model operations including loading, predicting,
    stacking results, and stitching outputs for geospatial analysis.
    """

    def preprocess_tiles(self, inf_stack: List[InferenceInput]):
        """
        Converts a list of InferenceInput objects into a processed tensor batch for model prediction.
        Processes image data contained in InferenceInput and prepares them for MASKRCNN model inference.
        """
        stack_tensors = [
            b64_image_to_array(record.image, tensor=True) / 255 for record in inf_stack
        ]
        logging.info(f"Images have shape {stack_tensors[0].shape}")
        return stack_tensors

    def predict(self, inf_stack: List[InferenceInput]) -> InferenceResultStack:
        """
        Predicts using the MASKRCNN model on the given input stack.

        Args:
            inf_stack: The input data stack for inference.

        Returns:
            A list of prediction results including geometries and their associated scores.
        """
        logging.info("Initiating cloud_run_offset_tiles/_predict()")
        logging.info(f"Stack has {len(inf_stack)} images")

        self.load()  # Load model into memory
        stack_tensors = self.preprocess_tiles(inf_stack)  # Preprocess imagery
        raw_preds = self.model(stack_tensors)[1]  # Run inference
        inference_result_stack = self.postprocess_tiles(  # Postprocess inference
            raw_preds, bounds=[record.bounds for record in inf_stack]
        )
        return inference_result_stack

    def postprocess_tiles(self, raw_preds, bounds) -> InferenceResultStack:
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
                InferenceResult(features_geojson=inf["polys"], bounds=bounds[i])
            )
        return InferenceResultStack(stack=inference_results)

    def postprocess_tiling(
        self, inference_result_stacks: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Stitches together inference results from the MASKRCNN model into a geojson feature collection.

        Args:
            inference_list: The list of inference results to stitch together.

        Returns:
            A geojson.FeatureCollection of stitched inference results.
        """
        # XXX TODO Move some of the inference processing into here.
        return geojson.FeatureCollection(
            features=flatten_feature_list(inference_result_stacks)
        )


class FASTAIUNETModel(BaseModel):
    """
    A class for handling FASTAIUNET model operations, including model loading, prediction,
    result stacking, and output stitching.
    """

    def preprocess_tiles(self, inf_stack: List[InferenceInput]):
        """
        Converts a list of InferenceInput objects into a processed tensor batch for model prediction.
        Processes image data contained in InferenceInput and prepares them for FASTAIUNET model inference.
        """
        # Pre-calculated statistics from training dataset
        SAR_stats = [0.2087162, 0.13736105]

        try:
            stack_tensors = [
                normalize_and_clamp(
                    b64_image_to_array(record.image, tensor=True),
                    mean=SAR_stats[0],
                    std=SAR_stats[1],
                    device=self.device,
                ).unsqueeze(0)
                for record in inf_stack
            ]
            batch_tensor = torch.cat(stack_tensors, dim=0).to(self.device)
            logging.info(f"Batch tensor shape: {batch_tensor.shape}")
            return batch_tensor  # Only the tensor batch is needed for the model
        except Exception as e:
            logging.error("Error in preprocessing: %s", e, exc_info=True)
            raise

    def predict(self, inf_stack: List[InferenceInput]) -> InferenceResultStack:
        """
        Predicts using the FASTAIUNET model on the given input stack.

        Args:
            inf_stack: The input data stack for inference.
        """
        logging.info(f"Stack has {len(inf_stack)} images")

        self.load()  # Load model into memory
        preprocessed_tensors = self.preprocess_tiles(inf_stack)  # Preprocess imagery
        raw_preds = self.model(preprocessed_tensors)  # Run inference
        inference_result_stack = self.postprocess_tiles(  # Postprocess inference
            raw_preds, bounds=[record.bounds for record in inf_stack]
        )
        return inference_result_stack

    def postprocess_tiles(
        self, raw_preds, bounds: List[List[float]]
    ) -> InferenceResultStack:
        """
        Process and stack the raw_preds of FASTAIUNET predictions.

        Args:
            raw_preds: The 4D results [batch_size, num_classes, pixel_count, pixel_count] to be processed and stacked.
            bounds: The bounds corresponding with each pred.
        """

        inference_results = [
            InferenceResult(
                tile_logits_b64=memfile_gtiff(
                    nparray=p.detach().numpy().astype("uint8"),
                    bounds=bounds[i],
                    encode=True,
                ),
                bounds=bounds[i],
            )
            for i, p in enumerate(raw_preds)
        ]
        return InferenceResultStack(stack=inference_results)

    def postprocess_tiling(
        self, inference_result_stacks: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Stitches together multiple InferenceResultStacks from the FASTAIUNET model.

        Args:
            inference_result_stacks: The list of InferenceResultStacks to stitch together.
        """
        logging.info("Stitching tiles into scene")
        scene_array_logits, transform = self.stitch(inference_result_stacks)
        logging.info("Finding instances in scene")
        features = self.instantiate(scene_array_logits)
        logging.info("Reducing feature count")
        reduced_features = self.reduce_scene_features(features)
        return geojson.FeatureCollection(features=reduced_features)

    def stitch(self, inference_result_stacks: List[InferenceResultStack]):
        """Merge arrays based on their geographical bounds and return the merged array and its bounds.

        Args:
            inference_result_stacks:

        Returns:
            tuple: A tuple containing the merged numpy array and the bounds (min_x, min_y, max_x, max_y) of the merged area.
        """
        ds_tiles = []
        try:
            ds_tiles = [
                memfile_gtiff(
                    nparray=b64_image_to_array(inf.tile_logits_b64),
                    bounds=inf.bounds,
                ).open()
                for inf_stack in inference_result_stacks
                for inf in inf_stack.stack
            ]

            logging.info("Merging tiles!")
            scene_array, transform = rio_merge(ds_tiles)
            return scene_array, transform
        finally:
            for ds in ds_tiles:
                ds.close()

    def instantiate(self, scene_array_logits):
        """
            Processes scene logits to probabilities using softmax, excluding the background class,
            and generates features for each class index.

        Args:
            scene_array_logits (Tensor or numpy.ndarray): A tensor containing logits for each class in the scene,
                where the first dimension corresponds to class indices.

            Returns:
                list: A list of features, where each feature is derived from class probabilities,
                    excluding the background class. Each feature includes additional properties
                    such as the class index.
        """

        # Convert numpy.ndarray to PyTorch tensor if necessary
        if isinstance(scene_array_logits, np.ndarray):
            scene_array_logits = torch.tensor(scene_array_logits)

        scene_probs = torch.nn.functional.softmax(
            scene_array_logits.to(dtype=torch.float32),
            dim=0,
        )
        features = []
        print("len(scene_probs)", len(scene_probs))
        for inf_idx, cls_probs in enumerate(scene_probs):
            # XXX Do we need to add 1 to inf_idx? i.e. do the logits include a background layer or not?
            if inf_idx != self.background_class_idx:
                features.append(
                    instances_from_probs(cls_probs, addl_props={"inf_idx": inf_idx})
                )

        return features

    def reduce_scene_features(self, features):
        """
        Placeholder for a method intended to reduce or process a list of features further.
        Currently, it directly returns the input list without modifications.

        Args:
            features (list): A list of features to be potentially reduced or processed.

        Returns:
            list: The unmodified input list of features, as the function currently does not
                implement any reduction or processing.
        """
        reduced_features = features
        return reduced_features


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
    logging.info(f"Model type is {model_type}")

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
        *[
            s for s in shps if s[1] != 0
        ]  # XXX HACK assumes inf_idx=0 is background should use self.background_class_idx instead
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

    memfile = memfile_gtiff(
        nparray=high_conf_mask.detach().numpy().astype("int16"),
        transform=transform,
    )
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


def memfile_gtiff(nparray, transform=None, bounds=None, encode=False):
    """
    Creates a raster in memory from an array and optionally returns it as a base64 encoded string.
    If encode is False, returns a numpy array of the raster data.
    """
    nparray = nparray[np.newaxis, :, :] if nparray.ndim == 2 else nparray
    transform = transform or (
        rasterio.transform.from_bounds(
            *bounds, width=nparray.shape[2], height=nparray.shape[1]
        )
        if bounds
        else rasterio.transform.from_origin(0, 0, 1, 1)
    )

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        count=nparray.shape[0],  # number of bands
        height=nparray.shape[1],
        width=nparray.shape[2],
        dtype=nparray.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dataset:
        dataset.write(nparray)

    memfile.seek(0)
    if encode:
        encoded = b64encode(memfile.read()).decode("ascii")
        print("XXX 655 encoded:", encoded)
        return encoded
    return memfile


# def logits_to_classes(out_batch_logits, conf_threshold=0.0):
#     """
#     Convert logits from a neural network batch output to class predictions based on probability confidence.

#     Parameters:
#     - out_batch_logits (Tensor): A tensor containing logits for each class, typically of shape (num_classes, num_samples).
#     - conf_threshold (float, optional): A threshold value for confidence. Class predictions with confidence below this value will be set to 0. Default is 0, meaning all predictions are returned regardless of confidence.

#     Returns:
#     - tuple(Tensor, Tensor): A tuple containing two tensors:
#         - The first tensor (`conf`) contains the maximum probabilities (confidences) for each sample.
#         - The second tensor (`classes`) contains the class indices of the highest probability for each sample. If `conf_threshold` is used and the confidence of a prediction is below the threshold, the class index is set to 0.

#     This function applies a softmax normalization to convert logits to probabilities, identifies the maximum probability and corresponding class for each sample, and applies a confidence threshold if specified.
#     """
#     probs = torch.nn.functional.softmax(out_batch_logits, dim=0)
#     conf, classes = torch.max(probs, 0)
#     if conf_threshold > 0:
#         high_conf_mask = torch.any(torch.where(conf > conf_threshold, 1, 0), axis=0)
#         classes = torch.where(high_conf_mask, classes, 0)
#     return (conf, classes)


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
            for f in i.features_geojson:
                flat_list.append(f)
    return flat_list


def b64_image_to_array(image: str, tensor: bool = False):
    """
    Converts a base64-encoded image string into a np.array or torch.tensor.

    Args:
        image (str): A base64-encoded image string.
        tensor (bool, optional): Whether to return a PyTorch tensor. Defaults to False.

    Returns:
        np.ndarray or torch.Tensor: A numpy array or torch tensor representation of the decoded image.
    """
    try:
        img_bytes = b64decode(image)

        with MemoryFile(img_bytes) as memfile:
            with memfile.open() as dataset:
                np_img = dataset.read()

        return torch.tensor(np_img) if tensor else np_img
    except Exception as e:
        logging.error(f"Failed to convert base64 image to array: {e}")
        raise


def normalize_and_clamp(x, mean, std, min_val=-3, max_val=3, device="cpu"):
    """
    Normalizes and clamps a tensor using specified mean and standard deviation,
    and clamps the resulting values to a specified range.

    Args:
        x (Tensor): The input tensor to be normalized and clamped.
        mean (float or list): The mean used for normalization, can be a single float
                              or a list of floats matching the tensor dimensions.
        std (float or list): The standard deviation used for normalization, can be a
                             single float or a list of floats matching the tensor dimensions.
        min_val (float, optional): The minimum value to clamp the tensor to. Defaults to -3.
        max_val (float, optional): The maximum value to clamp the tensor to. Defaults to 3.
        device (str, optional): The device to perform the operations on (e.g., 'cpu' or 'cuda').

    Returns:
        Tensor: The normalized and clamped tensor.
    """

    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)
    x = (x - mean) / std
    x = x.clamp(min=min_val, max=max_val)
    return x


def instances_from_probs(raster, p1, p2, p3, addl_props={}):
    """
    Converts raster predictions to GeoJSON based on probability thresholds.
    Effectively performs grouping, filtering, and trimming of a probability raster, to produce independent features.

    Args:
        raster (np.array): The input raster array to be processed.
        p1 (float): The lowest probability, used to group adjacent polygons into multipolygons. lower value = fewer groups
        p2 (float): The middle probability, used to trim the final polygons size. lower value = coarser polygons
        p3 (float): The highest probability, used to discard polygons that don't reach sufficient confidence. higher value = more restrictive
        Sample values: p1, p2, p3 = 0.1, 0.5, 0.95
        Analogous to p1, bbox_score_thresh, pixel_score_thresh
    Returns:
        GeoJSON: A GeoJSON feature collection of the processed predictions.
    """
    # Label components based on p3 to find peaks
    p1_islands, p1_island_count = label(raster >= p1)
    logging.info("p1_island_count", p1_island_count)
    p3_islands, p3_island_count = label(raster >= p3)
    logging.info("p3_island_count", p3_island_count)

    # Initialize an empty set for unique p1 labels corresponding to p3 components
    reduced_labels = set()

    # Iterate over each p3 component
    for i in range(1, p3_island_count + 1):
        p3_island_mask = p3_islands == i
        p1_label_at_p3 = p1_islands[p3_island_mask].flat[
            0
        ]  # Take the first pixel's p1 label
        reduced_labels.add(p1_label_at_p3)
    logging.info("reduced_labels", len(reduced_labels))

    features = []
    # Process into feature collections based on unique p1 labels
    for p1_label in reduced_labels:
        mask = (p1_islands == p1_label) & (raster >= p2)  # Apply p2 trimming
        masked_raster = raster[mask]
        shapes = rasterio.features.shapes(mask.astype(np.uint8), mask=mask)
        polygons = [shape(geom) for geom, value in shapes if value == 1]
        if polygons:  # Ensure there are polygons to process into a MultiPolygon
            multipolygon = MultiPolygon(polygons)
            features.append(
                geojson.Feature(
                    geometry=multipolygon,
                    properties={
                        "instance_id": p1_label,
                        "mean_conf": np.mean(masked_raster),
                        "median_conf": np.median(masked_raster),
                        "max_conf": np.max(masked_raster),
                        "pixel_count": masked_raster.size,
                        "machine_confidence": np.median(masked_raster),
                        **addl_props,
                    },
                )
            )

    return features
