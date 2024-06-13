"""
This module defines the base and specific model classes for handling various types
of machine learning models, particularly focusing on MASKRCNN and FASTAIUNET models.
These classes are designed to load models, make predictions, stack results,
and stitch together inference outputs for geospatial analysis.
"""

import json
import logging
import os
from base64 import b64decode, b64encode
from io import BytesIO
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

    def predict(self, inf_stack: List[InferenceInput]) -> InferenceResultStack:
        """
        Makes predictions on the given input stack. The submethods should be implemented by subclasses.

        Args:
            inf_stack: The input data stack for inference.
        """
        logging.info(f"Stack has {len(inf_stack)} images")

        self.load()  # Load model into memory
        preprocessed_tensors = self.preprocess_tiles(inf_stack)  # Preprocess imagery
        raw_preds = self.process_tiles(preprocessed_tensors)  # Run inference
        inference_results = self.postprocess_tiles(raw_preds)  # Postprocess inference
        return InferenceResultStack(
            stack=inference_results, bounds=[i.bounds for i in inf_stack]
        )

    def preprocess_tiles(self, inf_stack):
        """
        Prepares inf_stack for inference.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def process_tiles(self, inf_stack: List[InferenceInput]):
        """
        Runs inf_stack through inference.

        Args:
            inf_stack: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_tiles(self, raw_preds) -> List[InferenceResult]:
        """
        Process and stack the raw_preds of predictions. This method should be implemented by subclasses.

        Args:
            raw_preds: The results to be processed and stacked.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_tileset(
        self, tileset_results: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Process the InferenceResultStack into a scene and then return a FC. This method should be implemented by subclasses.

        Args:
            tileset_results: The input data stack for inference.
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
            b64_image_to_array(record.image, tensor=True, to_float=True)
            for record in inf_stack
        ]
        logging.info(f"Images have shape {stack_tensors[0].shape}")
        return stack_tensors

    def process_tiles(self, stack_tensors):
        """
        Run inference on the loaded model
        """

        return self.model(stack_tensors)[1]

    def postprocess_tiles(self, raw_preds) -> List[InferenceResult]:
        """
        Process and stack the raw_preds of MASKRCNN predictions.

        Args:
            raw_preds: The results to be processed and stacked.
        """
        inference_results = [
            InferenceResult(json_data=self.serialize(pred)) for pred in raw_preds
        ]
        return inference_results

    def serialize(self, pred):
        """
        Serializes a prediction dictionary by encoding its 'masks' (PyTorch tensors) using torch.save
        to base64 strings. All other elements in the dictionary are left as-is, and the entire
        dictionary is converted to a JSON string.

        Parameters:
        - pred (dict): The prediction data containing 'masks' and potentially other items.

        Returns:
        - str: A JSON string with the 'masks' converted to base64-encoded strings.
        """
        pred["masks_b64"] = [tensor_to_base64(mask) for mask in pred.pop("masks")]
        json_string = json.dumps(pred)
        return json_string

    def deserialize(self, json_string):
        """
        Deserializes a JSON string into a prediction dictionary, decoding 'masks' from base64 strings
        back to PyTorch tensors using torch.load, automatically handling tensor type and shape.

        Parameters:
        - json_string (str): The JSON string containing the serialized prediction data.

        Returns:
        - dict: The original prediction dictionary with 'masks' restored as PyTorch tensors.
        """
        pred = json.loads(json_string)
        pred["masks"] = [
            torch.load(BytesIO(b64decode(b64_str))) for b64_str in pred.pop("masks_b64")
        ]
        return pred

    def postprocess_tileset(
        self, tileset_results: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Stitches together inference results from the MASKRCNN model into a geojson feature collection.

        Args:
            tileset_results: The list of inference results to stitch together.

        Returns:
            A geojson.FeatureCollection of stitched inference results.
        """
        logging.info("Stitching tiles into scene")
        features_list = self.stitch(tileset_results)
        logging.info("Reducing feature count")
        reduced_features = self.reduce_scene_features(features_list)

        return geojson.FeatureCollection(features=reduced_features)

    def stitch(
        self,
        tileset_results: List[InferenceResultStack],
    ):
        """Merge arrays based on their geographical bounds and return the merged array and its bounds.

        Args:
            tileset_results:

        Returns:

        """
        # bounds = [
        #     bounds
        #     for inference_result_stack in tileset_results
        #     for bounds in inference_result_stack.bounds
        # ]

        return tileset_results

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
        # reduced_preds = self.reduce_preds(
        #     features,
        #     **self.model_dict["thresholds"],
        # )

        reduced_features = flatten_feature_list(features)
        # XXX TODO move feature reduction here
        return reduced_features

    def reduce_preds(
        self,
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
                keep = self.keep_by_bbox_score(pred_dict, bbox_score_thresh)
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            # remove instances with low scoring pixels
            if pixel_score_thresh is not None:
                keep = self.keep_by_pixel_score(pred_dict, pixel_score_thresh)
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            # non-maximum suppression, done globally, across classes, using pixels rather than bboxes
            if pixel_nms_thresh is not None:
                keep = self.keep_by_global_pixel_nms(pred_dict, pixel_nms_thresh)
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            # generate vectorized polygons from predictions
            # adds "polys" to pred_dict
            if poly_score_thresh is not None:
                pred_dict, keep = self.polygonize_pixel_segmentations(
                    pred_dict, poly_score_thresh, bounds[i]
                )
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            reduced_preds.extend([pred_dict])
        return reduced_preds

    def keep_boxes_by_idx(self, pred_dict, keep_idxs):
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

    def keep_by_bbox_score(self, pred_dict, bbox_score_thresh):
        """
        Finds the indices of bounding box predictions that have a score greater than bbox_score_thresh.

        Args:
        pred_dict (dict): The prediction dictionary to be filtered.
        bbox_score_thresh (float): The minimum score for a bounding box prediction to be kept.

        Returns:
        torch.Tensor: A tensor of indices for bounding box predictions that meet the score threshold.
        """
        return torch.where(pred_dict["scores"] > bbox_score_thresh)[0]

    def keep_by_pixel_score(self, pred_dict, pixel_score_thresh):
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

    def keep_by_global_pixel_nms(self, pred_dict, pixel_nms_thresh):
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
                    self.calculate_dice_coefficient_pixel(
                        current_mask.squeeze(), comparison_mask.squeeze()
                    )
                    > pixel_nms_thresh
                ):
                    masks_to_remove.append(j)

        # Return a list of mask indices that are not marked for removal
        return [i for i in range(len(masks)) if i not in masks_to_remove]

    def polygonize_pixel_segmentations(self, pred_dict, poly_score_thresh, bounds):
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
            self.vectorize_mask_instance(
                c, pred_dict["scores"][i].detach().item(), transform
            )
            for i, c in enumerate(high_conf_classes)
        ]
        keep_masks = [i for i, poly in enumerate(pred_dict["polys"]) if poly]
        return pred_dict, keep_masks

    def vectorize_mask_instance(
        self, high_conf_mask: torch.Tensor, machine_confidence, transform
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
            multipoly, inf_idx = self.extract_geometry(dataset)

        return (
            geojson.Feature(
                geometry=multipoly,
                properties=dict(machine_confidence=machine_confidence, inf_idx=inf_idx),
            )
            if multipoly
            else None
        )

    def extract_geometry(self, dataset):
        """Extracts the geometries from a raster dataset."""

        shps = shapes(
            dataset.read(1).astype("uint8"), connectivity=8, transform=dataset.transform
        )
        geoms, inf_idxs = zip(*[s for s in shps if s[1] != self.background_class_idx])
        return MultiPolygon([shape(g) for g in geoms]), inf_idxs[0] if inf_idxs else 0

    def calculate_dice_coefficient_pixel(self, u, v):
        """
        Takes two pixel-confidence masks, and calculates how similar they are to each other
        Returns a value between 0 (no overlap) and 1 (identical)
        Utilizes an IoU style construction
        Can be used as NMS across classes for mutually-exclusive classifications
        """
        return 2 * torch.sum(torch.sqrt(torch.mul(u, v))) / (torch.sum(u + v))


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
        # SAR_stats = [0.2087162, 0.13736105]

        try:
            stack_tensors = [
                b64_image_to_array(record.image, tensor=True, to_float=True).unsqueeze(
                    0
                )
                # normalize_and_clamp(
                #    b64_image_to_array(record.image, tensor=True, to_float=True),
                #    mean=SAR_stats[0],
                #    std=SAR_stats[1],
                #    device=self.device,
                # ).unsqueeze(0)
                for record in inf_stack
            ]
            batch_tensor = torch.cat(stack_tensors, dim=0).to(self.device)
            logging.info(f"Batch tensor shape: {batch_tensor.shape}")
            return batch_tensor  # Only the tensor batch is needed for the model
        except Exception as e:
            logging.error("Error in preprocessing: %s", e, exc_info=True)
            raise

    def process_tiles(self, preprocessed_tensors):
        """
        Run inference on the loaded model
        """
        return self.model(preprocessed_tensors)

    def postprocess_tiles(self, raw_preds) -> List[InferenceResult]:
        """
        Applies a softmax function to the raw predictions from FASTAIUNET, serializes them,
        and returns a list of InferenceResults with serialized data.

        Args:
            raw_preds (List[torch.Tensor]): A list of 4D tensors representing predictions
                                            [batch_size, num_classes, height, width].

        Returns:
            List[InferenceResult]: A list of InferenceResults with serialized prediction data.
        """
        processed_preds = [
            torch.nn.functional.softmax(pred, dim=0) for pred in raw_preds
        ]

        inference_results = [
            InferenceResult(json_data=self.serialize(pred)) for pred in processed_preds
        ]
        return inference_results

    def serialize(self, pred):
        """
        Serializes a PyTorch tensor using base64 encoding after converting it with tensor_to_base64.

        Args:
            pred (torch.Tensor): The tensor to be serialized.

        Returns:
            str: A JSON string containing the base64 encoded tensor.
        """
        return json.dumps(tensor_to_base64(pred))

    def deserialize(self, json_string):
        """
        Deserializes a JSON string into a PyTorch tensor by decoding the base64 string within it and loading it as a tensor.

        Args:
            json_string (str): The JSON string containing the serialized tensor data.

        Returns:
            torch.Tensor: The deserialized PyTorch tensor.
        """
        base64_string = json.loads(json_string)
        return torch.load(BytesIO(b64decode(base64_string)))

    def postprocess_tileset(
        self, tileset_results: List[InferenceResultStack]
    ) -> geojson.FeatureCollection:
        """
        Stitches together multiple InferenceResultStacks from the FASTAIUNET model.

        Args:
            tileset_results: The list of InferenceResultStacks to stitch together.
        """
        logging.info("Stitching tiles into scene")
        scene_array_probs, transform = self.stitch(tileset_results)
        logging.info("Finding instances in scene")
        features_list = self.instantiate(scene_array_probs)
        logging.info("Reducing feature count")
        reduced_features = self.reduce_scene_features(features_list)
        return geojson.FeatureCollection(features=reduced_features)

    def stitch(self, tileset_results: List[InferenceResultStack]):
        """Merge arrays based on their geographical bounds and return the merged array and its bounds.

        Args:
            tileset_results:

        Returns:
            tuple: A tuple containing the merged numpy array and the bounds (min_x, min_y, max_x, max_y) of the merged area.
        """
        tile_probs_by_class = [
            self.deserialize(inf.json_data).detach().numpy()
            for inference_result_stack in tileset_results
            for inf in inference_result_stack.stack
        ]
        bounds = [
            bounds
            for inference_result_stack in tileset_results
            for bounds in inference_result_stack.bounds
        ]
        ds_tiles = []
        try:
            ds_tiles = [
                memfile_gtiff(nparray=tile_probs, bounds=bounds).open()
                for tile_probs, bounds in zip(tile_probs_by_class, bounds)
            ]

            logging.info("Merging tiles!")
            scene_array, transform = rio_merge(ds_tiles)
            return scene_array, transform
        finally:
            for ds in ds_tiles:
                ds.close()

    def instantiate(self, scene_array_probs):
        """
            Processes scene probablities to probabilities using softmax, excluding the background class,
            and generates features for each class index.

        Args:
            scene_array_probs (Tensor or numpy.ndarray): A tensor containing probabilities for each class in the scene,
                where the first dimension corresponds to class indices.

            Returns:
                list: A list of features, where each feature is derived from class probabilities,
                    excluding the background class. Each feature includes additional properties
                    such as the class index.
        """

        # Convert numpy.ndarray to PyTorch tensor if necessary
        if isinstance(scene_array_probs, np.ndarray):
            scene_array_probs = torch.tensor(scene_array_probs)

        features = []
        for inf_idx, cls_probs in enumerate(scene_array_probs):
            if inf_idx != self.background_class_idx:
                features.extend(
                    self.instances_from_probs(
                        cls_probs,
                        p1=self.model_dict["thresholds"]["bbox_score_thresh"],
                        p2=self.model_dict["thresholds"]["poly_score_thresh"],
                        p3=self.model_dict["thresholds"]["pixel_score_thresh"],
                        addl_props={"inf_idx": inf_idx},
                    )
                )

        return features

    def reduce_scene_features(self, features):
        """
        Reduces the number of features in a scene by applying Non-Maximum Suppression (NMS)
        based on a global pixel threshold to remove similar overlapping features.

        This function utilizes a defined pixel NMS threshold from the model's configuration
        to determine which features are kept based on their uniqueness.

        Parameters:
            features (list of dict): A list of feature dictionaries. Each feature dictionary
                should contain at least the 'properties' key with necessary information.

        Returns:
            list of dict: A reduced list of features after applying the pixel NMS.
        """
        reduced_features = self.keep_by_global_pixel_nms(
            features, self.model_dict["thresholds"]["pixel_nms_thresh"]
        )
        return reduced_features

    def keep_by_global_pixel_nms(self, features, pixel_nms_thresh):
        """
        Filters out features that are similar based on the Dice coefficient, which is
        calculated between each pair of features. This is intended to perform a global
        Non-Maximum Suppression (NMS) where features that are too similar to each other
        (above the given threshold) are removed.

        Parameters:
            features (list of dict): A list of features to evaluate. Each feature must have
                a 'properties' dictionary containing at least 'inf_idx', which indicates
                the class index of the feature.
            pixel_nms_thresh (float): The similarity threshold for the Dice coefficient. If the
                Dice coefficient between two features exceeds this value, the later feature
                in the list is considered for removal.

        Returns:
            list of dict: A list of features that are not marked for removal. Features are
                considered non-redundant if their similarity (by Dice coefficient) to other
                features does not exceed the specified threshold.
        """

        feats_to_remove = []

        for i, current_feat in enumerate(features):  # Loop through all feats
            # Skip if the feat is already marked for removal
            if i in feats_to_remove:
                continue

            # Check similarity against all subsequent feats
            for j, comparison_feat in enumerate(features[i + 1 :], start=i + 1):
                # Skip if the feat is already marked for removal
                if j in feats_to_remove:
                    continue
                # Skip if the feat came from the same class (cannot overlap in FASTAIUNET by definition)
                if (
                    current_feat["properties"]["inf_idx"]
                    == comparison_feat["properties"]["inf_idx"]
                ):
                    continue

                # Calculate Dice Coefficient; if the similarity is too high, mark feat for removal
                if (
                    self.calculate_dice_coefficient_geojson(
                        current_feat, comparison_feat
                    )
                    > pixel_nms_thresh
                ):
                    feats_to_remove.append(j)

        # Return a list of feats that are not marked for removal
        return [
            feat for i, feat in enumerate(features) if i not in set(feats_to_remove)
        ]

    def instances_from_probs(self, raster, p1, p2, p3, addl_props={}):
        """
        Converts raster predictions to GeoJSON based on probability thresholds.
        Effectively performs grouping, filtering, and trimming of a probability raster, to produce independent features.

        Args:
            raster (np.array): The input raster array to be processed.
            p1 (float): The lowest probability, used to group adjacent polygons into multipolygons. lower value = fewer groups
            p2 (float): The middle probability, used to trim the final polygons size. lower value = coarser polygons
            p3 (float): The highest probability, used to discard polygons that don't reach sufficient confidence. higher value = more restrictive
            Sample values: p1, p2, p3 = 0.1, 0.5, 0.95
            Analogous to bbox_score_thresh, poly_score_thresh, pixel_score_thresh
        Returns:
            GeoJSON: A GeoJSON feature collection of the processed predictions.
        """

        raster = raster.float().detach().numpy()

        # Label components based on p3 to find peaks
        p1_islands, p1_island_count = label(raster >= p1)
        # logging.info(f"p1_island_count: {p1_island_count}")
        p3_islands, p3_island_count = label(raster >= p3)
        # logging.info(f"p3_island_count: {p3_island_count}")

        # Initialize an empty set for unique p1 labels corresponding to p3 components
        reduced_labels = set()

        # Iterate over each p3 component
        for i in range(1, p3_island_count + 1):
            p3_island_mask = p3_islands == i
            p1_label_at_p3 = p1_islands[p3_island_mask].flat[
                0
            ]  # Take the first pixel's p1 label
            reduced_labels.add(p1_label_at_p3)
        # logging.info(f"reduced_labels: {len(reduced_labels)}")

        features = []
        # Process into feature collections based on unique p1 labels
        for p1_label in reduced_labels:
            mask = (p1_islands == p1_label) & (raster >= p2)  # Apply p2 trimming
            masked_raster = raster[mask]
            shapes = rasterio.features.shapes(mask.astype(np.uint8), mask=mask)
            polygons = [shape(geom) for geom, value in shapes if value == 1]

            # Ensure there are polygons left after trimming to process into a MultiPolygon
            if polygons:
                multipolygon = MultiPolygon(polygons)
                features.append(
                    geojson.Feature(
                        geometry=multipolygon,
                        properties={
                            "instance_id": int(p1_label),
                            "mean_conf": float(np.mean(masked_raster)),
                            "median_conf": float(np.median(masked_raster)),
                            "max_conf": float(np.max(masked_raster)),
                            "pixel_count": int(masked_raster.size),
                            "machine_confidence": float(np.median(masked_raster)),
                            **addl_props,
                        },
                    )
                )

        return features

    def calculate_dice_coefficient_geojson(self, geojson1, geojson2):
        """
        Takes two GeoJSON feature dictionaries and calculates how similar they are to each other.
        Returns a value between 0 (no overlap) and 1 (identical).
        Utilizes an intersection over union (IoU) style construction.
        """
        # Convert GeoJSON features to Shapely polygons
        polygon1 = shape(geojson1["geometry"])
        polygon2 = shape(geojson2["geometry"])

        # Calculate the intersection and union of both polygons
        intersection = polygon1.intersection(polygon2)
        union = polygon1.union(polygon2)

        # Calculate Dice Coefficient using the area of the intersection and union
        if union.area == 0:
            return 0
        dice_coefficient = 2 * intersection.area / (polygon1.area + polygon2.area)
        return dice_coefficient


def get_model(
    model_dict,
    model_path_local=os.getenv("MODEL_PATH_LOCAL"),
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
        return encoded
    return memfile


# def probs_to_classes(out_batch_probs, conf_threshold=0.0):
#     """
#     Convert probabilitiess from a neural network batch output to class predictions based on probability confidence.

#     Parameters:
#     - out_batch_probs (Tensor): A tensor containing probs for each class, typically of shape (num_classes, num_samples).
#     - conf_threshold (float, optional): A threshold value for confidence. Class predictions with confidence below this value will be set to 0. Default is 0, meaning all predictions are returned regardless of confidence.

#     Returns:
#     - tuple(Tensor, Tensor): A tuple containing two tensors:
#         - The first tensor (`conf`) contains the maximum probabilities (confidences) for each sample.
#         - The second tensor (`classes`) contains the class indices of the highest probability for each sample. If `conf_threshold` is used and the confidence of a prediction is below the threshold, the class index is set to 0.

#     This function identifies the maximum probability and corresponding class for each sample, and applies a confidence threshold if specified.
#     """
#     conf, classes = torch.max(out_batch_probs, 0)
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


def b64_image_to_array(image: str, tensor: bool = False, to_float=False):
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

        if to_float:
            np_img = dtype_to_float(np_img)
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


def dtype_to_float(data, dtype=np.float32):
    """
    Convert numerical data to a specified floating-point type and normalize if data type is uint8.

    This function supports inputs that are either numpy arrays or PyTorch tensors. If the input data type
    is uint8, the function normalizes the data by dividing by 255.0, which is common practice for image data.
    Otherwise, it simply converts the data to the specified floating-point type.

    Parameters:
    data (Union[np.ndarray, torch.Tensor]): The data to convert. Must be either a numpy array or a PyTorch tensor.
    dtype (data-type, optional): The target data type for the conversion, defaults to np.float32. The dtype
                                 should be a floating-point type as defined by numpy or torch.

    Returns:
    Union[np.ndarray, torch.Tensor]: The converted and potentially normalized data as a numpy array or a PyTorch tensor.

    Raises:
    TypeError: If `data` is neither a numpy array nor a PyTorch tensor.

    Examples:
    >>> import numpy as np
    >>> import torch
    >>> array = np.array([0, 128, 255], dtype=np.uint8)
    >>> print(dtype_to_float(array))
    [0. , 0.50196078, 1. ]

    >>> tensor = torch.tensor([0, 128, 255], dtype=torch.uint8)
    >>> print(dtype_to_float(tensor))
    tensor([0.0000, 0.5020, 1.0000])
    """

    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            return (data.astype(dtype) / 255.0).astype(dtype)
        else:
            return data.astype(dtype)
    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.uint8:
            return (data.to(dtype) / 255.0).to(dtype)
        else:
            return data.to(dtype)
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")


def reproject_to_utm(gdf_wgs84):
    """Finds utm projection for a WGS84 gdf"""
    utm_crs = gdf_wgs84.estimate_utm_crs(datum_name="WGS 84")
    return gdf_wgs84.to_crs(utm_crs)


def tensor_to_base64(tensor):
    """
    Convert a PyTorch tensor to a base64 string by first saving it to a BytesIO buffer using torch.save, then encoding the buffer contents.

    Args:
        tensor (torch.Tensor): The tensor to be converted.

    Returns:
        str: A base64 encoded string of the tensor.
    """
    buffer = BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return b64encode(buffer.read()).decode("utf-8")
