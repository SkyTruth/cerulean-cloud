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
from typing import List, Union

import geojson
import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
import torch
import torchvision  # noqa
from rasterio.features import geometry_mask, shapes
from rasterio.io import MemoryFile
from rasterio.transform import Affine
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
                int(key)
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
        preprocessed_tensors = self.preprocess_tiles(inf_stack)
        raw_preds = self.process_tiles(preprocessed_tensors)  # Run inference
        inference_results = self.postprocess_tiles(raw_preds, preprocessed_tensors)
        return InferenceResultStack(stack=inference_results)

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

    def postprocess_tiles(
        self, raw_preds, preprocessed_tensors=None
    ) -> List[InferenceResult]:
        """
        Process and stack the raw_preds of predictions. This method should be implemented by subclasses.

        Args:
            raw_preds: The results to be processed and stacked.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def postprocess_tileset(
        self,
        tileset_results: List[InferenceResultStack],
        tileset_bounds: List[List[List[float]]],
    ) -> geojson.FeatureCollection:
        """
        Process the InferenceResultStack into a scene and then return a FC. This method should be implemented by subclasses.

        Args:
            tileset_results: The input data stack for inference.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def nms_feature_reduction(
        self,
        features: Union[geojson.FeatureCollection, List[geojson.FeatureCollection]],
        min_overlaps_to_keep: int = 0,
        in_class_only: bool = False,
    ) -> geojson.FeatureCollection:
        """
        Performs ensemble inference on a list of geojson Features to eliminate overlapping features based on a non-maximum suppression approach using IoU and inclusion thresholds. Features with fewer overlaps than a specified threshold are also discarded.

        Parameters:
        - feature_list: A list of geojson Features.
        - poly_nms_thresh: The threshold for the IoU above which one of the overlapping features is removed.
        - min_overlaps_to_keep: The minimum number of overlaps a feature must have to be retained.
        - in_class_only: Determines whether the NMS should only apply to like-classed instances, or classes should supress each other

        Returns:
        - A geojson FeatureCollection containing the retained features.
        """
        feature_list = []
        if isinstance(features, geojson.FeatureCollection):
            feature_list.extend(features["features"])
        elif isinstance(features, list) and all(
            isinstance(f, geojson.FeatureCollection) for f in features
        ):
            feature_list.extend([f for fc in features for f in fc["features"]])

        if not feature_list:
            return geojson.FeatureCollection([])

        # Filter out features with None geometry before processing
        feature_list = [
            feature for feature in feature_list if feature["geometry"] is not None
        ]

        # Precompute the areas of all features to optimize geometry operations
        gdf = gpd.GeoDataFrame(
            [feature["properties"] for feature in feature_list],
            geometry=[
                shape(feature["geometry"])
                for feature in feature_list
                if feature["geometry"] is not None
            ],
            crs="EPSG:4326",
        )
        gdf = reproject_to_utm(gdf)
        gdf["area"] = gdf.area

        # Initialize a set for efficient tracking of features to be removed
        feats_to_remove = []

        # If the feature has fewer overlaps than required, mark it for removal
        gdf["overlaps"] = gdf.apply(
            lambda x: sum(x.geometry.intersects(y) for y in gdf.geometry) - 1, axis=1
        )
        feats_to_remove.extend(gdf[gdf["overlaps"] < min_overlaps_to_keep].index)

        for i, feat_i in gdf.iterrows():
            # Compare the current feature against all subsequent features
            for j, feat_j in gdf.iloc[i + 1 :].iterrows():
                if j in feats_to_remove or i in feats_to_remove:
                    # Skip processing for features already marked for removal
                    continue

                if in_class_only and (feat_i["inf_idx"] != feat_j["inf_idx"]):
                    # Skip pairs that don't have matching classes
                    continue

                if not feat_i.geometry.intersects(feat_j.geometry):
                    continue

                # Compute intersection and union areas for IoU
                intersection = feat_i.geometry.intersection(feat_j.geometry).area
                union = feat_i["area"] + feat_j["area"] - intersection

                # Decide which feature to remove based on the IoU threshold
                iou = intersection / union
                if iou > self.model_dict["thresholds"]["poly_nms_thresh"]:
                    # Choose to remove the feature with lower confidence
                    if feat_i["machine_confidence"] <= feat_j["machine_confidence"]:
                        feats_to_remove.append(i)
                    else:
                        feats_to_remove.append(j)
                # Check for substantial inclusion and remove the encompassed feature
                elif intersection > 0.5 * feat_i["area"]:
                    feats_to_remove.append(i)
                elif intersection > 0.5 * feat_j["area"]:
                    feats_to_remove.append(j)

        # Collect features that are not marked for removal
        retained_features = [
            f for i, f in enumerate(feature_list) if i not in feats_to_remove
        ]

        # Return a new geojson FeatureCollection containing only the retained features
        return geojson.FeatureCollection(retained_features)


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

    def postprocess_tiles(
        self, raw_preds, preprocessed_tensors=None
    ) -> List[InferenceResult]:
        """
        Process and stack the raw_preds of MASKRCNN predictions.

        Args:
            raw_preds: The results to be processed and stacked.
        """
        reduced_preds = self.reduce_preds(
            raw_preds,
            bbox_score_thresh=self.model_dict["thresholds"]["bbox_score_thresh"],
        )  # If we don't reduce here, the cloud_run crashes on preds with more than ~8 bboxes
        inference_results = [
            InferenceResult(json_data=self.serialize(pred)) for pred in reduced_preds
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
        for key in ["boxes", "labels", "scores"]:
            pred[key] = pred[key].tolist()  # Tensors are not json serializable
        pred["masks_b64"] = [tensor_to_base64(mask) for mask in pred.pop("masks")]
        json_string = json.dumps(pred)
        return json_string

    def deserialize(self, json_string):
        """
        Deserializes a JSON string into a prediction dictionary, decoding 'masks' from base64 strings
        back to PyTorch tensors using torch.load, and converting 'boxes', 'labels', and 'scores' back to tensors

        Parameters:
        - json_string (str): The JSON string containing the serialized prediction data.

        Returns:
        - dict: The original prediction dictionary with 'masks' restored as PyTorch tensors and
                'boxes', 'labels', 'scores' converted back to tensors if they were originally tensors.
        """
        pred = json.loads(json_string)
        pred["masks"] = [
            torch.load(BytesIO(b64decode(b64_str))) for b64_str in pred.pop("masks_b64")
        ]
        for key in ["boxes", "labels", "scores"]:
            pred[key] = torch.tensor(pred[key])

        return pred

    def postprocess_tileset(
        self,
        tileset_results: List[InferenceResultStack],
        tileset_bounds: List[List[List[float]]],
    ) -> geojson.FeatureCollection:
        """
        Post-process a list of tileset results to create a unified geojson feature collection.
        This includes reducing the number of features per tile, stitching tiles into a single scene,
        and further reducing the number of features in the scene.

        Args:
            tileset_results (List[InferenceResultStack]): A list of inference results for each tile.

        Returns:
            geojson.FeatureCollection: A geojson feature collection representing the processed and combined geographical data.
        """

        logging.info("Reducing feature count on tiles")
        scene_polys = self.reduce_tile_features(tileset_results, tileset_bounds)
        logging.info("Stitching tiles into scene")
        feature_collection = self.stitch(scene_polys)
        logging.info("Reducing feature count on scene")
        reduced_feature_collection = self.nms_feature_reduction(feature_collection)
        return reduced_feature_collection

    def reduce_tile_features(
        self,
        tileset_results: List[InferenceResultStack],
        tileset_bounds: List[List[List[float]]],
    ):
        """
        Reduces the number of prediction features within each tile by applying thresholds and generating polygons.

        Args:
            tileset_results (List[InferenceResultStack]): A list containing stacks of inference results for tiles.

        Returns:
            List[geojson.Feature]: A list of geojson features representing the reduced set of features per tile.
        """

        bounds_list = []
        pred_list = []
        for i, inf_result_stack in enumerate(tileset_results):
            if inf_result_stack.stack:
                for j, inference_result in enumerate(inf_result_stack.stack):
                    pred_list.append(self.deserialize(inference_result.json_data))
                    bounds_list.append(tileset_bounds[i][j])

        reduced_pred_list = self.reduce_preds(
            pred_list, **self.model_dict["thresholds"]
        )

        scene_polys = []
        for pred_dict, bounds in zip(reduced_pred_list, bounds_list):
            # generate vectorized polygons from predictions
            # adds "polys" to pred_dict
            if self.model_dict["thresholds"]["poly_score_thresh"] is not None:
                pred_dict, keep = self.polygonize_pixel_segmentations(
                    pred_dict,
                    self.model_dict["thresholds"]["poly_score_thresh"],
                    bounds,
                )
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)
            scene_polys.extend(pred_dict["polys"])

        return scene_polys

    def stitch(
        self,
        scene_polys: List[geojson.Feature],
        proximity_meters: int = 1000,  # group nearby polygons
        closing_meters: int = 500,  # fill gaps between very close polygons
        opening_meters: int = 0,
    ):
        """
        Stitches together multiple geojson features into a single feature collection, optionally applying spatial operations
        like buffering to connect close polygons and filling gaps.

        Args:
            scene_polys (List[geojson.Feature]): List of geojson features to be stitched.
            proximity_meters (int, optional): Buffer radius to apply for connecting nearby polygons. Default is 0.
            closing_meters (int, optional): Buffer radius to apply for closing gaps between polygons. Default is 0.
            opening_meters (int, optional): Buffer radius to apply for opening inside the polygons. Default is 0.

        Returns:
            Dict: A geojson-compatible dictionary representing the stitched and optionally modified features.
        """

        if len(scene_polys) == 0:
            # No inferences found
            return geojson.FeatureCollection(features=[])

        # We reproject to UTM for processing. This assumes that all offset images will either be in the same UTM zone as
        # the input image chip, or that the difference that arise from an offset crossing into a second UTM zone will
        # have little or no impact on comparison to the original image.
        gdf = gpd.GeoDataFrame.from_features(scene_polys, crs="4326")
        gdf = reproject_to_utm(gdf)
        final_gdf = gdf.copy()

        # Expand the geometry of each feature to connect with neighboring instances
        gdf["geometry"] = gdf.buffer(proximity_meters)

        # Ensure the 'inf_idx' is the same before joining
        joined = gpd.sjoin(gdf, gdf, predicate="intersects")
        joined = joined[joined["inf_idx_left"] == joined["inf_idx_right"]].reset_index()

        # Create a graph where each node represents a feature and edges represent overlaps/intersections
        G = nx.from_pandas_edgelist(joined, "index", "index_right")

        # For each connected component in the graph, assign a group index and count its features
        group_mapping = {
            feature: group
            for group, component in enumerate(nx.connected_components(G))
            for feature in component
        }

        # Map the group indices and counts back to the GeoDataFrame
        final_gdf["group_index"] = final_gdf.index.map(group_mapping)
        final_gdf["mean_conf"] = final_gdf["machine_confidence"]
        final_gdf["median_conf"] = final_gdf["machine_confidence"]
        final_gdf["max_conf"] = final_gdf["machine_confidence"]

        # Dissolve overlapping features into one based on their group index and calculate the median confidence and maximum inference index
        dissolved_gdf = final_gdf.dissolve(
            by="group_index",
            aggfunc={
                "machine_confidence": "median",
                "inf_idx": lambda x: int(x.mode()[0]),
                "mean_conf": "mean",
                "median_conf": "median",
                "max_conf": "max",
            },
        )

        # If set, apply a morphological 'closing' operation to the geometries
        if closing_meters is not None:
            dissolved_gdf["geometry"] = dissolved_gdf.buffer(closing_meters).buffer(
                -closing_meters
            )

        # If set, apply a morphological 'opening' operation to the geometries
        if opening_meters is not None:
            dissolved_gdf["geometry"] = dissolved_gdf.buffer(-opening_meters).buffer(
                opening_meters
            )

        # Reproject the GeoDataFrame back to WGS 84 CRS
        result = dissolved_gdf.to_crs(crs="4326")

        # Clean up potentially memory heavy assets
        del dissolved_gdf
        del gdf
        del final_gdf
        del joined

        return geojson.FeatureCollection(features=result.__geo_interface__)

    def reduce_preds(
        self,
        pred_list,
        bbox_score_thresh=None,
        pixel_score_thresh=None,
        pixel_nms_thresh=None,
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

        Returns:
            A list of reduced and processed prediction dictionaries.
        """
        reduced_preds = []
        for pred_dict in pred_list:
            # remove instances with low scoring boxes
            if bbox_score_thresh is not None:
                keep = self.keep_by_bbox_score(pred_dict, bbox_score_thresh)
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            # remove instances with low scoring pixels
            if pixel_score_thresh is not None:
                keep = self.keep_by_pixel_score(pred_dict, pixel_score_thresh)
                pred_dict = self.keep_boxes_by_idx(pred_dict, keep)

            # non-maximum suppression, in-class, using pixels rather than bboxes
            if pixel_nms_thresh is not None:
                keep = self.keep_by_pixel_nms(
                    pred_dict, pixel_nms_thresh, in_class_only=True
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

    def keep_by_pixel_nms(self, pred_dict, pixel_nms_thresh, in_class_only):
        """
        Applies non-maximum suppression (NMS) based on the Dice coefficient between pixel masks to filter out overlapping masks.
        This function iteratively compares each mask against others and removes those with a high degree of overlap
        based on a defined threshold, optionally considering only masks within the same class.

        Args:
            pred_dict (dict): A dictionary containing 'masks' and 'labels'. 'masks' is a list of masks,
                            and 'labels' is a list of class labels corresponding to each mask.
            pixel_nms_thresh (float): The Dice coefficient threshold above which masks are considered overlapping and
                                    hence one of them is removed.
            in_class_only (bool): If True, NMS is applied only to masks within the same class.

        Returns:
            List[int]: A list of indices of masks that remain after applying pixel NMS.
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
                if in_class_only and (pred_dict["labels"][i] != pred_dict["labels"][j]):
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
        shps = shapes(
            high_conf_mask.detach().numpy().astype("int16"),
            connectivity=8,
            transform=transform,
        )
        shps = [s for s in shps]
        if len(shps) == 1:  # Only the background class was observed
            return geojson.Feature([])
        geoms, inf_idxs = zip(*[s for s in shps if s[1] != self.background_class_idx])
        multipoly, inf_idx = MultiPolygon([shape(g) for g in geoms]), (
            inf_idxs[0] if inf_idxs else 0
        )
        return (
            geojson.Feature(
                geometry=multipoly,
                properties=dict(machine_confidence=machine_confidence, inf_idx=inf_idx),
            )
            if multipoly
            else None
        )

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

    def postprocess_tiles(
        self, raw_preds, preprocessed_tensors=None
    ) -> List[InferenceResult]:
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

        if preprocessed_tensors is not None:
            data_mask = preprocessed_tensors != 0  # Pixels that are not zero
            for i, probs in enumerate(processed_preds):
                probs[1:, :, :] = probs[1:, :, :] * data_mask[i]
                # Zero out the background class; applied to each channel from index 1 onwards using broadcasting

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
        self,
        tileset_results: List[InferenceResultStack],
        tileset_bounds: List[List[List[float]]],
    ) -> geojson.FeatureCollection:
        """
        Stitches together multiple InferenceResultStacks from the FASTAIUNET model.

        Args:
            tileset_results: The list of InferenceResultStacks to stitch together.
        """
        logging.info("Stitching tiles into scene")
        scene_array_probs, transform = self.stitch(tileset_results, tileset_bounds)
        logging.info("Finding instances in scene")
        feature_collection = self.instantiate(scene_array_probs, transform)
        logging.info("Reducing feature count on scene")
        reduced_feature_collection = self.nms_feature_reduction(feature_collection)
        return reduced_feature_collection

    def stitch(
        self,
        tileset_results: List[InferenceResultStack],
        tileset_bounds: List[List[List[float]]],
    ):
        """Manually merge arrays based on their geographical bounds.

        Args:
            tileset_results: The list of InferenceResultStacks to stitch together.
            tileset_bounds: The list of bounds for each InferenceResultStack.

        Returns:
            tuple: A tuple containing the merged numpy array and the transform of the merged area.
        """
        bounds_list = []
        tile_probs_by_class = []
        for i, inf_result_stack in enumerate(tileset_results):
            if inf_result_stack.stack:
                for j, inference_result in enumerate(inf_result_stack.stack):
                    tile_probs_by_class.append(
                        self.deserialize(inference_result.json_data).detach().numpy()
                    )
                    bounds_list.append(tileset_bounds[i][j])
            # XXX BUG Not sure why, but on certain scenes rio_merge(ds_tiles) errors out.
            # Notably, tileset_bounds and tileset_results are both empty...???
            # e.g. S1A_IW_GRDH_1SDV_20240802T025056_20240802T025125_055028_06B441_281B and S1A_IW_GRDH_1SDV_20240728T024243_20240728T024312_054955_06B1BC_0458
            # Note: might be related to the is_tile_over_water() function NOT thinking that the Caspian Sea is water,
            # and therefore returning an empty list. If this is the case, then it's unclear why it's not throwing IndexError
        # Determine overall bounds
        min_x = min(b[0] for b in bounds_list)
        min_y = min(b[1] for b in bounds_list)
        max_x = max(b[2] for b in bounds_list)
        max_y = max(b[3] for b in bounds_list)

        # Get resolution from one tile
        sample_tile = tile_probs_by_class[0]
        tile_height, tile_width = sample_tile.shape[1], sample_tile.shape[2]
        tile_bounds = bounds_list[0]
        res_x = (tile_bounds[2] - tile_bounds[0]) / tile_width
        res_y = (
            tile_bounds[3] - tile_bounds[1]
        ) / tile_height  # Negative because Y decreases

        # Calculate final array dimensions
        final_width = int(np.ceil((max_x - min_x) / res_x))
        final_height = int(np.ceil((max_y - min_y) / res_y))
        num_classes = sample_tile.shape[0]

        # Pre-allocate final array
        scene_array_probs = np.zeros(
            (num_classes, final_height, final_width), dtype=sample_tile.dtype
        )
        # Place each tile into the final array
        for tile_probs, bounds in zip(tile_probs_by_class, bounds_list):
            x_offset = int(np.ceil((bounds[0] - min_x) / res_x))
            y_offset = int(np.ceil((max_y - bounds[3]) / res_y))

            tile_height, tile_width = tile_probs.shape[1], tile_probs.shape[2]

            # Handle potential overlaps if necessary here
            scene_array_probs[
                :, y_offset : y_offset + tile_height, x_offset : x_offset + tile_width
            ] = tile_probs

        # Create the transform
        transform = Affine.translation(min_x, max_y) * Affine.scale(res_x, -res_y)

        return scene_array_probs, transform

    def instantiate(self, scene_array_probs, transform):
        """
        Converts scene probability arrays into a GeoJSON FeatureCollection of detected instances, excluding the background class. It processes each class's probability, transforms them to features, and applies specified thresholds from the model's dictionary.

        Parameters:
        - scene_array_probs (np.ndarray or torch.Tensor): Array containing per-class instance probabilities.
        - transform: A transformation function or operation to apply to each instance feature.

        Returns:
        - geojson.FeatureCollection: A collection of geojson features representing detected instances.
        """
        # Convert numpy.ndarray to PyTorch tensor if necessary
        if isinstance(scene_array_probs, np.ndarray):
            scene_array_probs = torch.tensor(scene_array_probs)

        classes_to_consider = [1, 2, 3]
        scene_oil_probs = scene_array_probs[classes_to_consider].sum(0)
        features = self.instances_from_probs(
            scene_oil_probs,
            p1=self.model_dict["thresholds"]["bbox_score_thresh"],
            p2=self.model_dict["thresholds"]["poly_score_thresh"],
            p3=self.model_dict["thresholds"]["pixel_score_thresh"],
            transform=transform,
        )
        for feat in features:
            mask = geometry_mask(
                [shape(feat["geometry"])],
                out_shape=scene_oil_probs.shape,
                transform=transform,
                invert=True,
            )
            cls_sums = [
                cls_probs[mask].detach().sum()
                for cls_probs in scene_array_probs[classes_to_consider]
            ]
            feat["properties"]["inf_idx"] = classes_to_consider[
                cls_sums.index(max(cls_sums))
            ]

        return geojson.FeatureCollection(features=features)

    def instances_from_probs(
        self,
        raster,
        p1,
        p2,
        p3,
        transform=Affine.identity(),
        addl_props={},
        discard_edge_polygons_buffer=0.005,
    ):
        """
        Converts raster predictions to GeoJSON based on probability thresholds.
        Effectively performs grouping, filtering, and trimming of a probability raster, to produce independent features.

        Args:
            raster (np.array): The input raster array to be processed.
            p1 (float): The lowest probability, used to group adjacent polygons into multipolygons. lower value = fewer groups
            p2 (float): The middle probability, used to trim the final polygons size. lower value = coarser polygons
            p3 (float): The highest probability, used to discard polygons that don't reach sufficient confidence. higher value = more restrictive
            transform (Affine): The transform to apply to the raster.
            addl_props (dict): Additional properties to add to the features.
            discard_edge_polygons_buffer (float): The buffer distance to discard polygons that are too close to the edge of the scene.
            Sample values: p1, p2, p3 = 0.1, 0.5, 0.95
            Analogous to bbox_score_thresh, poly_score_thresh, pixel_score_thresh
        Returns:
            GeoJSON: A GeoJSON feature collection of the processed predictions.
        """

        def overlap_percent(a, b):
            """
            Calculate the percentage of overlap between two polygons.
            This is different from IoU, because it is not symmetric.
            """
            if not a.intersects(b):  # Avoid unnecessary intersection computation
                return 0.0
            elif a.within(b):  # Avoid unnecessary intersection computation
                return 1.0
            else:
                return a.intersection(b).area / a.area

        raster = raster.float().detach().numpy()

        reduced_labels = set()  # Initialize an empty set for unique p1 labels
        p1_islands, p1_island_count = label(raster >= p1)
        for label_num in range(1, p1_island_count + 1):
            p1_label_mask = p1_islands == label_num
            if np.any(raster[p1_label_mask] >= p3):
                reduced_labels.add(label_num)

        zero_mask = raster == 0  # Find all pixels that are zero (i.e. nodata value)
        shapes = rasterio.features.shapes(
            zero_mask.astype(np.uint8), mask=zero_mask, transform=transform
        )
        polygons = [shape(geom) for geom, value in shapes if value == 1]
        scene_edge = MultiPolygon(polygons).buffer(discard_edge_polygons_buffer)

        # Process into feature collections based on unique p1 labels
        features = []
        for p1_label in reduced_labels:
            mask = (p1_islands == p1_label) & (raster >= p2)  # Apply p2 trimming
            masked_raster = raster[mask]
            shapes = rasterio.features.shapes(
                mask.astype(np.uint8), mask=mask, transform=transform
            )
            polygons = [shape(geom) for geom, value in shapes if value == 1]
            polygons = [p for p in polygons if overlap_percent(p, scene_edge) <= 0.5]
            # Ensure there are polygons left after trimming to process into a MultiPolygon
            if polygons:
                multipolygon = MultiPolygon(polygons)
                features.append(
                    geojson.Feature(
                        geometry=multipolygon,
                        properties={
                            "mean_conf": float(np.mean(masked_raster)),
                            "median_conf": float(np.median(masked_raster)),
                            "max_conf": float(np.max(masked_raster)),
                            "machine_confidence": float(np.median(masked_raster)),
                            **addl_props,
                        },
                    )
                )

        return features


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
    torch.save(tensor.clone(), buffer)
    # PYTHON BUG If you don't clone() the tensor, then the (1, 512, 512) tensor will be recorded in N times the amount of memory, where N is the number of output predictions this tensor was extracted from?!? This causes N predictions to be require NxN (N^2) data for storing!
    buffer.seek(0)
    return b64encode(buffer.read()).decode("utf-8")
