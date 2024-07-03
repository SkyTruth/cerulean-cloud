import os
from cerulean_cloud.tiling import offset_bounds_from_base_tiles
from cerulean_cloud.titiler_client import TitilerClient
from cerulean_cloud.models import get_model, BaseModel
from cerulean_cloud.cloud_run_orchestrator.handler import (
    offset_group_shape_from_base_tiles,
    group_bounds_from_list_of_bounds,
)
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient


class Task:
    def __init__(
        self,
        model_dict,
        base_tiles,
        layers,
        sceneid,
        scale,
        gamma_correction,
        tiling_offset,
        zoom,
    ):

        self.model_dict = model_dict
        self.base_tiles = base_tiles
        self.layers = layers
        self.sceneid = sceneid
        self.scale = scale
        self.gamma_correction = gamma_correction
        self.tiling_offset = tiling_offset
        self.zoom = zoom

        self.titiler_client = TitilerClient(url=os.getenv("TITILER_URL"))
        self.model = get_model(model_dict)

        self.tile_bounds_list = offset_bounds_from_base_tiles(
            self.base_tiles, offset_amount=self.tiling_offset
        )

        tileset_hw_pixels = offset_group_shape_from_base_tiles(base_tiles, self.scale)
        tileset_envelope_bounds = group_bounds_from_list_of_bounds(self.tile_bounds_list)

        self.cloud_run_inference =CloudRunInferenceClient(
            url=os.getenv("INFERENCE_URL"),
            titiler_client=self.titiler_client,
            sceneid=self.sceneid,
            tileset_envelope_bounds=tileset_envelope_bounds,
            image_hw_pixels=tileset_hw_pixels,
            layers=layers,
            scale=scale,
            model_dict=model_dict,
        )

        self.tileset_results = None
        self.postprocess_results = None

    async def run_parallel_inference(self):
        self.tileset_results = await self.cloud_run_inference.run_parallel_inference(
            self.tile_bounds_list
        )

    def postprocess_tileset(self):
        self.postprocess_results = self.model.postprocess_tileset(
            self.tileset_results, self.tile_bounds_list
        )


class Ensemblers:
    def __init__(self, task_list) -> None:
        pass

    def nms_ensemble(task_list):
        task_model = task_list[0].model
        tileset_fc_list = [task.postprocess_results for task in task_list]
        return task_model.nms_feature_reduction(
            features=tileset_fc_list, min_overlaps_to_keep=1
        )

        pass
