"""Clients for other cloud run functions"""
from base64 import b64encode
from typing import List

import httpx
import morecantile
import numpy as np
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster

from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceInput, InferenceResult
from cerulean_cloud.tiling import TMS


def img_array_to_b64_image(img_array: np.ndarray) -> str:
    """convert input b64image to torch tensor"""
    img_array = img_array.astype("int8")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=img_array.shape[2],
            dtype=img_array.dtype,
            width=img_array.shape[0],
            height=img_array.shape[1],
        ) as dataset:
            dataset.write(reshape_as_raster(img_array))
        img_bytes = memfile.read()

    return b64encode(img_bytes).decode("ascii")


class CloudRunInferenceClient:
    """Client for inference cloud run"""

    def __init__(self, url: str, titiler_client):
        """init"""
        self.url = url
        self.titiler_client = titiler_client

    def get_base_tile_inference(
        self, sceneid: str, tile: morecantile.Tile, rescale=(0, 100)
    ) -> InferenceResult:
        """fetch inference for base tiles"""
        img_array = self.titiler_client.get_base_tile(
            sceneid=sceneid, tile=tile, scale=2, rescale=rescale
        )

        encoded = img_array_to_b64_image(img_array)

        inference_input = InferenceInput(image=encoded, bounds=TMS.bounds(tile))
        res = httpx.post(
            self.url + "predict/", data=inference_input.json(), timeout=None
        )
        return InferenceResult(**res.json())

    def get_offset_tile_inference(
        self, sceneid: str, bounds: List[float], rescale=(0, 100)
    ) -> InferenceResult:
        """fetch inference for offset tiles"""
        img_array = self.titiler_client.get_offset_tile(
            sceneid, *bounds, rescale=rescale
        )

        encoded = img_array_to_b64_image(img_array)

        inference_input = InferenceInput(image=encoded, bounds=bounds)
        res = httpx.post(
            self.url + "predict/", data=inference_input.json(), timeout=None
        )
        return InferenceResult(**res.json())
