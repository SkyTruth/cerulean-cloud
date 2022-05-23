"""Clients for other cloud run functions"""
from typing import List

import httpx
import morecantile
from schema import InferenceInput, InferenceResult


class CloudRunInferenceClient:
    """Client for inference cloud run"""

    def __init__(self, url: str):
        """init"""
        self.url = url

    def get_base_tile_inference(self, tile: morecantile.Tile) -> InferenceResult:
        """fetch inference for base tiles"""
        inference_input = InferenceInput(image="")
        print(inference_input)
        res = httpx.get(self.url)

        print(res)
        pass

    def get_offset_tile_inference(self, bounds: List[float]) -> InferenceResult:
        """fetch inference for offset tiles"""
        pass
