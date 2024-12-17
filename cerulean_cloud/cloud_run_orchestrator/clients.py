"""Clients for other cloud run functions"""

import asyncio
import gc  # Import garbage collection module
import json
import logging
import os
import zipfile
from base64 import b64encode
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import httpx
import numpy as np
import rasterio
import skimage
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster
from rio_tiler.io import COGReader

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceResultStack,
    PredictPayload,
)


def img_array_to_b64_image(img_array: np.ndarray, to_uint8=False) -> str:
    """Convert input image array to base64-encoded image."""
    if to_uint8 and not img_array.dtype == np.uint8:
        print(
            f"WARNING: changing from dtype {img_array.dtype} to uint8 without scaling!"
        )
        img_array = img_array.astype("uint8")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            count=img_array.shape[0],
            dtype=img_array.dtype,
            width=img_array.shape[1],
            height=img_array.shape[2],
        ) as dataset:
            dataset.write(img_array)
        img_bytes = memfile.read()

    return b64encode(img_bytes).decode("ascii")


class CloudRunInferenceClient:
    """Client for inference cloud run."""

    def __init__(
        self,
        url: str,
        titiler_client,
        sceneid: str,
        tileset_envelope_bounds: List[float],
        image_hw_pixels: Tuple[int, int],
        layers: List,
        scale: int,
        model_dict,
    ):
        """Initialize the inference client."""
        self.url = url
        self.titiler_client = titiler_client
        self.sceneid = sceneid
        self.scale = scale  # 1=256, 2=512, 3=...
        self.model_dict = model_dict

        # Configure logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger = logging.getLogger("InferenceClient")
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Handle auxiliary datasets and ensure they are properly managed
        self.aux_datasets = handle_aux_datasets(
            layers, self.sceneid, tileset_envelope_bounds, image_hw_pixels
        )

    async def fetch_and_process_image(
        self, tile_bounds, rescale=(0, 255), num_channels=1
    ):
        """
        Asynchronously fetches and processes an image tile either by specifying the geographic bounds.
        The image data is then rescaled, and optionally filtered to retain a specific number of channels.

        Parameters:
        - bounds (Optional[Tuple[float, float, float, float]]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) to fetch the image for.
        - rescale (Tuple[int, int], optional): The range to rescale the image data to. Defaults to (0, 255).
        - num_channels (int, optional): The number of channels to retain in the processed image. Defaults to 1, assuming VV data is desired.

        Returns:
        - np.ndarray: The processed image array with dimensions corresponding to the specified number of channels and the spatial dimensions of the bounds.
        """

        hw = self.scale * 256
        try:
            img_array = await self.titiler_client.get_offset_tile(
                self.sceneid,
                *tile_bounds,
                width=hw,
                height=hw,
                scale=self.scale,
                rescale=rescale,
            )

            img_array = reshape_as_raster(img_array)
            img_array = img_array[0:num_channels, :, :]
            return img_array
        except Exception as e:
            self.logger.warning(
                f"could not retrieve tile array for {self.sceneid}; {json.dumps(tile_bounds)}: {e}"
            )

    async def process_auxiliary_datasets(self, img_array, tile_bounds):
        """
        Processes auxiliary datasets for a given image array and its geographic bounds by overlaying additional data layers.
        This is useful for enriching the image data with extra contextual information such as additional spectral bands or derived indices.

        Parameters:
        - img_array (np.ndarray): The initial image array to augment, assumed to be of shape (channels, height, width).
        - bounds (Tuple[float, float, float, float]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) of the image.

        Returns:
        - np.ndarray: The augmented image array, which now includes the auxiliary datasets as additional channels.

        """
        with self.aux_datasets.open() as src:
            window = rasterio.windows.from_bounds(*tile_bounds, transform=src.transform)
            height, width = img_array.shape[1:]
            aux_ds = src.read(window=window, out_shape=(src.count, height, width))

        del src
        gc.collect()
        return np.concatenate([img_array, aux_ds], axis=0)

    async def send_inference_request_and_handle_response(self, http_client, img_array):
        """
        Sends an asynchronous request to an inference service with the processed image data and handles the response.
        This involves encoding the image array to a base64 format, constructing the inference payload, and interpreting the service response.

        Parameters:
        - http_client: The HTTP client for sending requests.
        - img_array (np.ndarray): The image array to send for inference, typically after processing and augmenting with auxiliary datasets.
        - bounds (Tuple[float, float, float, float]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) related to the image.

        Returns:
        - InferenceResultStack: An object representing the stacked inference results.

        Raises:
        - Exception: If the request fails or the service returns an unexpected status code, with details provided in the exception message.
        """

        encoded = img_array_to_b64_image(img_array, to_uint8=True)
        inf_stack = [InferenceInput(image=encoded)]
        payload = PredictPayload(inf_stack=inf_stack, model_dict=self.model_dict)

        max_retries = 2  # Total attempts including the first try
        retry_delay = 5  # Delay in seconds between retries

        for attempt in range(max_retries):
            try:
                res = await http_client.post(
                    self.url + "/predict", json=payload.dict(), timeout=None
                )
                if res.status_code == 200:
                    return InferenceResultStack(**res.json())
                else:
                    self.logger.warning(
                        f"Attempt {attempt + 1}: Failed with status code {res.status_code}. Retrying..."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)  # Wait before retrying
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}: Exception occurred: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)  # Wait before retrying

        # If all attempts fail, raise an exception
        raise Exception(
            f"All attempts failed after {max_retries} retries. Last known error: {res.content}"
        )

    async def get_tile_inference(self, http_client, tile_bounds, rescale=(0, 255)):
        """
        Orchestrates the complete process of fetching an image tile, processing it, enriching it with auxiliary datasets, and sending it for inference.

        Parameters:
        - http_client: The HTTP client for sending requests.
        - bounds (Optional[Tuple[float, float, float, float]]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) for fetching the image.
        - rescale (Tuple[int, int], optional): The range to rescale the image data to. Defaults to (0, 255).

        Returns:
        - InferenceResultStack: An object representing the stacked inference results, or an empty stack if the tile is empty.

        Note:
        - This function integrates several steps: fetching the image, processing it, adding auxiliary data, and sending it for inference. It also includes a check to optionally skip empty tiles.
        """

        img_array = await self.fetch_and_process_image(tile_bounds, rescale)
        if img_array is None or not np.any(img_array):
            self.logger.warning(f"no imagery for {str(tile_bounds)}")
            return InferenceResultStack(stack=[])

        if self.aux_datasets:
            img_array = await self.process_auxiliary_datasets(img_array, tile_bounds)
        res = await self.send_inference_request_and_handle_response(
            http_client, img_array
        )
        del img_array
        gc.collect()
        return res

    async def run_parallel_inference(self, tileset):
        """
        Perform inference on a set of tile bounds asynchronously.

        Parameters:
        - tileset (list): List of bounds for inference.

        Returns:
        - list: List of inference results, with exceptions filtered out.
        """
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"}
        ) as async_http_client:
            tasks = [
                self.get_tile_inference(
                    http_client=async_http_client, tile_bounds=tile_bounds
                )
                for tile_bounds in tileset
            ]
            inferences = await asyncio.gather(*tasks, return_exceptions=False)
            # False means this process will error out if any subtask errors out
            # True means this process will return a list including errors if any subtask errors out

        # After processing, close and delete aux_datasets
        if self.aux_datasets:
            del self.aux_datasets
            gc.collect()

        return inferences


def get_scene_date_month(scene_id: str) -> str:
    """From a scene id, fetch the month of the scene."""
    date_time_str = scene_id[17:32]
    date_time_obj = datetime.strptime(date_time_str, "%Y%m%dT%H%M%S")
    date_time_obj = date_time_obj.replace(day=1, hour=0, minute=0, second=0)
    return date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_ship_density(
    bounds: Tuple[float, float, float, float],
    img_shape: Tuple[int, int],
    scene_date_month: str = "2020-08-01T00:00:00Z",
    max_dens=100,
    url="http://gmtds.maplarge.com/Api/ProcessDirect?",
) -> np.ndarray:
    """Fetch ship density from gmtds service."""
    h, w = img_shape
    bbox_wms = bounds[0], bounds[2], bounds[1], bounds[-1]

    def get_query(bbox_wms, scene_date_month):
        query = {
            "action": "table/query",
            "query": {
                "engineVersion": 2,
                "sqlselect": [
                    "category_column",
                    "category",
                    f"GridCrop(grid_float_4326, {', '.join([str(b) for b in bbox_wms])}) as grid_float",
                ],
                "table": {
                    "query": {
                        "table": {"name": "ais/density"},
                        "where": [
                            [
                                {
                                    "col": "category_column",
                                    "test": "Equal",
                                    "value": "All",
                                },
                                {"col": "category", "test": "Equal", "value": "All"},
                            ]
                        ],
                        "withgeo": True,
                    }
                },
                "where": [
                    [{"col": "time", "test": "Equal", "value": f"{scene_date_month}"}]
                ],
            },
        }
        return query

    qs = (
        f"request={json.dumps(get_query(bbox_wms, scene_date_month))}"
        "&uParams=action:table/query;formatType:tiff;withgeo:false;withGeoJson:false;includePolicies:true"
    )

    r = httpx.get(f"{url}{qs}", timeout=None, follow_redirects=True)
    try:
        r.raise_for_status()
        tempbuf = BytesIO(r.content)
        zipfile_ob = zipfile.ZipFile(tempbuf)
        cont = list(zipfile_ob.namelist())
        with rasterio.open(BytesIO(zipfile_ob.read(cont[0]))) as dataset:
            ar = dataset.read(
                out_shape=img_shape[0:2],
                out_dtype="uint8",
                resampling=Resampling.nearest,
            )
    except (ValueError, rasterio.errors.RasterioIOError) as e:
        print(f"Failed to fetch ship density with {e}, trying 2019...")
        scene_date_month_obj = datetime.strptime(scene_date_month, "%Y-%m-%dT%H:%M:%SZ")
        scene_date_month_obj = scene_date_month_obj.replace(year=2019)
        new_scene_date_month = scene_date_month_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Trying {new_scene_date_month}...")
        qs = (
            f"request={json.dumps(get_query(bbox_wms, new_scene_date_month))}"
            "&uParams=action:table/query;formatType:tiff;withgeo:false;withGeoJson:false;includePolicies:true"
        )

        r = httpx.get(f"{url}{qs}", timeout=None, follow_redirects=True)
        try:
            r.raise_for_status()
            tempbuf = BytesIO(r.content)
            zipfile_ob = zipfile.ZipFile(tempbuf)
            cont = list(zipfile_ob.namelist())
            with rasterio.open(BytesIO(zipfile_ob.read(cont[0]))) as dataset:
                ar = dataset.read(
                    out_shape=img_shape[0:2],
                    out_dtype="uint8",
                    resampling=Resampling.nearest,
                )
        except httpx.HTTPError:
            print("Failed to fetch ship density!")
            return None

    except httpx.HTTPError:
        print("Failed to fetch ship density!")
        return None

    dens_array = ar / (max_dens / 255)
    dens_array[dens_array >= 255] = 255

    del tempbuf, zipfile_ob, cont, r, ar
    gc.collect()

    return np.squeeze(dens_array.astype("uint8"))


def get_dist_array(
    bounds: Tuple[float, float, float, float],
    img_shape: Tuple[int, int, int],
    raster_ds: str,
    max_distance: int = 60000,
):
    """Fetch distance array from pre-computed distance raster dataset."""
    with COGReader(raster_ds) as image:
        height, width = img_shape[0:2]
        img = image.part(
            bbox=bounds,
            height=height,
            width=width,
        )
        data = img.data_as_image()

    if (data == 0).all():
        data = np.ones(img_shape) * 255
    else:
        data = data / (max_distance / 255)  # 60 km
        data[data >= 255] = 255

    upsampled = skimage.transform.resize(
        data, (*img_shape[0:2], 1), order=1, preserve_range=True
    )  # resampling interpolation must match training data prep
    upsampled = np.squeeze(upsampled)

    del data, img
    gc.collect()

    return upsampled.astype(np.uint8)


def handle_aux_datasets(
    layers, scene_id, tileset_envelope_bounds, image_hw_pixels, **kwargs
):
    """Handle auxiliary datasets."""
    if layers[0].short_name != "VV":
        raise NotImplementedError(
            f"VV Layer must come first. Instead found: {layers[0].short_name}"
        )

    if len(layers) > 1:
        aux_dataset_channels = None
        for layer in layers[1:]:
            if layer.short_name == "VESSEL":
                scene_date_month = get_scene_date_month(scene_id)
                ar = get_ship_density(
                    tileset_envelope_bounds, image_hw_pixels, scene_date_month
                )
            elif layer.short_name == "INFRA":
                ar = get_dist_array(
                    tileset_envelope_bounds, image_hw_pixels, layer.source_url
                )
            elif layer.short_name == "ALL_255":
                ar = 255 * np.ones(shape=image_hw_pixels, dtype="uint8")
            elif layer.short_name == "ALL_ZEROS":
                ar = np.zeros(shape=image_hw_pixels, dtype="uint8")
            else:
                raise NotImplementedError(f"Unrecognized layer: {layer.short_name}")

            ar = np.expand_dims(ar, 2)
            if aux_dataset_channels is None:
                aux_dataset_channels = ar
            else:
                aux_dataset_channels = np.concatenate(
                    [aux_dataset_channels, ar], axis=2
                )

            del ar
            gc.collect()

        aux_memfile = MemoryFile()
        if aux_dataset_channels is not None:
            height, width = aux_dataset_channels.shape[0:2]
            transform = rasterio.transform.from_bounds(
                *tileset_envelope_bounds,
                height=image_hw_pixels[0],
                width=image_hw_pixels[1],
            )
            with aux_memfile.open(
                driver="GTiff",
                count=len(layers) - 1,  # XXX Hack, assumes layers follow a VV layer
                height=height,
                width=width,
                dtype=aux_dataset_channels.dtype,
                transform=transform,
                crs="EPSG:4326",
            ) as dataset:
                dataset.write(reshape_as_raster(aux_dataset_channels))

        del aux_dataset_channels
        gc.collect()

        return aux_memfile
    else:
        return None
