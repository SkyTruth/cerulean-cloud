"""Clients for other cloud run functions"""

import asyncio
import json
import os
import zipfile
from base64 import b64encode
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster
from rio_tiler.io import COGReader

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceResultStack,
    PredictPayload,
)
from cerulean_cloud.tiling import TMS


def img_array_to_b64_image(img_array: np.ndarray) -> str:
    """convert input b64image to torch tensor"""
    img_array = img_array.astype("int8")
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
    """Client for inference cloud run"""

    def __init__(
        self,
        url: str,
        titiler_client,
        sceneid: str,
        offset_bounds: List[float],
        offset_image_shape: Tuple[int, int],
        layers: List,
        scale: int,
        inference_parms,
        filter_empty_tiles=True,
    ):
        """init"""
        self.url = url
        self.titiler_client = titiler_client
        self.sceneid = sceneid
        self.aux_datasets = handle_aux_datasets(
            layers, self.sceneid, offset_bounds, offset_image_shape
        )
        self.scale = scale  # 1=256, 2=512, 3=...
        self.inference_parms = inference_parms
        self.filter_empty_tiles = filter_empty_tiles

    async def fetch_and_process_image(
        self, tile=None, bounds=None, rescale=(0, 255), num_channels=1
    ):
        """
        Asynchronously fetches and processes an image tile either by specifying the tile coordinates or geographic bounds.
        The image data is then rescaled, and optionally filtered to retain a specific number of channels.

        Parameters:
        - tile (Optional[Tuple[int, int, int]]): The XYZ tile coordinates to fetch the image for. If not provided, bounds must be given.
        - bounds (Optional[Tuple[float, float, float, float]]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) to fetch the image for. If not provided, tile must be given.
        - rescale (Tuple[int, int], optional): The range to rescale the image data to. Defaults to (0, 255).
        - num_channels (int, optional): The number of channels to retain in the processed image. Defaults to 1, assuming VV data is desired.

        Returns:
        - np.ndarray: The processed image array with dimensions corresponding to the specified number of channels and the spatial dimensions of the tile or bounds.

        Raises:
        - Exception: If neither tile nor bounds are provided.
        """
        if tile:
            img_array = await self.titiler_client.get_base_tile(
                sceneid=self.sceneid, tile=tile, scale=self.scale, rescale=rescale
            )
        else:
            hw = self.scale * 256
            img_array = await self.titiler_client.get_offset_tile(
                self.sceneid,
                *bounds,
                width=hw,
                height=hw,
                scale=self.scale,
                rescale=rescale,
            )

        img_array = reshape_as_raster(img_array)
        img_array = img_array[0:num_channels, :, :]
        return img_array

    async def process_auxiliary_datasets(self, img_array, bounds):
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
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            height, width = img_array.shape[1:]
            aux_ds = src.read(window=window, out_shape=(src.count, height, width))
        return np.concatenate([img_array, aux_ds], axis=0)

    async def send_inference_request_and_handle_response(
        self, http_client, img_array, bounds
    ):
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

        Note:
        - This function constructs the inference payload by encoding the image and specifying the geographic bounds and any additional inference parameters through `self.inference_parms`.
        """

        encoded = img_array_to_b64_image(img_array)
        inf_stack = [InferenceInput(image=encoded, bounds=bounds)]
        payload = PredictPayload(inf_stack=inf_stack, inf_parms=self.inference_parms)
        res = await http_client.post(
            self.url + "/predict", json=payload.dict(), timeout=None
        )
        if res.status_code == 200:
            return InferenceResultStack(**res.json())
        else:
            print(f"XXX Issue was found in: {self.sceneid}")
            raise Exception(
                f"Received unexpected status code: {res.status_code} {res.content}"
            )

    async def get_tile_inference(
        self, http_client, tile=None, bounds=None, rescale=(0, 255)
    ):
        """
        Orchestrates the complete process of fetching an image tile, processing it, enriching it with auxiliary datasets, and sending it for inference.

        Parameters:
        - http_client: The HTTP client for sending requests.
        - tile (Optional[Tuple[int, int, int]]): The XYZ tile coordinates to fetch the image for. If not provided, bounds must be given.
        - bounds (Optional[Tuple[float, float, float, float]]): The geographic bounds (min_lon, min_lat, max_lon, max_lat) for fetching the image. If not provided, tile must be given.
        - rescale (Tuple[int, int], optional): The range to rescale the image data to. Defaults to (0, 255).

        Returns:
        - InferenceResultStack: An object representing the stacked inference results, or an empty stack if the tile is empty and filtering is enabled.

        Raises:
        - Exception: If neither tile nor bounds are provided, or if an error occurs during any step of the process.

        Note:
        - This function integrates several steps: fetching the image, processing it, adding auxiliary data, and sending it for inference. It also includes a check to optionally skip empty tiles.
        """

        if bool(tile) == bool(bounds):  # XOR
            raise Exception(
                f"Inference requires (tile XOR bounds). Found {'neither' if not tile else 'both'}."
            )
        bounds = bounds or list(TMS.bounds(tile))
        img_array = await self.fetch_and_process_image(
            tile=tile, bounds=bounds, rescale=rescale
        )
        if self.filter_empty_tiles and not np.any(img_array):
            return InferenceResultStack(stack=[])
        if self.aux_datasets:
            img_array = await self.process_auxiliary_datasets(img_array, bounds)
        return await self.send_inference_request_and_handle_response(
            http_client, img_array, bounds
        )

    async def run_parallel_inference(self, inference_items):
        """
        Perform inference on a set of tiles or bounds asynchronously.

        Parameters:
        - inference_items (list): List of either tiles or bounds for inference.

        Returns:
        - list: List of inference results, with exceptions filtered out.
        """
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"}
        ) as async_http_client:
            tasks = [
                self.get_tile_inference(http_client=async_http_client, **item)
                for item in inference_items
            ]
            inferences = await asyncio.gather(*tasks, return_exceptions=False)
        return inferences


def get_scene_date_month(scene_id: str) -> str:
    """From a scene id, fetch the month of the scene"""
    # i.e. S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5
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
    """fetch ship density from gmtds service"""
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
    return np.squeeze(dens_array.astype("uint8"))


def get_dist_array(
    bounds: Tuple[float, float, float, float],
    img_shape: Tuple[int, int, int],
    raster_ds: str,
    max_distance: int = 60000,
):
    """fetch distance array from pre computed distance raster dataset"""
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
    data = np.squeeze(data)
    return data.astype(np.uint8)


def handle_aux_datasets(layers, scene_id, bounds, image_shape, **kwargs):
    """handle aux datasets"""
    if layers[0].short_name != "VV":
        raise NotImplementedError(
            f"VV Layer must come first. Instead found: {layers[0].short_name}"
        )

    if len(layers) > 1:
        aux_dataset_channels = None
        for layer in layers[1:]:
            if layer.short_name == "VESSEL":
                scene_date_month = get_scene_date_month(scene_id)
                ar = get_ship_density(bounds, image_shape, scene_date_month)
            elif layer.short_name == "INFRA":
                ar = get_dist_array(bounds, image_shape, layer.source_url)
            elif layer.short_name == "ALL_255":
                ar = 255 * np.ones(shape=image_shape)
            elif layer.short_name == "ALL_ZEROS":
                ar = np.zeros(shape=image_shape)
            else:
                raise NotImplementedError(f"Unrecognized layer: {layer.short_name}")

            ar = np.expand_dims(ar, 2)
            if aux_dataset_channels is None:
                aux_dataset_channels = ar
            else:
                aux_dataset_channels = np.concatenate(
                    [aux_dataset_channels, ar], axis=2
                )

        aux_memfile = MemoryFile()
        if aux_dataset_channels is not None:
            height, width = aux_dataset_channels.shape[0:2]
            transform = rasterio.transform.from_bounds(
                *bounds, height=image_shape[0], width=image_shape[1]
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

        return aux_memfile
    else:
        return None
