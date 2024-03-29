"""Clients for other cloud run functions"""

import json
import zipfile
from base64 import b64encode
from datetime import datetime
from io import BytesIO
from typing import List, Tuple

import httpx
import morecantile
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster
from rio_tiler.io import COGReader

from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceResult,
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
        # self.client = httpx.AsyncClient(
        #     headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"}
        # )
        self.scale = scale  # 1=256, 2=512, 3=...
        self.inference_parms = inference_parms
        self.filter_empty_tiles = filter_empty_tiles

    async def get_base_tile_inference(
        self,
        tile: morecantile.Tile,
        http_client: httpx.AsyncClient,
        rescale=(0, 255),
    ) -> InferenceResultStack:
        """fetch inference for base tiles"""
        img_array = await self.titiler_client.get_base_tile(
            sceneid=self.sceneid,
            tile=tile,
            scale=self.scale,
            rescale=rescale,
        )

        img_array = reshape_as_raster(img_array)

        bounds = list(TMS.bounds(tile))

        if self.filter_empty_tiles and not np.any(img_array):
            return InferenceResultStack(
                stack=[  # TODO Test if this can just be an empty list?
                    InferenceResult(
                        classes=None, confidence=None, bounds=bounds, features=[]
                    )
                ]
            )

        with self.aux_datasets.open() as src:
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            height, width = img_array.shape[1:]
            aux_ds = src.read(window=window, out_shape=(height, width))

        img_array = np.concatenate([img_array[0:1, :, :], aux_ds], axis=0)

        encoded = img_array_to_b64_image(img_array)

        inf_stack = [InferenceInput(image=encoded, bounds=TMS.bounds(tile))]
        payload = PredictPayload(inf_stack=inf_stack, inf_parms=self.inference_parms)
        res = await http_client.post(
            self.url + "/predict", json=payload.dict(), timeout=None
        )
        if res.status_code == 200:
            return InferenceResultStack(**res.json())
        else:
            print(f"XXX Issue was found in: {self.sceneid}")
            raise Exception(
                f"XXX Received unexpected status code: {res.status_code} {res.content}"
            )

    async def get_offset_tile_inference(
        self, bounds: List[float], http_client: httpx.AsyncClient, rescale=(0, 255)
    ) -> InferenceResultStack:
        """fetch inference for offset tiles"""
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
        if self.filter_empty_tiles and not np.any(img_array):
            return InferenceResultStack(
                stack=[  # TODO Test if this can just be an empty list?
                    InferenceResult(
                        classes=None, confidence=None, bounds=bounds, features=[]
                    )
                ]
            )

        with self.aux_datasets.open() as src:
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
            height, width = img_array.shape[1:]
            aux_ds = src.read(window=window, out_shape=(height, width))

        img_array = np.concatenate([img_array[0:1, :, :], aux_ds], axis=0)

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
                f"XXX Received unexpected status code: {res.status_code} {res.content}"
            )


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

    aux_dataset_channels = None  # XXX Assumes that the VV layer comes first. If it doesn't, then the aux_dataset_channels might be out of order
    for layer in layers:
        if layer.short_name == "VV":
            continue
        elif layer.short_name == "VESSEL":
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
            aux_dataset_channels = np.concatenate([aux_dataset_channels, ar], axis=2)

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
