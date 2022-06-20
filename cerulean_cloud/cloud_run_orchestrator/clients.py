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
            self.url + "/predict", data=inference_input.json(), timeout=None
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
            self.url + "/predict", data=inference_input.json(), timeout=None
        )
        return InferenceResult(**res.json())


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
                            {"col": "category_column", "test": "Equal", "value": "All"},
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

    qs = (
        f"request={json.dumps(query)}"
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


def handle_aux_datasets(aux_datasets, scene_id, bounds, image_shape, **kwargs):
    """handle aux datasets"""
    assert (
        len(aux_datasets) == 2 or len(aux_datasets) == 3
    )  # so save as png file need RGB or RGBA

    aux_dataset_channels = None
    for aux_ds in aux_datasets:
        if aux_ds == "ship_density":
            scene_date_month = get_scene_date_month(scene_id)
            ar = get_ship_density(bounds, image_shape, scene_date_month)
        elif aux_ds.endswith(".tiff"):
            ar = get_dist_array(bounds, image_shape, aux_ds)

        ar = np.expand_dims(ar, 2)
        if aux_dataset_channels is None:
            aux_dataset_channels = ar
        else:
            aux_dataset_channels = np.concatenate([aux_dataset_channels, ar], axis=2)

    return aux_dataset_channels
