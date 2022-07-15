from base64 import b64encode

import numpy as np
import pytest
import rasterio
import torch
from rasterio.plot import reshape_as_raster

import cerulean_cloud.cloud_run_offset_tiles.handler as handler
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceInput,
    InferenceInputStack,
)
from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient


@pytest.mark.skip
def test_create_fixture_tile(
    url="https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/",
):
    titiler_client = TitilerClient(url=url)
    S1_ID = "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"

    tiles = list(TMS.tiles(*titiler_client.get_bounds(S1_ID), [10], truncate=False))
    tile = tiles[20]
    array = titiler_client.get_base_tile(S1_ID, tile=tile, scale=2, rescale=(0, 100))
    with rasterio.open(
        "test/test_cerulean_cloud/fixtures/tile_512_512_3band.png",
        "w",
        driver="PNG",
        height=array.shape[0],
        width=array.shape[1],
        count=3,
        dtype=array.dtype,
        compress="deflate",
    ) as dst:
        dst.write(reshape_as_raster(np.repeat(array[:, :, 0:1], 3, 2)))


def test_b64_image_to_tensor():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512_3band.png", "rb") as src:
        encoded = b64encode(src.read()).decode("ascii")

    tensor = handler.b64_image_to_tensor(encoded)
    assert tensor.shape == torch.Size([3, 512, 512])


@pytest.mark.skip
def test_inference():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512_3band.png", "rb") as src:
        encoded = handler.b64encode(src.read()).decode("ascii")

    tensor = handler.b64_image_to_tensor(encoded)
    tensor = tensor[None, :, :, :]
    tensor = tensor.float() / 255

    model = handler.load_tracing_model(
        "cerulean_cloud/cloud_run_offset_tiles/model/model.pt"
    )
    res = model(tensor)
    for tile in res:  # iterating through the batch dimension.
        conf, classes = handler.logits_to_classes(tile)
        high_conf_classes = handler.apply_conf_threshold(
            conf, classes, conf_threshold=0.9
        )
        assert conf.shape == torch.Size([512, 512])
        assert classes.shape == torch.Size([512, 512])
        assert high_conf_classes.shape == torch.Size([512, 512])


@pytest.mark.skip
def test_inference_():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512_3band.png", "rb") as src:
        encoded = handler.b64encode(src.read()).decode("ascii")

    model = handler.load_tracing_model(
        "cerulean_cloud/cloud_run_offset_tiles/model/model.pt"
    )
    payload = InferenceInputStack(stack=[InferenceInput(image=encoded)])

    inference_stack = handler._predict(payload, model)
    classes, conf, bounds = inference_stack[0]
    enc_classes = handler.array_to_b64_image(classes)
    enc_conf = handler.array_to_b64_image(conf)

    array_classes = handler.b64_image_to_tensor(enc_classes)
    assert array_classes.shape == torch.Size([1, 512, 512])

    array_conf = handler.b64_image_to_tensor(enc_conf)
    assert array_conf.shape == torch.Size([1, 512, 512])
