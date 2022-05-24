from base64 import b64encode

import pytest
import torch
from PIL import Image

import cerulean_cloud.cloud_run_offset_tiles.handler as handler
from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceInput
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
    im = Image.fromarray(array[:, :, 0])
    im.save("test/test_cerulean_cloud/fixtures/tile_512_512.png")


def test_b64_image_to_tensor():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512.png", "rb") as src:
        encoded = b64encode(src.read()).decode("ascii")

    tensor = handler.b64_image_to_tensor(encoded)
    assert tensor.shape == torch.Size([512, 512])


@pytest.mark.skip
def test_inference():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512.png", "rb") as src:
        encoded = handler.b64encode(src.read()).decode("ascii")

    tensor = handler.b64_image_to_tensor(encoded)
    tensor = tensor[None, None, :, :]
    tensor = tensor.expand(1, 3, 512, 512).float()

    model = handler.load_tracing_model(
        "cerulean_cloud/cloud_run_offset_tiles/model/model.pt"
    )
    res = model(tensor)
    conf, classes = handler.logits_to_classes(res)
    assert conf.shape == torch.Size([1, 512, 512])
    assert classes.shape == torch.Size([1, 512, 512])


@pytest.mark.skip
def test_inference_():
    with open("test/test_cerulean_cloud/fixtures/tile_512_512.png", "rb") as src:
        encoded = handler.b64encode(src.read()).decode("ascii")

    model = handler.load_tracing_model(
        "cerulean_cloud/cloud_run_offset_tiles/model/model.pt"
    )
    payload = InferenceInput(image=encoded)

    classes, conf = handler._predict(payload, model)
    enc_classes = handler.array_to_b64_image(classes)
    enc_conf = handler.array_to_b64_image(conf)

    array_classes = handler.b64_image_to_tensor(enc_classes)
    assert array_classes.shape == torch.Size([512, 512])

    array_conf = handler.b64_image_to_tensor(enc_conf)
    assert array_conf.shape == torch.Size([512, 512])
