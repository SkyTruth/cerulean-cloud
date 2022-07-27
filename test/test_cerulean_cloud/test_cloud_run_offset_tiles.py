from base64 import b64encode

import numpy as np
import pytest
import rasterio
import torch
import torchvision  # noqa necessary for torch.jit.load of icevision mrcnn model
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_raster

import cerulean_cloud.cloud_run_offset_tiles.handler as handler
import cerulean_cloud.cloud_run_orchestrator.handler as orch_handler
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

        conf = high_conf_classes.detach().numpy().astype("int8")

        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                count=1,
                dtype=conf.dtype,
                width=conf.shape[0],
                height=conf.shape[1],
            ) as dataset:
                dataset.write(conf, 1)

            out_fc = orch_handler.get_fc_from_raster(memfile)
            assert len(out_fc.features) == 0


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


@pytest.mark.skip
def test_inference_mrcnn():
    bbox_conf_threshold = 0.5
    mask_conf_threshold = 0.05
    size = 512
    with open("test/test_cerulean_cloud/fixtures/tile_512_512_3band.png", "rb") as src:
        encoded = handler.b64encode(src.read()).decode("ascii")

    tensor = handler.b64_image_to_tensor(encoded)
    tensor = torch.stack([tensor, tensor, tensor])

    tensor = tensor.float() / 255

    model = handler.load_tracing_model(
        "cerulean_cloud/cloud_run_offset_tiles/model/model_mrcnn.pt"
    )
    print(torch.unbind(tensor))
    res_list = model(
        torch.unbind(tensor)
    )  # icevision mrcnn takes a list of 3D tensors not a 4D tensor like fastai unet
    print(res_list)  # Tuple[dict, list[dict]]
    res = []
    for tile in res_list[1]:  # iterating through the batch dimension.
        print(tile)
        pred_dict = handler.apply_conf_threshold_instances(
            tile, bbox_conf_threshold=bbox_conf_threshold
        )
        high_conf_classes = handler.apply_conf_threshold_masks(
            pred_dict, mask_conf_threshold=mask_conf_threshold, size=size
        )
        assert high_conf_classes.shape == torch.Size([512, 512])
        res.append(high_conf_classes)
    assert len(res) == 3
