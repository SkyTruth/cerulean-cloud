"""Checkpoint safety and compatibility for the shared inference models."""

import json
import pickle
from base64 import b64encode
from io import BytesIO
from zipfile import ZipFile

import pytest
import torch
import torchvision

from cerulean_cloud.models import BaseModel, FASTAIUNETModel, MASKRCNNModel


def test_unet_prediction_round_trip():
    model = FASTAIUNETModel(model_dict={"cls_map": {}})
    prediction = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

    restored = model.deserialize(model.serialize(prediction))

    torch.testing.assert_close(restored, prediction)


def test_maskrcnn_prediction_round_trip():
    model = MASKRCNNModel(model_dict={"cls_map": {}})
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
        "masks": [torch.ones(1, 2, 2)],
    }

    restored = model.deserialize(model.serialize(prediction.copy()))

    for key in ("boxes", "labels", "scores"):
        torch.testing.assert_close(restored[key], prediction[key])
    assert len(restored["masks"]) == len(prediction["masks"])
    torch.testing.assert_close(restored["masks"][0], prediction["masks"][0])


@pytest.mark.parametrize("model_type", [FASTAIUNETModel, MASKRCNNModel])
@pytest.mark.parametrize("opcode", [pickle.SETITEM, pickle.SETITEMS])
def test_prediction_rejects_checkpoint_with_non_dict_setitem(model_type, opcode):
    # Both opcodes must reject a list target (CVE-2026-24747). Use a valid
    # list index so an unpatched loader could complete without corrupting memory.
    items = pickle.BININT1 + b"\x00" + pickle.BININT1 + b"\x01"
    if opcode == pickle.SETITEMS:
        items = pickle.MARK + items
    malicious_pickle = pickle.dumps([0], protocol=2)[:-1] + items + opcode + pickle.STOP

    original = BytesIO()
    torch.save(torch.zeros(1), original)
    modified = BytesIO()
    with ZipFile(original) as source, ZipFile(modified, "w") as target:
        for entry in source.infolist():
            data = (
                malicious_pickle
                if entry.filename.endswith("/data.pkl")
                else source.read(entry)
            )
            target.writestr(entry, data)
    encoded = b64encode(modified.getvalue()).decode("ascii")
    payload = (
        encoded
        if model_type is FASTAIUNETModel
        else {"boxes": [], "labels": [], "scores": [], "masks_b64": [encoded]}
    )
    model = model_type(model_dict={"cls_map": {}})

    with pytest.raises(pickle.UnpicklingError, match="Can only SETITEM"):
        model.deserialize(json.dumps(payload))


class _NMSModel(torch.nn.Module):
    def forward(self, boxes, scores):
        return torchvision.ops.nms(boxes, scores, 0.5)


def test_load_torchscript_model_with_torchvision_operator(tmp_path):
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    scores = torch.tensor([0.9, 0.8])
    scripted = torch.jit.trace(_NMSModel(), (boxes, scores))
    model_path = tmp_path / "model.pt"
    scripted.save(str(model_path))
    model = BaseModel(model_dict={"cls_map": {}}, model_path_local=str(model_path))

    model.load()

    torch.testing.assert_close(model.model(boxes, scores), torch.tensor([0]))
