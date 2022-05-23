"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Dict

import numpy as np
import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from schema import InferenceInput, InferenceResult

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def load_tracing_model(savepath):
    """load tracing model"""
    tracing_model = torch.jit.load(savepath)
    return tracing_model


def get_model():
    """load model"""
    return load_tracing_model("model/model.pt")


def test_tracing_model_one_batch(dls, tracing_model):
    """test tracing"""
    x, _ = dls.one_batch()
    out_batch_logits = tracing_model(x)
    return out_batch_logits


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits, dim=1)
    conf, classes = torch.max(probs, 1)
    return (conf,)


def b64_image_to_tensor(image: str) -> torch.Tensor:
    """convert input b64image to torch tensor"""
    # handle image
    img_bytes = b64decode(image)
    tmp = BytesIO()
    tmp.write(img_bytes)
    img = Image.open(tmp)
    np_img = np.array(img)
    return torch.tensor(np_img)


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResult,
)
def predict(payload: InferenceInput, model=Depends(get_model)) -> Dict:
    """predict"""
    tensor = b64_image_to_tensor(payload.image)
    out_batch_logits = model(tensor)
    conf = logits_to_classes(out_batch_logits)
    res = b64encode(conf).decode("ascii")
    return InferenceResult(res=res, bounds=payload.bounds)
