"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from io import BytesIO
from typing import Dict

import httpx
import icedata
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from icedata.utils import load_model_weights_from_url
from icevision.all import Dataset, models, tfms
from PIL import Image

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

class_map = icedata.pets.class_map()
model_type = models.torchvision.faster_rcnn
backbone = model_type.backbones.resnet50_fpn()
model = model_type.model(backbone=backbone, num_classes=len(class_map))
WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m3/pets_faster_resnetfpn50.zip"
load_model_weights_from_url(model, WEIGHTS_URL, map_location=torch.device("cpu"))
infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


def image_from_url(url):
    """fetch an image from an url"""
    res = httpx.get(url)
    img = Image.open(BytesIO(res.content))
    return np.array(img)


@app.get("/predict", description="Health Check", tags=["Health Check"])
def predict() -> Dict:
    """predict"""
    image_url = "https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg"
    img = image_from_url(image_url)
    infer_ds = Dataset.from_images([img], infer_tfms)
    preds = model_type.predict(model, infer_ds)

    return {"prediction": [p.as_dict() for p in preds]}
