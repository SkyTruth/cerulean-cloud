"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
import logging
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import torch
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from schema import InferenceInput, InferenceResult

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

logger = logging.getLogger(__name__)


def load_tracing_model(savepath):
    """load tracing model"""
    tracing_model = torch.jit.load(savepath)
    return tracing_model


def get_model():
    """load model"""
    return load_tracing_model("model/model.pt")


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits, dim=1)
    conf, classes = torch.max(probs, 1)
    return (conf, classes)


def b64_image_to_tensor(image: str) -> torch.Tensor:
    """convert input b64image to torch tensor"""
    # handle image
    img_bytes = b64decode(image)
    tmp = BytesIO()
    tmp.write(img_bytes)
    img = Image.open(tmp)
    np_img = np.array(img)
    return torch.tensor(np_img)


def array_to_b64_image(np_array: np.ndarray) -> str:
    """convert input b64image to torch tensor"""
    # handle image
    im = Image.fromarray(np.squeeze(np_array).astype("int8"))
    tmp = BytesIO()
    im.save(tmp, format="PNG")
    return b64encode(tmp.getvalue()).decode("ascii")


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


def _predict(payload: InferenceInput, model) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Loading tensor!")
    tensor = b64_image_to_tensor(payload.image)
    logging.info(f"Original tensor has shape {tensor.shape}")
    tensor = tensor[None, None, :, :]
    tensor = tensor.expand(1, 3, 512, 512).float()
    logging.info(f"Expanded tensor has shape {tensor.shape}")

    logging.info("Running inference...")
    out_batch_logits = model(tensor)
    logging.info("Finished inference, applying softmax")
    conf, classes = logits_to_classes(out_batch_logits)
    logging.info(f"Output classes array is {classes.shape}")
    return classes.numpy(), conf.detach().numpy()


@app.post(
    "/predict",
    description="Run inference",
    tags=["Run inference"],
    response_model=InferenceResult,
)
def predict(payload: InferenceInput, model=Depends(get_model)) -> Dict:
    """predict"""
    classes, conf = _predict(payload, model)
    enc_classes = array_to_b64_image(classes)
    enc_conf = array_to_b64_image(conf)
    return InferenceResult(
        classes=enc_classes, confidence=enc_conf, bounds=payload.bounds
    )
