"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""

import logging
from typing import Dict

from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(title="Cloud Run for offset tiles")


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
