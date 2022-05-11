"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from typing import Dict

from fastapi import FastAPI

app = FastAPI(title="Cloud Run for offset tiles")


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}
