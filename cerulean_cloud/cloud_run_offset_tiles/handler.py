"""Cloud run handler for inference in offset tiles
Ref: https://github.com/python-engineer/ml-deployment/tree/main/google-cloud-run
"""
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cloud Run for offset tiles")
# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}
