"""Simple sentinel 1 based on MultiBandTilerFactory and rio-tiler-pds S1L1CReader.

Requirements:
  - titiler.core>=0.5,=<0.6
  - rio-tiler-pds>=0.6,=<0.7
  - starlette-cramjam>=0.1.0,<0.2
  - mangum>=0.10

Docs:
  - S1L1CReader: https://cogeotiff.github.io/rio-tiler-pds/usage/sentinel/#sentinel-1-aws
  - MultiBandTilerFactory: https://developmentseed.org/titiler/advanced/tiler_factories/#titilercorefactorymultibandtilerfactory

Input:
All endpoints created by the `MultiBandTilerFactory` will require `sceneid={sentinel 1 scene id}` as query parameters.

  - Get Info: "{endpoint}/info?sceneid=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"
  - Get Available Bands: "{endpoint}/bands?sceneid=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"
  - Get Statistics (for band HH): "{endpoint}/statistics?sceneid=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF&bands=hh"

Important:
The sentinel-1 data are stored in a `requester-pays` bucket, to be able to access the data you'll need to set `AWS_REQUEST_PAYER="requester"` in your environment.

see: https://cogeotiff.github.io/rio-tiler-pds/usage/overview/#requester-pays

"""

import logging
from typing import Dict

from fastapi import FastAPI, Query
from mangum import Mangum
from rio_tiler_pds.errors import InvalidSentinelSceneId
from rio_tiler_pds.sentinel.aws import S1L1CReader
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette_cramjam.middleware import CompressionMiddleware
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.core.factory import MultiBandTilerFactory

app = FastAPI(title="Sentinel-1 API")
add_exception_handlers(app, DEFAULT_STATUS_CODES)
add_exception_handlers(
    app,
    {
        InvalidSentinelSceneId: status.HTTP_404_NOT_FOUND,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(
    CompressionMiddleware,
    minimum_size=0,
    exclude_mediatype={
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/jp2",
        "image/webp",
    },
)


def DatasetPathParams(sceneid: str = Query(..., description="Scene Id")) -> str:
    """scene id"""
    return sceneid


S1Endpoints = MultiBandTilerFactory(  # type: ignore
    reader=S1L1CReader,
    path_dependency=DatasetPathParams,
)

app.include_router(S1Endpoints.router)


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)
logging.getLogger("rio-tiler").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
