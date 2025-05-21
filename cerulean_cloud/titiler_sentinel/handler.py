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
All endpoints created by the `MultiBandTilerFactory` will require `scene_id={sentinel 1 scene id}` as query parameters.

  - Get Info: "{endpoint}/info?scene_id=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"
  - Get Available Bands: "{endpoint}/bands?scene_id=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"
  - Get Statistics (for band HH): "{endpoint}/statistics?scene_id=S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF&bands=hh"

Important:
The sentinel-1 data are stored in a `requester-pays` bucket, to be able to access the data you'll need to set `AWS_REQUEST_PAYER="requester"` in your environment.

see: https://cogeotiff.github.io/rio-tiler-pds/usage/overview/#requester-pays

"""

import logging
import os
from typing import Dict

from auth import api_key_auth
from fastapi import Depends, FastAPI, Query
from mangum import Mangum
from rio_tiler_pds.errors import InvalidSentinelSceneId
from rio_tiler_pds.sentinel.aws import S1L1CReader
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.core.factory import MultiBandTilerFactory
from fastapi.responses import Response
from rasterio.enums import Resampling
from rio_tiler.utils import render
import rasterio
import numpy as np

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


def DatasetPathParams(scene_id: str = Query(..., description="Scene Id")) -> str:
    """scene id"""
    return scene_id


S1Endpoints = MultiBandTilerFactory(  # type: ignore
    reader=S1L1CReader,
    path_dependency=DatasetPathParams,
)


@S1Endpoints.router.get("/viewer", response_class=HTMLResponse)
def viewer(
    request: Request,
    scene_id=Depends(S1Endpoints.path_dependency),
):
    """Viewer."""
    viewer_template = Jinja2Templates(
        directory=os.path.dirname(__file__) + "/templates/"
    )

    return viewer_template.TemplateResponse(
        name="viewer.html",
        context={
            "request": request,
            "tilejson_endpoint": S1Endpoints.url_for(request, "tilejson")
            + f"?scene_id={scene_id}",
            "info_endpoint": S1Endpoints.url_for(request, "info")
            + f"?scene_id={scene_id}",
            "stats_endpoint": S1Endpoints.url_for(request, "statistics")
            + f"?scene_id={scene_id}",
        },
        media_type="text/html",
    )


app.include_router(S1Endpoints.router, dependencies=[Depends(api_key_auth)])


@S1Endpoints.router.get("/part", response_class=Response)
async def get_part_tile(
    scene_id: str = Query(..., description="Sentinel-1 scene ID"),
    bbox: str = Query(..., description="minx,miny,maxx,maxy"),
    bands: str = Query("vv", description="Band to extract (vv or vh)"),
    width: int = Query(512, description="output width in pixels"),
    height: int = Query(512, description="output height in pixels"),
    rescale: str = Query("0,255", description="min,max for rescaling"),
    format: str = Query("png", description="output format: png or jpeg"),
    nodata: int = Query(0, description="source nodata value"),
):
    # Parse parameters
    minx, miny, maxx, maxy = map(float, bbox.split(","))
    rmin, rmax = map(float, rescale.split(","))

    # Read the requested window
    with rasterio.Env(AWS_REQUEST_PAYER="requester"):
        with S1L1CReader(scene_id) as reader:
            try:
                img = reader.part(
                    bbox=(minx, miny, maxx, maxy),
                    bands=[bands],
                    shape=(height, width),
                    resampling=Resampling.bilinear,
                )
            except Exception as e:
                return Response(str(e), status_code=400, media_type="text/plain")

    # Normalize to 8-bit
    arr = np.clip((img.data - rmin) / (rmax - rmin) * 255, 0, 255).astype("uint8")

    # Render and return
    content, mime_type = render(arr, format=format)
    return Response(content, media_type=mime_type)


@app.get("/health", description="Health Check", tags=["Health Check"])
def ping() -> Dict:
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)
logging.getLogger("rio-tiler").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
