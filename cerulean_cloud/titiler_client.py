"""client code to interact with titiler for sentinel 1"""
import urllib.parse as urlib
from io import BytesIO
from typing import List

import httpx
import mercantile
import numpy as np
from PIL import Image


class TitilerClient:
    """client for titiler S1"""

    def __init__(self, url: str):
        """use deployment of titiler URL"""
        self.url = url

    def get_bounds(self, sceneid: str) -> List[float]:
        """fetch bounds of a scene"""
        url = urlib.urljoin(self.url, "bounds")
        url += f"?sceneid={sceneid}"
        resp = httpx.get(url)
        return resp.json()["bounds"]

    def get_base_tile(
        self, sceneid: str, tile: mercantile.Tile, band="vv", img_format="png"
    ) -> np.ndarray:
        """get base tile as numpy array"""
        url = urlib.urljoin(self.url, f"tiles/{tile.z}/{tile.x}/{tile.y}")
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        url += f"&format={img_format}"
        resp = httpx.get(url)

        img = Image.open(BytesIO(resp.content))

        return np.array(img)

    def get_offset_tile(
        self,
        sceneid: str,
        minx,
        miny,
        maxx,
        maxy,
        width=256,
        height=256,
        band="vv",
        img_format="png",
    ) -> np.ndarray:
        """get offset tile as numpy array"""
        url = urlib.urljoin(
            self.url, f"crop/{minx},{miny},{maxx},{maxy}/{width}x{height}.{img_format}"
        )
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        print(url)
        resp = httpx.get(url)
        img = Image.open(BytesIO(resp.content))

        return np.array(img)
