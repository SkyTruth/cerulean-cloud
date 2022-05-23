"""client code to interact with titiler for sentinel 1"""
import urllib.parse as urlib
from io import BytesIO
from typing import Dict, List, Tuple

import httpx
import mercantile
import numpy as np
from PIL import Image

TMS = "WorldCRS84Quad"


class TitilerClient:
    """client for titiler S1"""

    def __init__(self, url: str):
        """use deployment of titiler URL"""
        self.url = url

    def get_bounds(self, sceneid: str) -> List[float]:
        """fetch bounds of a scene

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF

        Returns:
            List[float]: A 4 item list containing bounding box
                            coordinates of the scene.
                            (minx, miny, maxx, maxy)
        """
        url = urlib.urljoin(self.url, "bounds")
        url += f"?sceneid={sceneid}"
        resp = httpx.get(url)
        return resp.json()["bounds"]

    def get_statistics(self, sceneid: str, band: str = "vv") -> Dict:
        """fetch bounds of a scene

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF
            band (str, optional): Which bands to include in the output. Defaults to "vv".

        Returns:
            Dict: Statistics for the bands scene
                includes keys such as min, max, mean, count, sum, std...
        """
        url = urlib.urljoin(self.url, "statistics")
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        resp = httpx.get(url)
        return resp.json()[band]

    def get_base_tile(
        self,
        sceneid: str,
        tile: mercantile.Tile,
        band: str = "vv",
        img_format: str = "png",
        scale: int = 1,
        rescale: Tuple[int, int] = (0, 1000),
    ) -> np.ndarray:
        """get base tile as numpy array

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF
            tile (mercantile.Tile): The Tile instance to fetch
            band (str, optional): Which bands to include in the output. Defaults to "vv".
            img_format (str, optional): File format of the output from the tiler. Defaults to "png".
            scale (int, optional): Proxy for tile size.  1=256x256, 2=512x512... Defaults to 1.
            rescale (Tuple[int, int], optional): Min max value to rescale to uint8.

        Returns:
            np.ndarray: The requested tile of the scene as a numpy array.
        """
        url = urlib.urljoin(self.url, f"tiles/{TMS}/{tile.z}/{tile.x}/{tile.y}")
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        url += f"&format={img_format}"
        url += f"&scale={scale}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        resp = httpx.get(url)

        img = Image.open(BytesIO(resp.content))

        return np.array(img)

    def get_offset_tile(
        self,
        sceneid: str,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        width: int = 256,
        height: int = 256,
        band: str = "vv",
        img_format: str = "png",
        rescale: Tuple[int, int] = (0, 1000),
    ) -> np.ndarray:
        """get offset tile as numpy array (with bounds)

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF
            minx (float): Bounding box of the image.
            miny (float): Bounding box of the image.
            maxx (float): Bounding box of the image.
            maxy (float): Bounding box of the image.
            width (int, optional): Output image width. Defaults to 256.
            height (int, optional): Output image height. Defaults to 256.
            band (str, optional): Which bands to include in the output. Defaults to "vv".
            img_format (str, optional): File format of the output. Defaults to "png".
            rescale (Tuple[int, int], optional): Min max value to rescale to uint8.

        Returns:
            np.ndarray: The requested image of the bounds of the scene as a numpy array.
        """
        url = urlib.urljoin(
            self.url, f"crop/{minx},{miny},{maxx},{maxy}/{width}x{height}.{img_format}"
        )
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        print(url)
        resp = httpx.get(url)
        img = Image.open(BytesIO(resp.content))

        return np.array(img)
