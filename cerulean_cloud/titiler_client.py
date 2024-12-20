"""client code to interact with titiler for sentinel 1"""

import logging
import os
import sys
import time
import traceback
import urllib.parse as urlib
from typing import Dict, List, Optional, Tuple

import httpx
import mercantile
import numpy as np
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image

from cerulean_cloud.cloud_run_orchestrator.utils import structured_log
from cerulean_cloud.tiling import TMS

TMS_TITLE = TMS.identifier


class TitilerClient:
    """client for titiler S1"""

    def __init__(self, url: str, timeout=None):
        """use deployment of titiler URL"""
        self.url = url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.getenv('TITILER_API_KEY')}"}
        )
        self.timeout = timeout

        # Configure logger
        self.logger = logging.getLogger("DatabaseClient")
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def get_bounds(self, sceneid: str, retries: int = 3) -> List[float]:
        """fetch bounds of a scene

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF

        Returns:
            List[float]: A 4 item list containing bounding box
                            coordinates of the scene.
                            (minx, miny, maxx, maxy)
        Raises:
            HTTPException: For various HTTP related errors including authentication issues.
        """

        url = urlib.urljoin(self.url, "bounds")
        url += f"?sceneid={sceneid}"
        for attempt in range(1, retries + 1):
            try:
                resp = await self.client.get(url, timeout=self.timeout)
                resp.raise_for_status()  # Raises error for 4XX or 5XX status codes
                return resp.json()["bounds"]
            except Exception:
                if attempt == retries:
                    raise
                self.logger.warning(
                    structured_log(
                        f"Error retrieving {url}", severity="WARNING", scene_id=sceneid
                    )
                )
                time.sleep(5**attempt)

        raise RuntimeError("Failed to retrieve scene bounds")

    async def get_statistics(
        self, sceneid: str, band: str = "vv", retries: int = 3
    ) -> Dict:
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
        for attempt in range(1, retries + 1):
            try:
                resp = await self.client.get(url, timeout=self.timeout)
                resp.raise_for_status()  # Raises error for 4XX or 5XX status codes
                return resp.json()[band]
            except Exception:
                if attempt == retries:
                    raise
                self.logger.warning(
                    structured_log(
                        f"Error retrieving {url}", severity="WARNING", scene_id=sceneid
                    )
                )
                time.sleep(5**attempt)

        raise RuntimeError("Failed to retrieve scene stats")

    def get_base_tile_url(
        self,
        sceneid: str,
        band: str = "vv",
        img_format: Optional[str] = None,
        scale: int = 1,
        rescale: Tuple[int, int] = (0, 255),
        z="{z}",
        x="{x}",
        y="{y}",
    ) -> str:
        """Forge titiler URL with scene id and stats.

        Args:
            sceneid (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF
            band (str, optional): Which bands to include in the output.. Defaults to "vv".
            img_format (Optional[str], optional): ile format of the output from the tiler. Defaults to None.
            scale (int, optional): Proxy for tile size. Defaults to 1.
            rescale (Tuple[int, int], optional): Min max value to rescale to uint8. Defaults to (0, 1000).
            z (str, optional): Z. Defaults to "{z}".
            x (str, optional): X. Defaults to "{x}".
            y (str, optional): Y. Defaults to "{y}".

        Returns:
            str: URL to get XYZ server for a specific scene.
        """
        url = urlib.urljoin(self.url, f"tiles/{z}/{x}/{y}")
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        url += f"&scale={scale}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        if img_format:
            url += f"&format={img_format}"
        return url

    async def get_base_tile(
        self,
        sceneid: str,
        tile: mercantile.Tile,
        band: str = "vv",
        img_format: str = "png",
        scale: int = 1,
        rescale: Tuple[int, int] = (0, 255),
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
        url = urlib.urljoin(self.url, f"tiles/{TMS_TITLE}/{tile.z}/{tile.x}/{tile.y}")
        url += f"?sceneid={sceneid}"
        url += f"&bands={band}"
        url += f"&format={img_format}"
        url += f"&scale={scale}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        resp = await self.client.get(url, timeout=self.timeout)

        with MemoryFile(resp.content) as memfile:
            with memfile.open() as dataset:
                np_img = reshape_as_image(dataset.read())

        return np_img

    async def get_offset_tile(
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
        scale: int = 1,
        rescale: Tuple[int, int] = (0, 255),
        retries: int = 3,
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
        url += f"&scale={scale}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        for attempt in range(1, retries + 1):
            try:
                resp = await self.client.get(url, timeout=self.timeout)
                with MemoryFile(resp.content) as memfile:
                    with memfile.open() as dataset:
                        np_img = reshape_as_image(dataset.read())

                return np_img
            except Exception as e:
                if attempt == retries:
                    self.logger.error(
                        structured_log(
                            f"Failed to retrieve {url}",
                            severity="ERROR",
                            scene_id=sceneid,
                            exception=str(e),
                            traceback=traceback.format_exc(),
                        )
                    )
                    raise
                self.logger.warning(
                    structured_log(
                        f"Failed to retrieve {url}, retrying . . .",
                        severity="WARNING",
                        scene_id=sceneid,
                    )
                )
                time.sleep(5**attempt)
