"""client code to interact with titiler for sentinel 1"""

import logging
import os
import traceback
import urllib.parse as urlib
from typing import Dict, List, Optional, Tuple

import httpx
import mercantile
import numpy as np
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image

from cerulean_cloud.tiling import TMS

TMS_TITLE = TMS.id

logger = logging.getLogger("cerulean_cloud")


class TitilerClient:
    """client for titiler S1"""

    def __init__(self, url: str, timeout=None):
        """use deployment of titiler URL"""
        self.url = url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {os.getenv('TITILER_API_KEY')}"}
        )
        self.timeout = timeout
        # simple in-memory cache for scene bounds to avoid extra HTTP calls
        self._bounds_cache: Dict[str, List[float]] = {}

    async def get_bounds(self, scene_id: str) -> List[float]:
        """fetch bounds of a scene

        Args:
            scene_id (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF

        Returns:
            List[float]: A 4 item list containing bounding box
                            coordinates of the scene.
                            (minx, miny, maxx, maxy)
        Raises:
            HTTPException: For various HTTP related errors including authentication issues.
        """
        url = urlib.urljoin(self.url, "bounds")
        url += f"?scene_id={scene_id}"
        try:
            resp = await self.client.get(url, timeout=self.timeout)
            resp.raise_for_status()  # Raises error for 4XX or 5XX status codes
            bounds = resp.json()["bounds"]
            # cache for later bbox requests
            self._bounds_cache[scene_id] = bounds
            return bounds
        except Exception as e:
            logger.error(
                {
                    "message": "Failed to retrieve scene bounds",
                    "url": url,
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            raise

    async def get_statistics(self, scene_id: str, band: str = "vv") -> Dict:
        """fetch bounds of a scene

        Args:
            scene_id (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF
            band (str, optional): Which bands to include in the output. Defaults to "vv".

        Returns:
            Dict: Statistics for the bands scene
                includes keys such as min, max, mean, count, sum, std...
        """
        url = urlib.urljoin(self.url, "statistics")
        url += f"?scene_id={scene_id}"
        url += f"&bands={band}"
        try:
            resp = await self.client.get(url, timeout=self.timeout)
            resp.raise_for_status()  # Raises error for 4XX or 5XX status codes
            return resp.json()[band]
        except Exception as e:
            logger.error(
                {
                    "message": "Failed to retrieve scene statistics",
                    "url": url,
                    "exception": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            raise

    def get_base_tile_url(
        self,
        scene_id: str,
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
            scene_id (str): A valid S1 scene id
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
        url += f"?scene_id={scene_id}"
        url += f"&bands={band}"
        url += f"&scale={scale}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        if img_format:
            url += f"&format={img_format}"
        return url

    async def get_base_tile(
        self,
        scene_id: str,
        tile: mercantile.Tile,
        band: str = "vv",
        img_format: str = "png",
        scale: int = 1,
        rescale: Tuple[int, int] = (0, 255),
    ) -> np.ndarray:
        """get base tile as numpy array

        Args:
            scene_id (str): A valid S1 scene id
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
        url += f"?scene_id={scene_id}"
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
        scene_id: str,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        width: int = 512,
        height: int = 512,
        band: str = "vv",
        img_format: str = "png",
        rescale: Tuple[int, int] = (0, 255),
    ) -> np.ndarray:
        """get offset tile as numpy array (with bounds)

        Args:
            scene_id (str): A valid S1 scene id
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
        # Clip requested bbox to scene bounds to avoid boundless reads in WarpedVRT
        try:
            scene_bounds = self._bounds_cache.get(scene_id)
            if scene_bounds is None:
                scene_bounds = await self.get_bounds(scene_id)
        except Exception:
            scene_bounds = None

        if scene_bounds is not None:
            sb_minx, sb_miny, sb_maxx, sb_maxy = scene_bounds
            req_w = maxx - minx
            req_h = maxy - miny
            # Compute intersection
            ixmin = max(minx, sb_minx)
            iymin = max(miny, sb_miny)
            ixmax = min(maxx, sb_maxx)
            iymax = min(maxy, sb_maxy)

            # If no overlap, return an empty tile
            if ixmin >= ixmax or iymin >= iymax or req_w <= 0 or req_h <= 0:
                return np.zeros((height, width, 1), dtype=np.uint8)

            # Proportional size of intersection in output pixels
            out_w = int(round(width * ((ixmax - ixmin) / req_w)))
            out_h = int(round(height * ((iymax - iymin) / req_h)))
            out_w = max(1, min(width, out_w))
            out_h = max(1, min(height, out_h))

            # Offsets (origin at top-left of requested bbox)
            off_x = int(round(width * ((ixmin - minx) / req_w)))
            off_y = int(round(height * ((maxy - iymax) / req_h)))
            off_x = max(0, min(width - 1, off_x))
            off_y = max(0, min(height - 1, off_y))

            # Ensure paste region fits
            out_w = min(out_w, width - off_x)
            out_h = min(out_h, height - off_y)

            # Fetch just the intersection from Titiler
            sub_url = urlib.urljoin(
                self.url,
                f"bbox/{ixmin},{iymin},{ixmax},{iymax}/{out_w}x{out_h}.{img_format}",
            )
            sub_url += f"?scene_id={scene_id}"
            sub_url += f"&bands={band}"
            sub_url += f"&rescale={','.join([str(r) for r in rescale])}"
            sub_resp = await self.client.get(sub_url, timeout=self.timeout)
            sub_resp.raise_for_status()

            with MemoryFile(sub_resp.content) as memfile:
                with memfile.open() as dataset:
                    sub_img = reshape_as_image(dataset.read())

            # Prepare full-size canvas and paste
            canvas = np.zeros((height, width, sub_img.shape[2]), dtype=sub_img.dtype)
            canvas[
                off_y : off_y + sub_img.shape[0], off_x : off_x + sub_img.shape[1], :
            ] = sub_img
            return canvas

        # Fallback: no bounds available, try direct request (may 500 on boundless)
        url = urlib.urljoin(
            self.url, f"bbox/{minx},{miny},{maxx},{maxy}/{width}x{height}.{img_format}"
        )
        url += f"?scene_id={scene_id}"
        url += f"&bands={band}"
        url += f"&rescale={','.join([str(r) for r in rescale])}"
        resp = await self.client.get(url, timeout=self.timeout)
        resp.raise_for_status()
        with MemoryFile(resp.content) as memfile:
            with memfile.open() as dataset:
                return reshape_as_image(dataset.read())
