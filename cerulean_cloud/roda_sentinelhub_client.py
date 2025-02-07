"""client code to interact with roda http proxy to s3 s1 bucket"""

import posixpath
import urllib.parse as urlib
from typing import Dict

import httpx
from rio_tiler_pds.sentinel.utils import s1_sceneid_parser


class RodaSentinelHubClient:
    """client for http proxy"""

    def __init__(
        self, url="https://roda.sentinel-hub.com/sentinel-s1-l1c/", timeout=60
    ):
        """init"""
        self.url = url
        self.client = httpx.AsyncClient()
        self.timeout = timeout

    async def get_product_info(self, scene_id: str) -> Dict:
        """Get S1 product info

        Args:
            scene_id (str): A valid S1 scene id
                            i.e. S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF

        Returns:
            dict: a dictionary containing S1 product info
        """
        parsed_scene = s1_sceneid_parser(scene_id)
        url_path = posixpath.join(
            parsed_scene["product"],
            parsed_scene["acquisitionYear"],
            parsed_scene["_month"],
            parsed_scene["_day"],
            parsed_scene["beam"],
            parsed_scene["polarisation"],
            parsed_scene["scene"],
        )

        url = urlib.urljoin(self.url, url_path)
        url += "/productInfo.json"

        resp = await self.client.get(url, timeout=self.timeout)
        return resp.json()
