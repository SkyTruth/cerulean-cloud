"""Cloud run main for tifeatures
"""
import logging

from mangum import Mangum
from tifeatures.main import app

logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
