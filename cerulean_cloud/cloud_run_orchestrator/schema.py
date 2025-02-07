"""schema for orchestration enpoint"""

from typing import Optional

# import geojson
from pydantic import BaseModel


class OrchestratorInput(BaseModel):
    """
    Input values for orchestrator
    """

    scene_id: str
    trigger: Optional[int]
    zoom: Optional[int] = None
    scale: Optional[int] = None
    dry_run: bool = False


class OrchestratorResult(BaseModel):
    """
    orchestrator result from the model
    """

    # classification_base: geojson.FeatureCollection
    # classification_offset: geojson.FeatureCollection
    # classification_merged: geojson.FeatureCollection
    # ntiles: int
    # noffsettiles: int
    status: str
