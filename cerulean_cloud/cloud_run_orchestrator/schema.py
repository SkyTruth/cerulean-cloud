"""schema for orchestration enpoint"""
from typing import Optional

import geojson
from pydantic import BaseModel


class OrchestratorInput(BaseModel):
    """
    Input values for orchestrator
    """

    sceneid: str
    trigger: Optional[int]
    vessel_density: Optional[str]
    infra_distance: Optional[str]
    zoom: int = 9
    scale: int = 2


class OrchestratorResult(BaseModel):
    """
    orchestrator result from the model
    """

    base_inference: str
    offset_inference: str
    classification: geojson.FeatureCollection
    ntiles: int
    noffsettiles: int
