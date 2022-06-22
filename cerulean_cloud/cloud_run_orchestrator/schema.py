"""schema for orchestration enpoint"""
import geojson
from pydantic import BaseModel


class OrchestratorInput(BaseModel):
    """
    Input values for orchestrator
    """

    sceneid: str


class OrchestratorResult(BaseModel):
    """
    orchestrator result from the model
    """

    base_inference: str
    offset_inference: str
    classification: geojson.FeatureCollection
    ntiles: int
    noffsettiles: int
