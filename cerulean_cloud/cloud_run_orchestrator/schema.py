"""schema for orchestration enpoint"""
from typing import List, Optional

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

    ntiles: int


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    image: str
    bounds: Optional[List[float]]


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """

    res: str
    bounds: Optional[List[float]]
