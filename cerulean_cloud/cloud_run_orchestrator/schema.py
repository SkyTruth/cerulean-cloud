"""schema for orchestration enpoint"""
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
