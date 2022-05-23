"""schema for inference enpoint"""
from typing import List, Optional

from pydantic import BaseModel, Field


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


class InferenceResponse(BaseModel):
    """
    Output response for model inference
    """

    error: bool = Field(example=False, title="Whether there is error")
    results: InferenceResult


class ErrorResponse(BaseModel):
    """
    Error response for the API
    """

    error: bool = Field(example=True, title="Whether there is error")
    message: str = Field(example="", title="Error message")
    traceback: str = Field(None, example="", title="Detailed traceback of the error")
