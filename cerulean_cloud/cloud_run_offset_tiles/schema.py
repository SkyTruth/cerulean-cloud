"""schema for inference enpoint"""
from typing import List, Optional

import geojson
from pydantic import BaseModel


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    image: str
    bounds: Optional[List[float]]


class InferenceInputStack(BaseModel):
    """
    Stack of InferenceInput
    """

    stack: List[InferenceInput]


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """

    classes: Optional[str]
    confidence: Optional[str]
    bounds: Optional[List[float]]
    features: Optional[List[geojson.Feature]]


class InferenceResultStack(BaseModel):
    """
    Stack of InferenceResult
    """

    stack: List[InferenceResult]
