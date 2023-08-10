"""schema for inference enpoint"""
from typing import Any, Dict, List, Optional

import geojson
from pydantic import BaseModel


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    image: str
    bounds: Optional[List[float]]


class PredictPayload(BaseModel):
    """
    Stack of InferenceInputs and a dictionary of parms like thresholds
    """

    inf_stack: List[InferenceInput]
    inf_parms: Dict[str, Any]


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
