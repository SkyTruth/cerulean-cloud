"""schema for inference enpoint"""

from typing import Any, Dict, List, Optional

import geojson
from pydantic import BaseModel, Field


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

    classes: Optional[str] = Field(default=None)
    confidence: Optional[str] = Field(default=None)
    bounds: Optional[List[float]] = None
    features: Optional[List[geojson.Feature]] = None

    class Config:
        """
        This tells Pydantic to allow arbitrary types, like geojson.Feature,
        within this model without trying to validate them based on Pydantic's
        internal schema constraints.
        """

        arbitrary_types_allowed = True  # Allow geojson.Feature


class InferenceResultStack(BaseModel):
    """
    Stack of InferenceResult
    """

    stack: List[InferenceResult]
