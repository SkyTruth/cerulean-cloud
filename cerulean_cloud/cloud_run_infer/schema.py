"""schema for inference enpoint"""

from typing import Any, Dict, List

from pydantic import BaseModel


class InferenceInput(BaseModel):
    """
    Input values for model inference
    """

    image: str


class PredictPayload(BaseModel):
    """
    Stack of InferenceInputs and a dictionary of parms like thresholds
    """

    inf_stack: List[InferenceInput]
    model_dict: Dict[str, Any]
    scene_id: str


class InferenceResult(BaseModel):
    """
    Inference result from the model
    """

    json_data: str


class InferenceResultStack(BaseModel):
    """
    Stack of InferenceResult
    """

    stack: List[InferenceResult]
