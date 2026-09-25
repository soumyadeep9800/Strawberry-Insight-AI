from pydantic import BaseModel, Field
from typing import List


class Detection(BaseModel):
    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float]


class PredictionResponse(BaseModel):
    model: str
    inference_time_ms: float
    energy_joules: float
    carbon_footprint_gco2e: float
    detections: list[Detection]
    annotated_image: str | None = None