from pydantic import BaseModel


class DetectionForAdvisory(BaseModel):
    disease: str
    confidence: float | None = None


class MultipleAdvisoryRequest(BaseModel):
    detections: list[DetectionForAdvisory]

