from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    strawberry_plant_detected: bool

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    inference_time_ms: float