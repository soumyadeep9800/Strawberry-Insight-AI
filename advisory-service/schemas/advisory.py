from pydantic import BaseModel


class AdvisoryRequest(BaseModel):
    disease: str
    confidence: float | None = None


class AdvisoryOutput(BaseModel):
    summary: str

    symptoms: list[str]

    prevention: list[str]

    management: list[str]

    hydroponic_considerations: list[str]


class AdvisoryResponse(BaseModel):
    disease: str
    confidence: float | None = None
    advisory: AdvisoryOutput