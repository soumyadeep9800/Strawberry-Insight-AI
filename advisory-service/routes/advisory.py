from fastapi import APIRouter

from schemas.advisory import (
    AdvisoryRequest,
    AdvisoryResponse
)

from services.advisory_service import create_advisory


router = APIRouter(
    prefix="/api/advisory",
    tags=["Advisory"]
)


@router.post(
    "",
    response_model=AdvisoryResponse
)
def get_advisory(
    request: AdvisoryRequest
):

    advisory = create_advisory(
        disease=request.disease,
        confidence=request.confidence
    )

    return {
        "disease": request.disease,
        "confidence": request.confidence,
        "advisory": advisory
    }