import httpx
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from schemas.advisory import (
    MultipleAdvisoryRequest
)

from orchestrator import (
    disease_prediction,
    generate_multiple_advisories,
)

app = FastAPI(
    title="Strawberry AI Gateway Service",
    description="Gateway for communicating with independent Strawberry AI services",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "Strawberry AI Gateway Service",
        "status": "running",
    }

DISEASE_SERVICE_URL = "http://127.0.0.1:8001"
ADVISORY_SERVICE_URL = "http://127.0.0.1:8002"

@app.get("/health")
async def health():
    services = {}

    async with httpx.AsyncClient(timeout=2.0) as client:

        # Disease Detection
        try:
            response = await client.get(f"{DISEASE_SERVICE_URL}/health")
            services["disease_detection"] = response.status_code == 200
        except Exception:
            services["disease_detection"] = False

        # Advisory
        try:
            response = await client.get(f"{ADVISORY_SERVICE_URL}/health")
            services["advisory"] = response.status_code == 200
        except Exception:
            services["advisory"] = False

    return {
        "status": "healthy",
        "services": services
    }
# -------------------------
# Disease Detection
# -------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    model: Literal["yolov8s", "rtdetr"] = Form("yolov8s"),
    confidence: float = Form(0.25),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    try:
        file_bytes = await file.read()

        result = await disease_prediction(
            file_bytes=file_bytes,
            filename=file.filename or "image.jpg",
            content_type=file.content_type,
            model=model,
            confidence=confidence,
        )

        return {
            "status": "success",
            "disease_detection": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Disease detection service error: {exc}",
        )


# -------------------------
# Advisory
# -------------------------

@app.post("/advisory/multiple")
async def multiple_advisory(
    request: MultipleAdvisoryRequest
):

    if not request.detections:
        raise HTTPException(
            status_code=400,
            detail="At least one disease detection is required."
        )

    try:

        detections = [
            {
                "disease": detection.disease,
                "confidence": detection.confidence
            }
            for detection in request.detections
        ]

        results = await generate_multiple_advisories(
            detections
        )

        return {
            "status": "success",
            "advisories": results
        }

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"AI advisory service error: {exc}"
        )
