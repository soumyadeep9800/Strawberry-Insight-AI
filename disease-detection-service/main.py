from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from predictor import DiseasePredictor
from schemas import PredictionResponse


app = FastAPI(
    title="Strawberry Disease Detection Service",
    description="YOLOv8s and RT-DETR-L based strawberry disease detection API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = DiseasePredictor()


@app.get("/")
def root():
    return {
        "service": "Strawberry Disease Detection Service",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "available_models": [
            "YOLOv8s",
            "RT-DETR-L",
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    model: Literal["yolov8s", "rtdetr"] = Form("yolov8s"),
    confidence: float = Form(0.25),
):
    # Check image type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    # Check confidence range
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1.",
        )

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        result = predictor.predict(
            image=image,
            model_name=model,
            confidence=confidence,
        )

        return result

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )