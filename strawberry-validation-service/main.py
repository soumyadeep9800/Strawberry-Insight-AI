from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, UnidentifiedImageError

from predictor import StrawberryPlantPredictor
from schemas import ValidationResponse


app = FastAPI(
    title="Strawberry Plant Validation Service",
    description=(
        "YOLOv8s based strawberry plant "
        "presence validation API"
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = StrawberryPlantPredictor()


@app.get("/")
def root():

    return {
        "service": "Strawberry Plant Validation Service",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "YOLOv8s",
        "class": "strawberry_plant",
    }


@app.post(
    "/validate",
    response_model=ValidationResponse,
)
async def validate(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
):

    # --------------------------------------------------------
    # Check image type
    # --------------------------------------------------------

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    # --------------------------------------------------------
    # Check confidence
    # --------------------------------------------------------

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

        result = predictor.validate(
            image=image,
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
            detail=f"Validation failed: {exc}",
        )