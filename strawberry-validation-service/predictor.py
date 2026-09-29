from pathlib import Path
import time

from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "strawberry_plant_yolov8s.pt"

# Model 1 is a single-class detector:
# 0 = strawberry_plant

CLASS_NAME = "strawberry_plant"


class StrawberryPlantPredictor:

    def __init__(self) -> None:
        self.model = None

    def _load_model(self):

        if self.model is None:

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}"
                )

            self.model = YOLO(str(MODEL_PATH))

        return self.model

    def validate(
        self,
        image: Image.Image,
        confidence: float = 0.25,
    ) -> dict:

        model = self._load_model()

        image = image.convert("RGB")

        # Start inference timer
        start_time = time.perf_counter()

        results = model.predict(
            source=image,
            imgsz=640,
            conf=confidence,
            device="cpu",
            verbose=False,
        )

        # End inference timer
        end_time = time.perf_counter()

        inference_time_ms = (
            end_time - start_time
        ) * 1000

        # ----------------------------------------------------
        # Find highest-confidence strawberry plant detection
        # ----------------------------------------------------

        max_confidence = 0.0

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):

                cls_id = int(
                    boxes.cls[i].item()
                )

                conf = float(
                    boxes.conf[i].item()
                )

                # Model 1 has only class 0
                if cls_id == 0:
                    max_confidence = max(
                        max_confidence,
                        conf,
                    )

        strawberry_detected = (
            max_confidence >= confidence
        )

        return {
            "strawberry_plant_detected": strawberry_detected,
            "confidence": round(
                max_confidence,
                4,
            ),
            "inference_time_ms": round(
                inference_time_ms,
                2,
            ),
        }