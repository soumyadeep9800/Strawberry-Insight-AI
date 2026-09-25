# Strawberry AI — Disease Detection Service

FastAPI microservice for detecting strawberry diseases from images using trained object-detection models.

## Overview

This service provides an independent API for strawberry disease detection.

Supported models:

* **YOLOv8s Tuned** — primary model for fast inference and future real-time detection
* **RT-DETR-L** — alternative model with higher detection accuracy

## Architecture

```text
Image
  ↓
FastAPI
  ↓
Disease Detection Service
  ↓
YOLOv8s / RT-DETR-L
  ↓
Disease + Confidence + Bounding Box
  ↓
Annotated Image
```

## Project Structure

```text
disease-detection-service/
├── .venv/
├── main.py
├── predictor.py
├── schemas.py
├── requirements.txt
├── models/
│   ├── yolov8s_tuned_final.pt
│   └── rtdetr_best.pt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8001
```

Service:

```text
http://127.0.0.1:8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

## API

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Input:

* `file` — strawberry image
* `model` — `yolov8s` or `rtdetr`
* `confidence` — detection confidence threshold

Example response:

```json
{
  "model": "YOLOv8s",
  "inference_time_ms": 2.73,
  "detections": [
    {
      "disease": "Angular Leafspot",
      "confidence": 0.9603,
      "bbox": [145.1, 152.97, 272.78, 279.95]
    }
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

## Disease Classes

```text
Leaf Spot
Powdery Mildew Leaf
Gray Mold
Angular Leafspot
Blossom Blight
Powdery Mildew Fruit
Anthracnose Fruit Rot
```

## Responsibility

This service is responsible only for **computer-vision disease detection**.

It does not perform:

* Environmental analysis
* Agricultural knowledge retrieval
* LLM reasoning
* Final treatment/decision generation

Those responsibilities belong to other services in the Strawberry AI architecture.
