# Strawberry AI — Gateway Service

FastAPI gateway and orchestration service for the Strawberry AI microservice architecture.

## Overview

The Gateway acts as the main backend entry point for the React frontend.

The frontend communicates with the Gateway instead of directly communicating with individual microservices.

## Current Architecture

```text
React Frontend
      ↓
Gateway Service :8000
      ↓
Disease Detection Service :8001
      ↓
YOLOv8s / RT-DETR-L
```

Future services will also be connected through the Gateway.

```text
Gateway
 ├── Disease Detection
 ├── Environment
 ├── Growth
 ├── Knowledge / RAG
 ├── Decision
 └── Authentication
```

## Project Structure

```text
gateway-service/
├── .venv/
├── main.py
├── orchestrator.py
├── requirements.txt
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
uvicorn main:app --reload --port 8000
```

Gateway:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Current API

### Analyze Image

```http
POST /analyze
```

Input:

* `file` — strawberry image
* `model` — `yolov8s` or `rtdetr`
* `confidence` — detection threshold

The Gateway forwards the request to the Disease Detection Service and returns its structured response.

Example:

```json
{
  "status": "success",
  "disease_detection": {
    "model": "YOLOv8s",
    "inference_time_ms": 2.73,
    "detections": [],
    "annotated_image": "data:image/jpeg;base64,..."
  }
}
```

## Responsibility

The Gateway is responsible for:

* Receiving frontend requests
* Routing requests to backend services
* Orchestrating microservices
* Returning structured responses to the frontend

The Gateway does not perform machine-learning inference itself.

## Service Communication

Current Disease Detection Service URL:

```text
http://127.0.0.1:8001
```

Gateway URL:

```text
http://127.0.0.1:8000
```

The React frontend should communicate only with the Gateway.
