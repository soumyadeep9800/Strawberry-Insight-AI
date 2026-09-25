import httpx
import asyncio

DISEASE_SERVICE_URL = "http://127.0.0.1:8001"
ADVISORY_SERVICE_URL = "http://127.0.0.1:8002"

# -------------------------
# Disease Service
# -------------------------

async def disease_prediction(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    model: str,
    confidence: float,
) -> dict:

    files = {
        "file": (
            filename,
            file_bytes,
            content_type,
        )
    }

    data = {
        "model": model,
        "confidence": str(confidence),
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{DISEASE_SERVICE_URL}/predict",
            files=files,
            data=data,
        )

    response.raise_for_status()

    return response.json()


# -------------------------
# Advisory Service
# -------------------------

async def generate_advisory(
    disease: str,
    confidence: float | None
) -> dict:

    advisory_data = {
        "disease": disease,
        "confidence": confidence,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:

        response = await client.post(
            f"{ADVISORY_SERVICE_URL}/api/advisory",
            json=advisory_data,
        )

    response.raise_for_status()

    return response.json()



async def generate_multiple_advisories(
    detections: list[dict]
) -> list[dict]:

    unique_diseases = {}

    for detection in detections:

        disease = detection.get("disease")
        confidence = detection.get("confidence")

        if not disease:
            continue

        if disease not in unique_diseases:
            unique_diseases[disease] = {
                "disease": disease,
                "confidence": confidence
            }

        elif (
            confidence is not None
            and (
                unique_diseases[disease]["confidence"] is None
                or confidence > unique_diseases[disease]["confidence"]
            )
        ):
            unique_diseases[disease] = {
                "disease": disease,
                "confidence": confidence
            }

    tasks = [
        generate_advisory(
            disease=detection["disease"],
            confidence=detection["confidence"]
        )
        for detection in unique_diseases.values()
    ]

    if not tasks:
        return []

    return await asyncio.gather(*tasks)
