from pathlib import Path
import requests

IMAGE_DIR = Path(__file__).resolve().parent / "images"
API_URL = "http://127.0.0.1:8007/validate"
CONFIDENCE = 0.25

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

true_count = 0
false_count = 0

for image_path in images:

    with open(image_path, "rb") as f:

        response = requests.post(
            API_URL,
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"confidence": CONFIDENCE},
            timeout=120,
        )

    result = response.json()

    if result["strawberry_plant_detected"]:
        true_count += 1
    else:
        false_count += 1

print("\nRESULT")
print("======")
print("Total:", len(images))
print("True:", true_count)
print("False:", false_count)