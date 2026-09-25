from pathlib import Path 
from typing import Dict 
import base64 
from io import BytesIO 
import time 
from PIL import Image, ImageDraw, ImageFont 
from ultralytics import YOLO, RTDETR 
from energy import calculate_energy, calculate_carbon_footprint
 
BASE_DIR = Path(__file__).resolve().parent 
MODEL_DIR = BASE_DIR / "models" 
 
MODEL_PATHS = { 
    "yolov8s": MODEL_DIR / "yolov8s_tuned_final.pt", 
    "rtdetr": MODEL_DIR / "rtdetr_best.pt", 
} 
 
MODEL_DISPLAY_NAMES = { 
    "yolov8s": "YOLOv8s", 
    "rtdetr": "RT-DETR-L", 
} 
 
CLASS_NAMES = { 
    0: "Leaf Spot", 
    1: "Powdery Mildew Leaf", 
    2: "Gray Mold", 
    3: "Angular Leafspot", 
    4: "Blossom Blight", 
    5: "Powdery Mildew Fruit", 
    6: "Anthracnose Fruit Rot", 
} 
 
 
class DiseasePredictor: 
    def __init__(self) -> None: 
        self.models: Dict[str, object] = {} 
 
    def _load_model(self, model_name: str): 
        if model_name not in MODEL_PATHS: 
            raise ValueError(f"Unsupported model: {model_name}") 
 
        if model_name not in self.models: 
            model_path = MODEL_PATHS[model_name] 
 
            if not model_path.exists(): 
                raise FileNotFoundError( 
                    f"Model not found: {model_path}" 
                ) 
 
            if model_name == "yolov8s": 
                self.models[model_name] = YOLO(str(model_path)) 
            else: 
                self.models[model_name] = RTDETR(str(model_path)) 
 
        return self.models[model_name] 
 
    @staticmethod 
    def _encode_image(image: Image.Image) -> str: 
        buffer = BytesIO() 
        image.save(buffer, format="JPEG", quality=95) 
 
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8") 
 
        return f"data:image/jpeg;base64,{encoded}" 
 
    @staticmethod 
    def _annotate_image( 
        image: Image.Image, 
        detections: list[dict], 
    ) -> Image.Image: 
 
        annotated = image.copy().convert("RGB") 
 
        draw = ImageDraw.Draw(annotated) 
 
        try: 
            font = ImageFont.truetype("arial.ttf", 14) 
        except OSError: 
            font = ImageFont.load_default() 
 
        for detection in detections: 
            x1, y1, x2, y2 = detection["bbox"] 
 
            disease = detection["disease"] 
            confidence = detection["confidence"] 
 
            label = f"{disease} {confidence:.1%}" 
 
            # Red bounding box 
            draw.rectangle( 
                [x1, y1, x2, y2], 
                outline="red", 
                width=3, 
            ) 
 
            # Label background 
            text_bbox = draw.textbbox( 
                (x1, y1), 
                label, 
                font=font, 
            ) 
 
            text_x1 = x1 
            text_y1 = max(0, y1 - (text_bbox[3] - text_bbox[1]) - 6) 
 
            text_x2 = text_x1 + (text_bbox[2] - text_bbox[0]) + 8 
            text_y2 = y1 
 
            draw.rectangle( 
                [text_x1, text_y1, text_x2, text_y2], 
                fill="red", 
            ) 
 
            # White label text 
            draw.text( 
                (text_x1 + 4, text_y1 + 2), 
                label, 
                fill="white", 
                font=font, 
            ) 
 
        return annotated 
    def predict( 
        self, 
        image: Image.Image, 
        model_name: str = "yolov8s", 
        confidence: float = 0.25, 
    ) -> dict: 
 
        model = self._load_model(model_name) 
 
        image = image.convert("RGB") 
 
        # Start measuring model inference time 
        start_time = time.perf_counter() 
 
        results = model.predict( 
            source=image, 
            imgsz=320, 
            conf=confidence, 
            device="cpu", 
            verbose=False, 
        ) 
 
        # End measuring model inference time 
        end_time = time.perf_counter() 
 
        inference_time_ms = (end_time - start_time) * 1000 
        energy_joules = calculate_energy(inference_time_ms)
        carbon_footprint_gco2e = calculate_carbon_footprint(energy_joules)
        detections = [] 
 
        for result in results: 
 
            if result.boxes is None: 
                continue 
 
            boxes = result.boxes 
 
            for i in range(len(boxes)): 
 
                cls_id = int(boxes.cls[i].item()) 
                conf = float(boxes.conf[i].item()) 
 
                bbox = boxes.xyxy[i].cpu().tolist() 
 
                detections.append({ 
                    "disease": CLASS_NAMES.get( 
                        cls_id, 
                        f"Unknown Class {cls_id}", 
                    ), 
                    "confidence": round(conf, 4), 
                    "bbox": [ 
                        round(float(value), 2) 
                        for value in bbox 
                    ], 
                }) 
 
        annotated = self._annotate_image( 
            image, 
            detections, 
        ) 
 
        return { 
            "model": MODEL_DISPLAY_NAMES[model_name], 
            "inference_time_ms": round(inference_time_ms, 2), 
            "detections": detections, 
            "energy_joules": energy_joules,
            "carbon_footprint_gco2e": carbon_footprint_gco2e,
            "annotated_image": self._encode_image(annotated), 
        } 