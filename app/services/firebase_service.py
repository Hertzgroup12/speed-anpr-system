from __future__ import annotations

import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None


class YoloANPRService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.model = None
        if YOLO is not None:
            self.model = YOLO(self.model_path)

        self.reader = None
        if easyocr is not None:
            self.reader = easyocr.Reader(["en"], gpu=False)

    def detect_vehicles(self, image_array: np.ndarray) -> list[dict[str, Any]]:
        if self.model is None:
            return [{
                "class_name": "vehicle",
                "confidence": 0.0,
                "bbox": [0, 0, image_array.shape[1], image_array.shape[0]],
                "note": "YOLO model not loaded. Install ultralytics and pass a valid model.",
            }]

        results = self.model(image_array, conf=0.25, verbose=False)
        detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names.get(cls, "vehicle")
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def read_plate(self, cropped_image: np.ndarray) -> str:
        if self.reader is None:
            return "OCR unavailable: easyocr not installed."

        result = self.reader.readtext(cropped_image, detail=0, paragraph=False)
        text = " ".join(part for part in result if part)
        return text.strip() or "No plate detected"


def process_anpr_image(image_bytes: bytes, filename: str) -> dict[str, Any]:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}") from exc

    image_array = np.array(image)
    service = YoloANPRService()
    detections = service.detect_vehicles(image_array)

    if not detections:
        return {
            "filename": filename,
            "vehicles_detected": 0,
            "detections": [],
            "plate_text": "No vehicle detected",
        }

    best_detection = max(detections, key=lambda item: item["confidence"])
    x1, y1, x2, y2 = best_detection["bbox"]
    cropped = image_array[y1:y2, x1:x2]
    plate_text = service.read_plate(cropped)

    return {
        "filename": filename,
        "vehicles_detected": len(detections),
        "detections": detections,
        "best_detection": best_detection,
        "plate_text": plate_text,
    }
