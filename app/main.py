from __future__ import annotations

import os
from typing import Any

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class VehicleDetector:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.model = YOLO(self.model_path) if YOLO is not None else None

    def predict(self, image: Any) -> list[dict[str, Any]]:
        if self.model is None:
            return [{
                "class_name": "vehicle",
                "confidence": 0.0,
                "bbox": [0, 0, 0, 0],
                "status": "Model not available. Install ultralytics.",
            }]

        results = self.model(image, conf=0.25, verbose=False)
        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_name": result.names.get(int(box.cls[0]), "vehicle"),
                })
        return detections
