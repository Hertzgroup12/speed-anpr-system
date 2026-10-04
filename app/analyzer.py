"""YOLO tracking, two-line speed measurement, and best-effort plate OCR."""

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import cv2
import easyocr
from ultralytics import YOLO

from app.config import Settings
from app.speed import crossed_line, estimate_speed_kmh, travel_direction

# COCO class IDs used by the standard YOLOv8 model.
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
_ANALYSIS_LOCK = threading.Lock()
_PLATE_TEXT = re.compile(r"^[A-Z0-9]{4,10}$")


class AnalysisInputError(ValueError):
    """An invalid upload or measurement setting that the API can report as 422."""


@dataclass
class TrackState:
    """Keep only the crossing and OCR state needed for one vehicle track."""

    previous_center_y: float
    crossing_times: dict[str, float] = field(default_factory=dict)
    plate_number: str | None = None
    plate_confidence: float = 0.0
    emitted: bool = False


@lru_cache
def _load_detector(model_path: str) -> YOLO:
    """Load detector weights on demand, not when the API server starts."""
    return YOLO(model_path)


@lru_cache
def _load_ocr_reader(languages: tuple[str, ...], gpu: bool) -> easyocr.Reader:
    """Load the OCR model on demand and reuse it for later videos."""
    return easyocr.Reader(list(languages), gpu=gpu)


def _read_plate(crop: Any, reader: easyocr.Reader) -> tuple[str | None, float]:
    """Choose the strongest plausible plate-like text returned for a crop."""
    best_text: str | None = None
    best_confidence = 0.0
    for _bounds, text, confidence in reader.readtext(crop):
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        if (
            _PLATE_TEXT.fullmatch(normalized)
            and any(character.isalpha() for character in normalized)
            and any(character.isdigit() for character in normalized)
            and confidence > best_confidence
        ):
            best_text = normalized
            best_confidence = float(confidence)
    return best_text, best_confidence


def analyze_video(
    video_path: str,
    distance_m: float,
    line_a_ratio: float,
    line_b_ratio: float,
    confidence: float,
    settings: Settings,
    ocr_interval: int = 5,
) -> dict[str, Any]:
    """Analyze one video and return metadata plus completed speed events."""
    if distance_m <= 0:
        raise AnalysisInputError("distance_m must be greater than zero")
    if not 0 < line_a_ratio < 1 or not 0 < line_b_ratio < 1:
        raise AnalysisInputError("line ratios must be between zero and one")
    if line_a_ratio == line_b_ratio:
        raise AnalysisInputError("line_a_ratio and line_b_ratio must be different")
    if not 0 <= confidence <= 1:
        raise AnalysisInputError("confidence must be between zero and one")
    if ocr_interval < 1:
        raise AnalysisInputError("ocr_interval must be at least one")

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise AnalysisInputError(
                "The uploaded file could not be opened as a video"
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise AnalysisInputError(
                "The video does not report a valid frame rate"
            )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    line_y = {
        "line_a": line_a_ratio * height,
        "line_b": line_b_ratio * height,
    }
    states: dict[int, TrackState] = {}
    events: list[dict[str, Any]] = []

    # Ultralytics creates fresh tracking state for each video when persist is false.
    # The lock prevents simultaneous requests from sharing its mutable predictor.
    with _ANALYSIS_LOCK:
        detector = _load_detector(settings.yolo_model_path)
        reader = _load_ocr_reader(tuple(settings.ocr_languages), settings.ocr_gpu)
        results = detector.track(
            source=video_path,
            stream=True,
            persist=False,
            conf=confidence,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )

        frame_count = 0
        for frame_count, result in enumerate(results, start=1):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            frame = result.orig_img
            frame_height, frame_width = frame.shape[:2]
            boxes_xyxy = boxes.xyxy.cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()

            for coordinates, track_id in zip(boxes_xyxy, track_ids):
                x1, y1, x2, y2 = (int(value) for value in coordinates)
                center_y = (y1 + y2) / 2
                state = states.get(track_id)
                if state is None:
                    states[track_id] = TrackState(previous_center_y=center_y)
                    state = states[track_id]

                # OCR is sampled to reduce expensive recognition work on long videos.
                if (
                    frame_count % ocr_interval == 0
                    and state.plate_confidence < 0.9
                ):
                    crop = frame[
                        max(0, y1):min(frame_height, y2),
                        max(0, x1):min(frame_width, x2),
                    ]
                    if crop.size:
                        plate, plate_confidence = _read_plate(crop, reader)
                        if plate and plate_confidence > state.plate_confidence:
                            state.plate_number = plate
                            state.plate_confidence = plate_confidence

                for line_name, crossing_y in line_y.items():
                    if (
                        line_name not in state.crossing_times
                        and crossed_line(
                            state.previous_center_y,
                            center_y,
                            crossing_y,
                        )
                    ):
                        state.crossing_times[line_name] = (frame_count - 1) / fps

                # A speed is valid only after this same track crosses both lines.
                if not state.emitted and len(state.crossing_times) == 2:
                    elapsed = abs(
                        state.crossing_times["line_b"]
                        - state.crossing_times["line_a"]
                    )
                    if elapsed > 0:
                        events.append(
                            {
                                "track_id": track_id,
                                "plate_number": state.plate_number,
                                "plate_confidence": state.plate_confidence or None,
                                "speed_kmh": round(
                                    estimate_speed_kmh(distance_m, elapsed),
                                    2,
                                ),
                                "direction": travel_direction(
                                    line_a_ratio,
                                    line_b_ratio,
                                    state.crossing_times["line_a"],
                                    state.crossing_times["line_b"],
                                ),
                                "frame": frame_count,
                                "timestamp_seconds": round(
                                    (frame_count - 1) / fps,
                                    3,
                                ),
                            }
                        )
                        state.emitted = True

                state.previous_center_y = center_y

    if frame_count == 0:
        raise AnalysisInputError("The video contains no readable frames")

    return {
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "frames_processed": frame_count,
        "duration_seconds": round(frame_count / fps, 3),
        "events": events,
    }
