"""YOLO tracking, two-line speed measurement, and best-effort plate OCR."""

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
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
    plate_crop: Any | None = None
    emitted: bool = False


def acquire_analysis_lock() -> None:
    """Keep uploads and live camera sessions from sharing mutable YOLO tracking state."""
    _ANALYSIS_LOCK.acquire()


def release_analysis_lock() -> None:
    """Release the model lock after a live session ends."""
    _ANALYSIS_LOCK.release()


@lru_cache
def _load_detector(model_path: str) -> YOLO:
    """Load detector weights on demand, not when the API server starts."""
    return YOLO(model_path)


@lru_cache
def _load_ocr_reader(languages: tuple[str, ...], gpu: bool) -> easyocr.Reader:
    """Load the OCR model on demand and reuse it for later videos."""
    return easyocr.Reader(list(languages), gpu=gpu)


def _read_plate(
    crop: Any,
    reader: easyocr.Reader,
) -> tuple[str | None, float, Any | None]:
    """Read text in a vehicle crop and retain a crop around the best plate text."""
    best_text: str | None = None
    best_confidence = 0.0
    best_crop: Any | None = None
    image_height, image_width = crop.shape[:2]
    for bounds, text, confidence in reader.readtext(crop):
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        if (
            _PLATE_TEXT.fullmatch(normalized)
            and any(character.isalpha() for character in normalized)
            and any(character.isdigit() for character in normalized)
            and confidence > best_confidence
        ):
            best_text = normalized
            best_confidence = float(confidence)
            left = max(0, int(min(point[0] for point in bounds)))
            top = max(0, int(min(point[1] for point in bounds)))
            right = min(image_width, int(max(point[0] for point in bounds)))
            bottom = min(image_height, int(max(point[1] for point in bounds)))
            if right > left and bottom > top:
                best_crop = crop[top:bottom, left:right].copy()
    return best_text, best_confidence, best_crop


def _save_capture(image: Any, directory: str) -> str:
    """Save a compact JPEG capture and return its random, non-path identifier."""
    capture_id = uuid.uuid4().hex
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 75],
    )
    if not success:
        raise RuntimeError("Could not encode the vehicle image capture")
    (output_directory / f"{capture_id}.jpg").write_bytes(encoded.tobytes())
    return capture_id


class LiveCameraSession:
    """Track frames sent from a browser webcam and emit completed speed events."""

    def __init__(
        self,
        distance_m: float,
        line_a_ratio: float,
        line_b_ratio: float,
        confidence: float,
        settings: Settings,
    ) -> None:
        if distance_m <= 0:
            raise AnalysisInputError("distance_m must be greater than zero")
        if not 0 < line_a_ratio < 1 or not 0 < line_b_ratio < 1:
            raise AnalysisInputError("line ratios must be between zero and one")
        if line_a_ratio == line_b_ratio:
            raise AnalysisInputError(
                "line_a_ratio and line_b_ratio must be different"
            )
        if not 0 <= confidence <= 1:
            raise AnalysisInputError("confidence must be between zero and one")

        self.distance_m = distance_m
        self.line_a_ratio = line_a_ratio
        self.line_b_ratio = line_b_ratio
        self.confidence = confidence
        self.settings = settings
        self.detector = _load_detector(settings.yolo_model_path)
        # Each browser session gets a fresh tracker while reusing downloaded weights.
        self.detector.predictor = None
        self.reader = _load_ocr_reader(
            tuple(settings.ocr_languages),
            settings.ocr_gpu,
        )
        self.states: dict[int, TrackState] = {}
        self.started_at = time.monotonic()
        self.frame_count = 0

    def process_frame(self, frame: Any) -> list[dict[str, Any]]:
        """Process one webcam frame and return any newly completed speed events."""
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        now = time.monotonic()
        elapsed_session = now - self.started_at
        line_y = {
            "line_a": self.line_a_ratio * frame_height,
            "line_b": self.line_b_ratio * frame_height,
        }
        result = self.detector.track(
            source=frame,
            persist=True,
            conf=self.confidence,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []

        events: list[dict[str, Any]] = []
        boxes_xyxy = boxes.xyxy.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        for coordinates, track_id in zip(boxes_xyxy, track_ids):
            x1, y1, x2, y2 = (int(value) for value in coordinates)
            center_y = (y1 + y2) / 2
            state = self.states.get(track_id)
            if state is None:
                state = TrackState(previous_center_y=center_y)
                self.states[track_id] = state

            vehicle_crop = frame[
                max(0, y1):min(frame_height, y2),
                max(0, x1):min(frame_width, x2),
            ]
            if (
                self.frame_count % 5 == 0
                and state.plate_confidence < 0.9
                and vehicle_crop.size
            ):
                plate, plate_confidence, plate_crop = _read_plate(
                    vehicle_crop,
                    self.reader,
                )
                if plate and plate_confidence > state.plate_confidence:
                    state.plate_number = plate
                    state.plate_confidence = plate_confidence
                    state.plate_crop = plate_crop

            for line_name, crossing_y in line_y.items():
                if (
                    line_name not in state.crossing_times
                    and crossed_line(state.previous_center_y, center_y, crossing_y)
                ):
                    state.crossing_times[line_name] = elapsed_session

            if not state.emitted and len(state.crossing_times) == 2:
                elapsed = abs(
                    state.crossing_times["line_b"]
                    - state.crossing_times["line_a"]
                )
                if elapsed > 0:
                    speed_kmh = round(
                        estimate_speed_kmh(self.distance_m, elapsed),
                        2,
                    )
                    vehicle_capture_id = None
                    plate_capture_id = None
                    if speed_kmh > self.settings.speed_limit_kmh and vehicle_crop.size:
                        vehicle_capture_id = _save_capture(
                            vehicle_crop,
                            self.settings.capture_directory,
                        )
                        if state.plate_crop is not None:
                            plate_capture_id = _save_capture(
                                state.plate_crop,
                                self.settings.capture_directory,
                            )
                    events.append(
                        {
                            "track_id": track_id,
                            "plate_number": state.plate_number,
                            "plate_confidence": state.plate_confidence or None,
                            "vehicle_capture_id": vehicle_capture_id,
                            "plate_capture_id": plate_capture_id,
                            "speed_kmh": speed_kmh,
                            "direction": travel_direction(
                                self.line_a_ratio,
                                self.line_b_ratio,
                                state.crossing_times["line_a"],
                                state.crossing_times["line_b"],
                            ),
                            "frame": self.frame_count,
                            "timestamp_seconds": round(elapsed_session, 3),
                        }
                    )
                    state.emitted = True

            state.previous_center_y = center_y
        return events


def analyze_video(
    video_path: str,
    distance_m: float,
    line_a_ratio: float,
    line_b_ratio: float,
    confidence: float,
    settings: Settings,
    speed_limit_kmh: float = 50.0,
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
    if speed_limit_kmh <= 0:
        raise AnalysisInputError("speed_limit_kmh must be greater than zero")
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
                        plate, plate_confidence, plate_crop = _read_plate(
                            crop,
                            reader,
                        )
                        if plate and plate_confidence > state.plate_confidence:
                            state.plate_number = plate
                            state.plate_confidence = plate_confidence
                            state.plate_crop = plate_crop

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
                        speed_kmh = round(
                            estimate_speed_kmh(distance_m, elapsed),
                            2,
                        )
                        vehicle_crop = frame[
                            max(0, y1):min(frame_height, y2),
                            max(0, x1):min(frame_width, x2),
                        ]
                        vehicle_capture_id = None
                        plate_capture_id = None
                        if speed_kmh > speed_limit_kmh and vehicle_crop.size:
                            vehicle_capture_id = _save_capture(
                                vehicle_crop,
                                settings.capture_directory,
                            )
                            if state.plate_crop is not None:
                                plate_capture_id = _save_capture(
                                    state.plate_crop,
                                    settings.capture_directory,
                                )
                        events.append(
                            {
                                "track_id": track_id,
                                "plate_number": state.plate_number,
                                "plate_confidence": state.plate_confidence or None,
                                "vehicle_capture_id": vehicle_capture_id,
                                "plate_capture_id": plate_capture_id,
                                "speed_kmh": speed_kmh,
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
