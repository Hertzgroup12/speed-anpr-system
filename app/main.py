"""FastAPI application and video-analysis endpoint."""

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.analyzer import (
    AnalysisInputError,
    LiveCameraSession,
    acquire_analysis_lock,
    analyze_video,
    release_analysis_lock,
)
from app.config import get_settings
from app.firebase_store import persist_events
from app.offenses import (
    create_offenses,
    list_offenses,
    notify_driver,
    update_case_decision,
)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
UPLOAD_CHUNK_SIZE = 1024 * 1024

app = FastAPI(
    title="Vehicle Speed Detection and ANPR",
    description=(
        "Detect and track vehicles, estimate speed between calibrated video "
        "lines, recognize visible plate text, and create human-review cases for "
        "speed estimates over the configured limit."
    ),
    version="1.0.0",
)

WEB_APP_FILE = Path(__file__).parent / "static" / "index.html"


class VehicleEvent(BaseModel):
    track_id: int
    plate_number: str | None
    plate_confidence: float | None
    speed_kmh: float
    direction: str
    frame: int
    timestamp_seconds: float
    offense: bool
    case_id: str | None
    vehicle_capture_url: str | None
    plate_capture_url: str | None


class AnalysisResponse(BaseModel):
    filename: str
    fps: float
    width: int
    height: int
    frames_processed: int
    duration_seconds: float
    detected_at: str
    location: str
    speed_limit_kmh: float
    events: list[VehicleEvent]
    offenses_created: int
    firebase_enabled: bool
    events_persisted: int


class ReviewDecisionRequest(BaseModel):
    decision: Literal["reviewed", "dismissed"]


class LiveCameraConfig(BaseModel):
    distance_m: float = Field(gt=0)
    line_a_ratio: float = Field(default=0.35, gt=0, lt=1)
    line_b_ratio: float = Field(default=0.65, gt=0, lt=1)
    confidence: float = Field(default=0.35, ge=0, le=1)
    location: str = Field(min_length=1, max_length=150)


def _record_events(
    events: list[dict[str, Any]],
    source_name: str,
    location: str,
    detected_at: str,
    settings: Any,
) -> tuple[int, int]:
    """Attach review cases and storage metadata to events from either input mode."""
    metadata = {
        "location": location,
        "detected_at": detected_at,
        "speed_limit_kmh": settings.speed_limit_kmh,
    }
    for event in events:
        event["offense"] = event["speed_kmh"] > settings.speed_limit_kmh
        event["case_id"] = None
        event["vehicle_capture_url"] = (
            f"/api/v1/captures/{event['vehicle_capture_id']}.jpg"
            if event["vehicle_capture_id"]
            else None
        )
        event["plate_capture_url"] = (
            f"/api/v1/captures/{event['plate_capture_id']}.jpg"
            if event["plate_capture_id"]
            else None
        )

    offense_cases = create_offenses(
        events,
        source_name,
        location,
        detected_at,
        settings,
    )
    case_by_track = {case["track_id"]: case["case_id"] for case in offense_cases}
    for event in events:
        event["case_id"] = case_by_track.get(event["track_id"])

    events_persisted = persist_events(events, source_name, settings, metadata)
    return len(offense_cases), events_persisted


async def _save_upload(upload: UploadFile, max_bytes: int) -> tuple[str, int]:
    """Stream an upload to a private temporary file while enforcing its size limit."""
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        await upload.close()
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video extension. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    temporary = tempfile.NamedTemporaryFile(
        prefix="anpr-",
        suffix=extension,
        delete=False,
    )
    total_bytes = 0
    keep_file = False
    try:
        with temporary:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
                    )
                temporary.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty")
        await upload.close()
        keep_file = True
        return temporary.name, total_bytes
    finally:
        if not keep_file and os.path.exists(temporary.name):
            os.unlink(temporary.name)
        await upload.close()


@app.get("/health")
def health() -> dict[str, str]:
    """Return API liveness without loading the machine-learning models."""
    return {"status": "ok"}


@app.get("/api/v1/config")
def public_config() -> dict[str, Any]:
    """Expose non-secret settings that help an operator interpret the dashboard."""
    settings = get_settings()
    return {
        "speed_limit_kmh": settings.speed_limit_kmh,
        "camera_location": settings.camera_location,
        "sms_simulation": settings.sms_simulation,
        "firebase_enabled": settings.firebase_enabled,
    }


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the browser client from the same origin as the API."""
    return FileResponse(WEB_APP_FILE)


@app.get("/api/v1/captures/{capture_id}.jpg", include_in_schema=False)
def get_capture(capture_id: str) -> FileResponse:
    """Serve a locally retained image using its random capture identifier."""
    if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
        raise HTTPException(status_code=404, detail="Capture not found")
    capture_path = Path(get_settings().capture_directory) / f"{capture_id}.jpg"
    if not capture_path.is_file():
        raise HTTPException(status_code=404, detail="Capture not found")
    return FileResponse(capture_path, media_type="image/jpeg")


@app.get("/api/v1/offenses")
def offenses() -> dict[str, Any]:
    """List speed-estimate cases for the operator dashboard."""
    return {"offenses": list_offenses(get_settings())}


@app.patch("/api/v1/offenses/{case_id}")
def review_offense(
    case_id: str,
    request: ReviewDecisionRequest,
) -> dict[str, Any]:
    """Require an operator to review or dismiss a speed-estimate case."""
    case = update_case_decision(case_id, request.decision, get_settings())
    if case is None:
        raise HTTPException(status_code=404, detail="Offense case not found")
    return {"offense": case}


@app.post("/api/v1/offenses/{case_id}/notify")
def send_driver_notice(case_id: str) -> dict[str, Any]:
    """Simulate or manually send a notice only after a case is reviewed."""
    try:
        return notify_driver(case_id, get_settings())
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.websocket("/api/v1/live")
async def live_camera(websocket: WebSocket) -> None:
    """Receive browser webcam frames and report live speed events."""
    await websocket.accept()
    lock_acquired = False
    try:
        raw_config = await websocket.receive_json()
        try:
            camera_config = LiveCameraConfig.model_validate(raw_config)
            if camera_config.line_a_ratio == camera_config.line_b_ratio:
                raise ValueError(
                    "line_a_ratio and line_b_ratio must be different"
                )
        except (ValidationError, ValueError) as error:
            await websocket.send_json({"type": "error", "detail": str(error)})
            await websocket.close(code=4400)
            return

        settings = get_settings()
        if settings.max_upload_mb < 1:
            await websocket.send_json(
                {"type": "error", "detail": "The server upload limit is misconfigured"}
            )
            await websocket.close(code=1011)
            return

        # YOLO trackers are stateful; only one upload or webcam session can use them.
        await run_in_threadpool(acquire_analysis_lock)
        lock_acquired = True
        session = await run_in_threadpool(
            LiveCameraSession,
            camera_config.distance_m,
            camera_config.line_a_ratio,
            camera_config.line_b_ratio,
            camera_config.confidence,
            settings,
        )
        await websocket.send_json(
            {
                "type": "ready",
                "speed_limit_kmh": settings.speed_limit_kmh,
                "location": camera_config.location,
            }
        )

        invalid_frames = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            payload = message.get("bytes")
            if payload is None:
                if message.get("text") == "stop":
                    break
                continue
            if len(payload) > 2 * 1024 * 1024:
                await websocket.send_json(
                    {"type": "error", "detail": "Camera frame exceeds the 2 MB limit"}
                )
                continue

            def process_payload() -> tuple[list[dict[str, Any]], int]:
                image = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise AnalysisInputError("Could not decode webcam frame")
                return session.process_frame(image), session.frame_count

            try:
                events, frame_count = await run_in_threadpool(process_payload)
            except AnalysisInputError as error:
                invalid_frames += 1
                await websocket.send_json({"type": "error", "detail": str(error)})
                if invalid_frames >= 5:
                    await websocket.close(code=4400)
                    break
                continue

            if events:
                detected_at = datetime.now(timezone.utc).isoformat()
                offenses_created, events_persisted = await run_in_threadpool(
                    _record_events,
                    events,
                    "live-webcam",
                    camera_config.location,
                    detected_at,
                    settings,
                )
                await websocket.send_json(
                    {
                        "type": "events",
                        "events": events,
                        "detected_at": detected_at,
                        "offenses_created": offenses_created,
                        "events_persisted": events_persisted,
                    }
                )
            else:
                await websocket.send_json(
                    {"type": "frame_processed", "frames_processed": frame_count}
                )
    except WebSocketDisconnect:
        pass
    finally:
        if lock_acquired:
            await run_in_threadpool(release_analysis_lock)


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    video: Annotated[UploadFile, File(description="Traffic video to analyze")],
    distance_m: Annotated[float, Form(gt=0)],
    line_a_ratio: Annotated[float, Form(gt=0, lt=1)] = 0.35,
    line_b_ratio: Annotated[float, Form(gt=0, lt=1)] = 0.65,
    confidence: Annotated[float, Form(ge=0, le=1)] = 0.35,
    location: Annotated[str | None, Form(max_length=150)] = None,
) -> dict[str, Any]:
    """Analyze an uploaded video using two virtual lines and an actual distance."""
    if line_a_ratio == line_b_ratio:
        raise HTTPException(
            status_code=422,
            detail="line_a_ratio and line_b_ratio must be different",
        )

    settings = get_settings()
    if settings.max_upload_mb < 1:
        raise HTTPException(
            status_code=500,
            detail="MAX_UPLOAD_MB must be configured as a positive integer",
        )

    video_path, _size = await _save_upload(
        video,
        settings.max_upload_mb * 1024 * 1024,
    )
    try:
        form_data = await request.form()
        raw_location = form_data.get("location")
        if raw_location == "":
            raise HTTPException(
                status_code=422,
                detail="Camera location must not be empty",
            )
        resolved_location = (
            raw_location.strip()
            if raw_location is not None
            else settings.camera_location.strip()
            if location is None
            else location.strip()
        )
        camera_location = resolved_location
        if not camera_location:
            raise HTTPException(
                status_code=422,
                detail="Camera location must not be empty",
            )
        detected_at = datetime.now(timezone.utc).isoformat()
        try:
            analysis = await run_in_threadpool(
                analyze_video,
                video_path,
                distance_m,
                line_a_ratio,
                line_b_ratio,
                confidence,
                settings,
                settings.speed_limit_kmh,
            )
        except AnalysisInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        offenses_created, events_persisted = await run_in_threadpool(
            _record_events,
            analysis["events"],
            video.filename or "uploaded-video",
            camera_location,
            detected_at,
            settings,
        )

        return {
            "filename": video.filename or "uploaded-video",
            **analysis,
            "detected_at": detected_at,
            "location": camera_location,
            "speed_limit_kmh": settings.speed_limit_kmh,
            "offenses_created": offenses_created,
            "firebase_enabled": settings.firebase_enabled,
            "events_persisted": events_persisted,
        }
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
