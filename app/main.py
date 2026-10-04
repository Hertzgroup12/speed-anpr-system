"""FastAPI application and video-analysis endpoint."""

import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.analyzer import AnalysisInputError, analyze_video
from app.config import get_settings
from app.firebase_store import persist_events

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
UPLOAD_CHUNK_SIZE = 1024 * 1024

app = FastAPI(
    title="Vehicle Speed Detection and ANPR",
    description=(
        "Detect and track vehicles, estimate speed between calibrated video "
        "lines, recognize visible plate text, and optionally save events to Firebase."
    ),
    version="1.0.0",
)


class VehicleEvent(BaseModel):
    track_id: int
    plate_number: str | None
    plate_confidence: float | None
    speed_kmh: float
    direction: str
    frame: int
    timestamp_seconds: float


class AnalysisResponse(BaseModel):
    filename: str
    fps: float
    width: int
    height: int
    frames_processed: int
    duration_seconds: float
    events: list[VehicleEvent]
    firebase_enabled: bool
    events_persisted: int


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


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze(
    video: Annotated[UploadFile, File(description="Traffic video to analyze")],
    distance_m: Annotated[float, Form(gt=0)],
    line_a_ratio: Annotated[float, Form(gt=0, lt=1)] = 0.35,
    line_b_ratio: Annotated[float, Form(gt=0, lt=1)] = 0.65,
    confidence: Annotated[float, Form(ge=0, le=1)] = 0.35,
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
        try:
            analysis = await run_in_threadpool(
                analyze_video,
                video_path,
                distance_m,
                line_a_ratio,
                line_b_ratio,
                confidence,
                settings,
            )
        except AnalysisInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        events_persisted = await run_in_threadpool(
            persist_events,
            analysis["events"],
            video.filename or "uploaded-video",
            settings,
        )
        return {
            "filename": video.filename or "uploaded-video",
            **analysis,
            "firebase_enabled": settings.firebase_enabled,
            "events_persisted": events_persisted,
        }
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
