"""FastAPI application and video-analysis endpoint."""

import os
import re
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

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
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.accounts import (
    authenticate_registered_user,
    invite_code_matches,
    register_user,
    valid_signup_username,
)
from app.analyzer import (
    AnalysisInputError,
    LiveCameraSession,
    acquire_analysis_lock,
    analyze_video,
    release_analysis_lock,
)
from app.auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE_SECONDS,
    auth_is_configured,
    create_session_token,
    credentials_match,
    verify_session_token,
)
from app.chatbot import MAX_MESSAGE_LENGTH, ask_gemini
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
CHAT_REQUESTS_PER_MINUTE = 10
LOGIN_ATTEMPTS_PER_WINDOW = 10
LOGIN_WINDOW_SECONDS = 15 * 60
SIGNUP_ATTEMPTS_PER_WINDOW = 10
_CHAT_REQUESTS: dict[str, deque[float]] = {}
_CHAT_REQUESTS_LOCK = threading.Lock()
_LOGIN_ATTEMPTS: dict[str, deque[float]] = {}
_LOGIN_ATTEMPTS_LOCK = threading.Lock()

app = FastAPI(
    title="JNRD PRO",
    description=(
        "Detect and track vehicles, estimate speed between calibrated video "
        "lines, recognize visible plate text, and create human-review cases for "
        "speed estimates over the configured limit."
    ),
    version="1.0.0",
)

WEB_APP_FILE = Path(__file__).parent / "static" / "index.html"


def _secure_session_cookie(request: Request) -> bool:
    """Use Secure cookies for HTTPS, including TLS-terminating proxy tunnels."""
    return (
        get_settings().auth_cookie_secure
        or request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").split(",", maxsplit=1)[0]
        .strip()
        .lower()
        == "https"
    )


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=256)
    invite_code: str = Field(min_length=1, max_length=256)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


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


@app.middleware("http")
async def protect_operator_routes(request: Request, call_next: Any) -> Any:
    """Require an authenticated session for API, capture, and docs routes."""
    response = None
    protected_path = (
        request.url.path.startswith("/api/")
        or request.url.path in {"/docs", "/redoc", "/openapi.json"}
    )
    public_auth_paths = {
        "/api/v1/auth/login",
        "/api/v1/auth/logout",
        "/api/v1/auth/session",
    }
    if protected_path:
        origin = request.headers.get("origin")
        if (
            request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and origin
            and urlsplit(origin).netloc != request.headers.get("host")
        ):
            response = JSONResponse(
                status_code=403,
                content={"detail": "Cross-origin requests are not allowed"},
            )
        if (
            response is None
            and request.url.path not in public_auth_paths
            and request.url.path != "/health"
        ):
            settings = get_settings()
            if not auth_is_configured(settings):
                response = JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "Login is not configured. Set APP_USERNAME, "
                            "APP_PASSWORD, and a 32-character APP_SECRET_KEY."
                        )
                    },
                )
            elif not verify_session_token(
                request.cookies.get(SESSION_COOKIE),
                settings,
            ):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Please sign in to continue"},
                )
    else:
        response = None

    if response is None:
        response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "media-src 'self' blob:; connect-src 'self' ws: wss:; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


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
        "chat_enabled": bool(settings.gemini_api_key),
    }


@app.get("/api/v1/auth/session")
def auth_session(request: Request) -> dict[str, bool]:
    """Tell the login page whether credentials and a valid session are present."""
    settings = get_settings()
    return {
        "configured": auth_is_configured(settings),
        "authenticated": verify_session_token(
            request.cookies.get(SESSION_COOKIE),
            settings,
        ),
        "signup_enabled": bool(
            settings.firebase_enabled
            and settings.admin_invite_code
            and len(settings.admin_invite_code) >= 20
        ),
    }


@app.post("/api/v1/auth/login")
def login(request: Request, credentials: LoginRequest) -> JSONResponse:
    """Authenticate the single configured operator and set an HTTP-only cookie."""
    settings = get_settings()
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=503,
            detail=(
                "Login is not configured. Set APP_USERNAME, APP_PASSWORD, "
                "and a 32-character APP_SECRET_KEY."
            ),
        )
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _LOGIN_ATTEMPTS_LOCK:
        recent_attempts = _LOGIN_ATTEMPTS.setdefault(client_host, deque())
        while recent_attempts and now - recent_attempts[0] >= LOGIN_WINDOW_SECONDS:
            recent_attempts.popleft()
        if len(recent_attempts) >= LOGIN_ATTEMPTS_PER_WINDOW:
            raise HTTPException(
                status_code=429,
                detail="Too many sign-in attempts. Wait 15 minutes before trying again.",
            )
    authenticated = credentials_match(
        credentials.username,
        credentials.password,
        settings,
    )
    if not authenticated:
        try:
            authenticated = authenticate_registered_user(
                credentials.username,
                credentials.password,
                settings,
            )
        except GoogleAPICallError as error:
            raise HTTPException(
                status_code=503,
                detail="Account service is temporarily unavailable",
            ) from error
    if not authenticated:
        with _LOGIN_ATTEMPTS_LOCK:
            _LOGIN_ATTEMPTS[client_host].append(now)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    with _LOGIN_ATTEMPTS_LOCK:
        _LOGIN_ATTEMPTS.pop(client_host, None)
    response = JSONResponse({"authenticated": True})
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(settings, credentials.username),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=_secure_session_cookie(request),
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/v1/auth/signup", status_code=201)
def signup(request: Request, credentials: SignupRequest) -> JSONResponse:
    """Create an invite-only Firestore account and sign the new user in."""
    settings = get_settings()
    if not auth_is_configured(settings):
        raise HTTPException(
            status_code=503,
            detail="Login signing is not configured on this server",
        )
    if not settings.firebase_enabled:
        raise HTTPException(
            status_code=503,
            detail="New account registration is unavailable because Firebase is disabled",
        )
    if not settings.admin_invite_code or len(settings.admin_invite_code) < 20:
        raise HTTPException(
            status_code=503,
            detail="Invite-only sign-up is not configured on this server",
        )

    if not valid_signup_username(credentials.username):
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    if (
        settings.app_username
        and credentials.username.strip().casefold()
        == settings.app_username.strip().casefold()
    ):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _LOGIN_ATTEMPTS_LOCK:
        recent_attempts = _LOGIN_ATTEMPTS.setdefault(
            f"signup:{client_host}",
            deque(),
        )
        while recent_attempts and now - recent_attempts[0] >= LOGIN_WINDOW_SECONDS:
            recent_attempts.popleft()
        if len(recent_attempts) >= SIGNUP_ATTEMPTS_PER_WINDOW:
            raise HTTPException(
                status_code=429,
                detail="Too many sign-up attempts. Wait 15 minutes before trying again.",
            )
        recent_attempts.append(now)

    if not invite_code_matches(credentials.invite_code, settings):
        raise HTTPException(status_code=403, detail="Invalid invitation code")
    try:
        register_user(credentials.username, credentials.password, settings)
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except GoogleAPICallError as error:
        raise HTTPException(
            status_code=503,
            detail="Account service is temporarily unavailable",
        ) from error

    username = credentials.username.strip().casefold()
    response = JSONResponse(
        {"authenticated": True, "username": username},
        status_code=201,
    )
    response.set_cookie(
        SESSION_COOKIE,
        create_session_token(settings, username),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=_secure_session_cookie(request),
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/v1/auth/logout")
def logout(request: Request) -> JSONResponse:
    """Clear the operator's session cookie."""
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=_secure_session_cookie(request),
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/v1/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    """Answer a bounded operator question without sending case data to Gemini."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Message must be at most {MAX_MESSAGE_LENGTH} characters",
        )
    now = time.monotonic()
    with _CHAT_REQUESTS_LOCK:
        recent_requests = _CHAT_REQUESTS.setdefault(
            get_settings().app_username or "operator",
            deque(),
        )
        while recent_requests and now - recent_requests[0] >= 60:
            recent_requests.popleft()
        if len(recent_requests) >= CHAT_REQUESTS_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Chat limit reached. Please wait a minute before trying again.",
            )
        recent_requests.append(now)
    try:
        answer = await run_in_threadpool(ask_gemini, message, get_settings())
    except RuntimeError as error:
        status_code = 429 if "rate limiting" in str(error) else 503
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    return {"reply": answer}


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
    origin = websocket.headers.get("origin")
    if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
        await websocket.close(code=4403, reason="Cross-origin connection denied")
        return
    if not verify_session_token(
        websocket.cookies.get(SESSION_COOKIE),
        get_settings(),
    ):
        await websocket.close(code=4401, reason="Sign in required")
        return

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
