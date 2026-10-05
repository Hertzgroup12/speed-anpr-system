from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import SpeedEstimateRequest
from app.services.anpr_service import process_anpr_image
from app.services.firebase_service import initialize_firebase
from app.services.speed_service import estimate_speed

settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Vehicle Speed Detection and ANPR service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    initialize_firebase()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": settings.app_title}


@app.get("/api/v1/metadata")
def metadata() -> dict:
    return {
        "name": settings.app_title,
        "version": settings.app_version,
        "features": [
            "vehicle-detection",
            "number-plate-recognition",
            "speed-estimation",
            "firebase-integration",
        ],
    }


@app.post("/api/v1/anpr")
async def anpr(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        contents = await file.read()
        result = process_anpr_image(contents, file.filename)
        return result
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/speed")
def speed(payload: SpeedEstimateRequest) -> dict:
    try:
        speed_kmh = estimate_speed(
            distance_m=payload.distance_m,
            time_seconds=payload.time_seconds,
            vehicle_length_m=payload.vehicle_length_m,
        )
        return {
            "distance_m": payload.distance_m,
            "time_seconds": payload.time_seconds,
            "vehicle_length_m": payload.vehicle_length_m,
            "estimated_speed_kmh": speed_kmh,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
