"""Centralized environment-based application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings can be supplied through environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yolo_model_path: str = "yolov8n.pt"
    max_upload_mb: int = 250
    ocr_languages: list[str] = ["en"]
    ocr_gpu: bool = False
    firebase_enabled: bool = False
    firebase_project_id: str | None = None
    firebase_credentials: str | None = None
    firebase_web_api_key: str | None = None
    firebase_auth_domain: str | None = None
    firebase_app_id: str | None = None
    firebase_measurement_id: str | None = None
    firebase_collection: str = "speed_events"
    offense_collection: str = "speed_offenses"
    app_username: str | None = None
    app_password: str | None = None
    app_secret_key: str | None = None
    admin_invite_code: str | None = None
    auth_cookie_secure: bool = False
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    speed_limit_kmh: float = Field(default=50.0, gt=0)
    camera_location: str = "Demo camera"
    capture_directory: str = "data/captures"
    sms_simulation: bool = True
    driver_directory: dict[str, str] = Field(default_factory=dict)
    africastalking_username: str | None = None
    africastalking_api_key: str | None = None
    africastalking_sender_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Reuse one validated settings object for the lifetime of the process."""
    return Settings()
