"""Centralized environment-based application settings."""

from functools import lru_cache

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
    firebase_collection: str = "speed_events"


@lru_cache
def get_settings() -> Settings:
    """Reuse one validated settings object for the lifetime of the process."""
    return Settings()
