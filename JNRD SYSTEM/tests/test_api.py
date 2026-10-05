"""Lightweight API tests; AI models are not loaded by these endpoints."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app

client = TestClient(app)


def test_health_does_not_require_model_initialization() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_serves_the_browser_app() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Vehicle Speed &amp; ANPR" in response.text
    assert 'fetch("/api/v1/analyze"' in response.text
    assert "getUserMedia" in response.text


def test_live_websocket_rejects_invalid_calibration() -> None:
    with client.websocket_connect("/api/v1/live") as websocket:
        websocket.send_json(
            {
                "distance_m": 10,
                "line_a_ratio": 0.5,
                "line_b_ratio": 0.5,
                "location": "Test camera",
            }
        )
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert "must be different" in response["detail"]


def test_live_websocket_processes_browser_camera_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    class FakeLiveSession:
        def __init__(self, *_args: object) -> None:
            self.frame_count = 0

        def process_frame(self, _frame: np.ndarray) -> list[dict]:
            self.frame_count += 1
            return []

    monkeypatch.setattr(main, "get_settings", lambda: Settings())
    monkeypatch.setattr(main, "LiveCameraSession", FakeLiveSession)
    monkeypatch.setattr(main, "acquire_analysis_lock", lambda: None)
    monkeypatch.setattr(main, "release_analysis_lock", lambda: None)
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((32, 48, 3), dtype=np.uint8),
    )
    assert success

    with client.websocket_connect("/api/v1/live") as websocket:
        websocket.send_json(
            {
                "distance_m": 10,
                "line_a_ratio": 0.35,
                "line_b_ratio": 0.65,
                "location": "Test camera",
            }
        )
        ready = websocket.receive_json()
        status = {}
        for _ in range(25):
            websocket.send_bytes(encoded.tobytes())
            status = websocket.receive_json()
        websocket.send_text("stop")

    assert ready["type"] == "ready"
    assert status == {"type": "frame_processed", "frames_processed": 25}


def test_offense_dashboard_endpoint_returns_cases() -> None:
    response = client.get("/api/v1/offenses")

    assert response.status_code == 200
    assert isinstance(response.json()["offenses"], list)


def test_public_config_does_not_expose_sms_credentials() -> None:
    response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json()["speed_limit_kmh"] > 0
    assert "africastalking_api_key" not in response.json()


def test_analyze_rejects_equal_measurement_lines() -> None:
    response = client.post(
        "/api/v1/analyze",
        data={
            "distance_m": "10",
            "line_a_ratio": "0.5",
            "line_b_ratio": "0.5",
        },
        files={"video": ("traffic.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "line_a_ratio and line_b_ratio must be different"
    )


def test_analyze_rejects_blank_camera_location(tmp_path) -> None:
    video_path = tmp_path / "traffic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 64),
    )
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    response = client.post(
        "/api/v1/analyze",
        data={
            "distance_m": "10",
            "line_a_ratio": "0.35",
            "line_b_ratio": "0.65",
            "location": "",
        },
        files={"video": ("traffic.mp4", video_path.read_bytes(), "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Camera location must not be empty"


def test_analyze_rejects_unsupported_video_extension() -> None:
    response = client.post(
        "/api/v1/analyze",
        data={"distance_m": "10"},
        files={"video": ("traffic.txt", b"not-a-video", "text/plain")},
    )

    assert response.status_code == 415
