"""Lightweight API tests; AI models are not loaded by these endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_does_not_require_model_initialization() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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


def test_analyze_rejects_unsupported_video_extension() -> None:
    response = client.post(
        "/api/v1/analyze",
        data={"distance_m": "10"},
        files={"video": ("traffic.txt", b"not-a-video", "text/plain")},
    )

    assert response.status_code == 415
