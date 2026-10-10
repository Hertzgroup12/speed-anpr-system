"""Lightweight API tests; AI models are not loaded by these endpoints."""

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def sign_in_api_client() -> Iterator[None]:
    from app import main
    from app.config import get_settings

    get_settings.cache_clear()
    with main._LOGIN_ATTEMPTS_LOCK:
        main._LOGIN_ATTEMPTS.clear()
    client.cookies.clear()
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "test-operator", "password": "test-password-123"},
    )
    assert response.status_code == 200
    yield
    client.cookies.clear()
    with main._LOGIN_ATTEMPTS_LOCK:
        main._LOGIN_ATTEMPTS.clear()
    get_settings.cache_clear()


def test_health_does_not_require_model_initialization() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_serves_the_browser_app() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>JNRD PRO</title>" in response.text
    assert "<h1>JNRD PRO</h1>" in response.text
    assert 'id="login-form"' in response.text
    assert 'id="login-theme-toggle"' in response.text
    assert 'class="auth-heading"' in response.text
    assert 'id="create-account-button"' in response.text
    assert 'id="signup-form"' in response.text
    assert 'fetch("/api/v1/auth/signup"' in response.text
    assert 'id="logout-button"' in response.text
    assert 'class="control-icon dashboard-icon"' in response.text
    assert 'class="control-icon logout-icon"' in response.text
    assert 'id="workspace-navigation-label">Speed cases</span>' in response.text
    assert 'workspaceNavigationLabel.textContent = "Dashboard"' in response.text
    assert 'id="theme-toggle"' in response.text
    assert 'class="sun-icon"' in response.text
    assert 'class="moon-icon"' in response.text
    assert 'aria-label="Switch to dark mode"' in response.text
    assert 'aria-pressed="false"' in response.text
    assert 'localStorage.setItem(currentThemeKey' in response.text
    assert 'const authThemeKey = "jnrd-theme:auth"' in response.text
    assert 'id="inspect-video-location"' in response.text
    assert 'fetch("/api/v1/location/video"' in response.text
    assert "navigator.geolocation.getCurrentPosition" in response.text
    assert 'id="chat-form"' in response.text
    assert 'fetch("/api/v1/analyze"' in response.text
    assert "getUserMedia" in response.text


def test_speed_cases_open_in_a_dedicated_page() -> None:
    response = client.get("/cases")

    assert response.status_code == 200
    assert 'id="offenses-panel"' in response.text
    assert 'href="/cases" target="_blank"' in response.text
    assert 'window.location.pathname === "/cases"' in response.text
    assert ':root[data-page="cases"] #operator-app > section:not(#offenses-panel)' in response.text
    assert 'id="case-report-form"' in response.text
    assert 'value="weekly">Current calendar week' in response.text
    assert 'value="monthly">Current calendar month' in response.text
    assert 'value="yearly">Current calendar year' in response.text
    assert '/api/v1/offenses/report.pdf?period=' in response.text


def test_case_report_download_is_account_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    from app import main
    from app.accounts import user_id_for_username
    from app.time_utils import GMT

    settings = main.get_settings()
    requested_owner_ids: list[str] = []
    detected_at = datetime.now(GMT).isoformat()
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def list_user_cases(_settings, owner_id):
        requested_owner_ids.append(owner_id)
        return [
            {
                "case_id": "case-for-report",
                "plate_number": "ABC1234",
                "speed_kmh": 61,
                "speed_limit_kmh": 50,
                "location": "Test road",
                "detected_at": detected_at,
                "review_status": "reviewed",
                "notification_status": "not_sent",
            }
        ]

    monkeypatch.setattr(main, "list_offenses", list_user_cases)

    response = client.get("/api/v1/offenses/report.pdf?period=monthly")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert 'attachment; filename="speed-case-review-monthly-' in response.headers[
        "content-disposition"
    ]
    assert response.content.startswith(b"%PDF-")
    assert requested_owner_ids == [user_id_for_username(settings.app_username or "")]


def test_case_report_rejects_unsupported_period() -> None:
    response = client.get("/api/v1/offenses/report.pdf?period=daily")

    assert response.status_code == 422


def test_openapi_uses_the_product_name() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "JNRD PRO"


def test_operator_api_requires_login() -> None:
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.get("/api/v1/config")

    assert response.status_code == 401


def test_login_rejects_invalid_credentials() -> None:
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "wrong-user", "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert "jnrd_session" not in unauthenticated_client.cookies


def test_signup_is_public_but_still_checks_registration_configuration() -> None:
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.post(
        "/api/v1/auth/signup",
        json={
            "username": "new-operator@example.com",
            "password": "a-long-test-password",
            "invite_code": "test-invitation-code-that-is-long",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "New account registration is unavailable because Firebase is disabled"
    )


def test_login_rate_limits_repeated_invalid_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(main, "LOGIN_ATTEMPTS_PER_WINDOW", 2)
    rate_limited_client = TestClient(app, client=("192.0.2.1", 50000))

    first = rate_limited_client.post(
        "/api/v1/auth/login",
        json={"username": "wrong-user", "password": "wrong-password"},
    )
    second = rate_limited_client.post(
        "/api/v1/auth/login",
        json={"username": "wrong-user", "password": "wrong-password"},
    )
    third = rate_limited_client.post(
        "/api/v1/auth/login",
        json={"username": "test-operator", "password": "test-password-123"},
    )

    assert first.status_code == 401
    assert second.status_code == 401
    assert third.status_code == 429


def test_login_cookie_is_http_only_and_strict() -> None:
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "test-operator", "password": "test-password-123"},
    )

    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie


def test_registered_user_login_can_access_protected_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(main, "authenticate_registered_user", lambda *_args: True)
    monkeypatch.setattr(main, "record_auth_event", lambda *_args: None)
    registered_client = TestClient(app)
    response = registered_client.post(
        "/api/v1/auth/login",
        json={
            "username": "registered@example.com",
            "password": "a-long-test-password",
        },
    )

    assert response.status_code == 200
    assert registered_client.get("/api/v1/config").status_code == 200


def test_successful_login_records_an_authentication_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    recorded_events: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        main,
        "record_auth_event",
        lambda username, action, auth_method, _settings: recorded_events.append(
            (username, action, auth_method)
        ),
    )
    unauthenticated_client = TestClient(app)
    response = unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"username": "test-operator", "password": "test-password-123"},
    )

    assert response.status_code == 200
    assert recorded_events == [("test-operator", "login", "configured_operator")]


def test_logout_clears_session_and_protected_routes_reject_it() -> None:
    response = client.post("/api/v1/auth/logout")
    protected_response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert protected_response.status_code == 401


def test_operator_api_rejects_cross_origin_mutations() -> None:
    response = client.post(
        "/api/v1/chat",
        headers={"Origin": "https://attacker.example"},
        json={"message": "hello"},
    )

    assert response.status_code == 403


def test_api_fails_closed_when_login_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None))
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.get("/api/v1/config")

    assert response.status_code == 503


def test_chat_endpoint_uses_server_side_gemini_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(
        main,
        "ask_gemini",
        lambda message, _settings: f"Help for: {message}",
    )

    response = client.post("/api/v1/chat", json={"message": "How do I review a case?"})

    assert response.status_code == 200
    assert response.json() == {"reply": "Help for: How do I review a case?"}


def test_chat_endpoint_rejects_empty_messages() -> None:
    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422


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

    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            app_username="test-operator",
            app_password="test-password-123",
            app_secret_key="test-session-secret-that-is-long-enough",
        ),
    )
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
    assert response.json()["firebase_enabled"] is False
    assert "africastalking_api_key" not in response.json()


def test_public_config_omits_incomplete_firebase_analytics_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            app_username="test-operator",
            app_password="test-password-123",
            app_secret_key="test-session-secret-that-is-long-enough",
            firebase_web_api_key="public-test-key",
        ),
    )

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json()["firebase_analytics_config"] is None


def test_public_config_includes_complete_firebase_analytics_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: Settings(
            app_username="test-operator",
            app_password="test-password-123",
            app_secret_key="test-session-secret-that-is-long-enough",
            firebase_project_id="test-project",
            firebase_web_api_key="public-test-key",
            firebase_auth_domain="test-project.firebaseapp.com",
            firebase_app_id="1:123:web:abcdef",
            firebase_measurement_id="G-TEST123456",
        ),
    )

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert response.json()["firebase_analytics_config"] == {
        "apiKey": "public-test-key",
        "authDomain": "test-project.firebaseapp.com",
        "projectId": "test-project",
        "appId": "1:123:web:abcdef",
        "measurementId": "G-TEST123456",
    }


def test_video_location_api_returns_embedded_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(
        main,
        "extract_video_location",
        lambda _path: {"latitude": 37.421998, "longitude": -122.084},
    )

    response = client.post(
        "/api/v1/location/video",
        files={"video": ("traffic.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "location": {"latitude": 37.421998, "longitude": -122.084},
    }


def test_video_location_api_reports_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(main, "extract_video_location", lambda _path: None)

    response = client.post(
        "/api/v1/location/video",
        files={"video": ("traffic.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == {"available": False, "location": None}


def test_analyze_uses_embedded_video_coordinates_for_recorded_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    monkeypatch.setattr(
        main,
        "extract_video_location",
        lambda _path: {"latitude": 37.421998, "longitude": -122.084},
    )
    monkeypatch.setattr(
        main,
        "analyze_video",
        lambda *_args: {
            "fps": 10.0,
            "width": 64,
            "height": 64,
            "frames_processed": 10,
            "duration_seconds": 1.0,
            "events": [],
        },
    )
    recorded = {}

    def record_events(
        _events,
        _source,
        location,
        _detected_at,
        _settings,
        _owner_id,
    ):
        recorded["location"] = location
        return 0, 0

    monkeypatch.setattr(main, "_record_events", record_events)

    response = client.post(
        "/api/v1/analyze",
        data={
            "distance_m": "10",
            "line_a_ratio": "0.35",
            "line_b_ratio": "0.65",
            "location": "Roadside camera",
        },
        files={"video": ("traffic.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["video_location"] == {
        "latitude": 37.421998,
        "longitude": -122.084,
    }
    assert response.json()["location"] == (
        "Roadside camera · GPS 37.421998, -122.084000"
    )
    assert recorded["location"] == response.json()["location"]


def test_accounts_have_private_settings_cases_and_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main
    from app.accounts import user_id_for_username
    from app.offenses import create_offenses

    monkeypatch.setattr(main, "authenticate_registered_user", lambda *_args: True)
    monkeypatch.setattr(main, "record_auth_event", lambda *_args: None)
    settings = main.get_settings()
    registered_username = "private.workspace@example.com"
    registered_user_id = user_id_for_username(registered_username)
    operator_user_id = user_id_for_username(settings.app_username or "")
    registered_client = TestClient(app)
    login = registered_client.post(
        "/api/v1/auth/login",
        json={
            "username": registered_username,
            "password": "a-long-test-password",
        },
    )
    assert login.status_code == 200

    saved_settings = registered_client.put(
        "/api/v1/settings",
        json={
            "speed_limit_kmh": 37,
            "camera_location": "Registered user's road",
        },
    )
    assert saved_settings.status_code == 200
    assert saved_settings.json()["settings"] == {
        "speed_limit_kmh": 37,
        "camera_location": "Registered user's road",
    }
    assert registered_client.get("/api/v1/config").json()["username"] == registered_username
    assert client.get("/api/v1/config").json()["camera_location"] == settings.camera_location

    event = {
        "track_id": 1,
        "plate_number": "TEST1234",
        "plate_confidence": 0.9,
        "speed_kmh": settings.speed_limit_kmh + 1,
        "timestamp_seconds": 1.5,
        "vehicle_capture_id": "c" * 32,
        "plate_capture_id": None,
    }
    operator_case = create_offenses(
        [event.copy()],
        "operator.mp4",
        "Operator road",
        "2026-01-01T00:00:00+00:00",
        settings,
        operator_user_id,
    )[0]
    registered_event = {**event, "vehicle_capture_id": "d" * 32}
    registered_case = create_offenses(
        [registered_event],
        "registered.mp4",
        "Registered road",
        "2026-01-01T00:00:00+00:00",
        settings,
        registered_user_id,
    )[0]

    operator_cases = client.get("/api/v1/offenses").json()["offenses"]
    registered_cases = registered_client.get("/api/v1/offenses").json()["offenses"]
    assert operator_case["case_id"] in {case["case_id"] for case in operator_cases}
    assert registered_case["case_id"] not in {case["case_id"] for case in operator_cases}
    assert registered_case["case_id"] in {
        case["case_id"] for case in registered_cases
    }
    assert operator_case["case_id"] not in {
        case["case_id"] for case in registered_cases
    }

    assert client.patch(
        f"/api/v1/offenses/{registered_case['case_id']}",
        json={"decision": "reviewed"},
    ).status_code == 404
    assert client.get(
        f"/api/v1/captures/{registered_event['vehicle_capture_id']}.jpg"
    ).status_code == 404


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
