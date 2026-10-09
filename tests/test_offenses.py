"""Tests for speed-case review and simulation-only driver notifications."""

import pytest

from app.config import Settings
from app.offenses import create_offenses, notify_driver, update_case_decision
from app.time_utils import gmt_now_iso
from fastapi.testclient import TestClient


def make_event(speed_kmh: float, plate_number: str = "TEST1234") -> dict:
    return {
        "track_id": 7,
        "plate_number": plate_number,
        "plate_confidence": 0.9,
        "speed_kmh": speed_kmh,
        "timestamp_seconds": 2.5,
        "vehicle_capture_id": "a" * 32,
        "plate_capture_id": "b" * 32,
    }


def test_create_offenses_flags_only_speeds_above_limit() -> None:
    settings = Settings(speed_limit_kmh=50)

    cases = create_offenses(
        [make_event(50), make_event(51)],
        "test.mp4",
        "Test road",
        gmt_now_iso(),
        settings,
    )

    assert len(cases) == 1
    assert cases[0]["speed_kmh"] == 51
    assert cases[0]["excess_kmh"] == 1
    assert cases[0]["review_status"] == "pending_review"


def test_simulated_notice_requires_review_and_is_not_repeatable() -> None:
    settings = Settings(
        speed_limit_kmh=50,
        sms_simulation=True,
        driver_directory={"TEST1234": "SIMULATED_DRIVER_001"},
    )
    cases = create_offenses(
        [make_event(61)],
        "test.mp4",
        "Test road",
        gmt_now_iso(),
        settings,
    )
    case_id = cases[0]["case_id"]

    with pytest.raises(ValueError, match="must be reviewed"):
        notify_driver(case_id, settings)

    assert update_case_decision(case_id, "reviewed", settings) is not None
    result = notify_driver(case_id, settings)

    assert result == {
        "case_id": case_id,
        "notification_status": "simulated",
        "simulation": True,
    }
    with pytest.raises(ValueError, match="already has"):
        notify_driver(case_id, settings)


def test_simulated_notice_requires_a_driver_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DRIVER_DIRECTORY", raising=False)
    settings = Settings(
        speed_limit_kmh=50,
        sms_simulation=True,
        driver_directory={},
        _env_file=None,
    )
    cases = create_offenses(
        [make_event(61)],
        "test.mp4",
        "Test road",
        gmt_now_iso(),
        settings,
    )
    update_case_decision(cases[0]["case_id"], "reviewed", settings)

    with pytest.raises(LookupError, match="No driver phone number"):
        notify_driver(cases[0]["case_id"], settings)


def test_reviewed_case_can_use_dashboard_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    settings = Settings(
        speed_limit_kmh=50,
        sms_simulation=True,
        driver_directory={"TEST1234": "SIMULATED_DRIVER_002"},
        app_username="test-operator",
        app_password="test-password-123",
        app_secret_key="test-session-secret-that-is-long-enough",
    )
    cases = create_offenses(
        [make_event(62)],
        "dashboard-test.mp4",
        "Test road",
        gmt_now_iso(),
        settings,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    client = TestClient(main.app)
    login = client.post(
        "/api/v1/auth/login",
        json={"username": settings.app_username, "password": settings.app_password},
    )
    assert login.status_code == 200
    case_id = cases[0]["case_id"]

    blocked = client.post(f"/api/v1/offenses/{case_id}/notify")
    reviewed = client.patch(
        f"/api/v1/offenses/{case_id}",
        json={"decision": "reviewed"},
    )
    notified = client.post(f"/api/v1/offenses/{case_id}/notify")

    assert blocked.status_code == 409
    assert reviewed.status_code == 200
    assert reviewed.json()["offense"]["review_status"] == "reviewed"
    assert notified.status_code == 200
    assert notified.json()["simulation"] is True
