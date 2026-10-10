"""Tests for calendar-period speed-case PDF reports."""

from datetime import date

from app.reports import (
    build_case_report_pdf,
    calendar_period_bounds,
    cases_in_calendar_period,
)


def test_calendar_period_bounds_use_gmt_calendar_boundaries() -> None:
    today = date(2026, 10, 7)

    assert calendar_period_bounds("weekly", today) == (
        date(2026, 10, 5),
        date(2026, 10, 12),
    )
    assert calendar_period_bounds("monthly", today) == (
        date(2026, 10, 1),
        date(2026, 11, 1),
    )
    assert calendar_period_bounds("yearly", today) == (
        date(2026, 1, 1),
        date(2027, 1, 1),
    )


def test_cases_are_filtered_by_detection_date_and_timezone() -> None:
    cases = [
        {"case_id": "inside", "detected_at": "2026-10-05T00:00:00+00:00"},
        {"case_id": "outside", "detected_at": "2026-10-12T00:00:00+00:00"},
        {"case_id": "gmt-previous-day", "detected_at": "2026-10-05T00:30:00+02:00"},
        {"case_id": "invalid", "detected_at": "not-a-timestamp"},
    ]

    selected, start, end = cases_in_calendar_period(
        cases,
        "weekly",
        date(2026, 10, 7),
    )

    assert [case["case_id"] for case in selected] == ["inside"]
    assert start == date(2026, 10, 5)
    assert end == date(2026, 10, 12)


def test_pdf_report_is_valid_and_includes_a_case() -> None:
    cases = [
        {
            "case_id": "case-123",
            "plate_number": "ABC1234",
            "speed_kmh": 61,
            "speed_limit_kmh": 50,
            "location": "Main road",
            "detected_at": "2026-10-07T12:00:00+00:00",
            "review_status": "pending_review",
            "notification_status": "not_sent",
        }
    ]

    pdf = build_case_report_pdf(
        cases,
        "weekly",
        date(2026, 10, 5),
        date(2026, 10, 12),
    )

    assert pdf.startswith(b"%PDF-")
    assert b"%%EOF" in pdf
