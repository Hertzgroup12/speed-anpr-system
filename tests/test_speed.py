"""Tests for the calibrated speed calculation used by the video analyzer."""

import pytest

from app.speed import crossed_line, estimate_speed_kmh, travel_direction


@pytest.mark.parametrize(
    ("previous_y", "current_y", "line_y", "expected"),
    [
        (10, 20, 15, True),
        (20, 10, 15, True),
        (10, 14, 15, False),
        (15, 20, 15, True),
        (20, 15, 15, True),
    ],
)
def test_crossed_line(
    previous_y: float,
    current_y: float,
    line_y: float,
    expected: bool,
) -> None:
    assert crossed_line(previous_y, current_y, line_y) is expected


def test_estimate_speed_converts_meters_per_second_to_kmh() -> None:
    assert estimate_speed_kmh(distance_m=10, elapsed_seconds=2) == 18


@pytest.mark.parametrize(
    ("line_a_ratio", "line_b_ratio", "line_a_time", "line_b_time", "expected"),
    [
        (0.35, 0.65, 1, 2, "down"),
        (0.35, 0.65, 2, 1, "up"),
        (0.65, 0.35, 1, 2, "up"),
        (0.65, 0.35, 2, 1, "down"),
    ],
)
def test_travel_direction_uses_crossing_order(
    line_a_ratio: float,
    line_b_ratio: float,
    line_a_time: float,
    line_b_time: float,
    expected: str,
) -> None:
    assert travel_direction(
        line_a_ratio,
        line_b_ratio,
        line_a_time,
        line_b_time,
    ) == expected


@pytest.mark.parametrize(
    ("distance_m", "elapsed_seconds"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_estimate_speed_rejects_non_positive_measurements(
    distance_m: float,
    elapsed_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        estimate_speed_kmh(distance_m, elapsed_seconds)
