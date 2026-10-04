"""Small, dependency-free helpers for calibrated speed measurement."""


def crossed_line(previous_y: float, current_y: float, line_y: float) -> bool:
    """Return true when a tracked point moves across a horizontal image line."""
    return (previous_y < line_y <= current_y) or (current_y <= line_y < previous_y)


def estimate_speed_kmh(distance_m: float, elapsed_seconds: float) -> float:
    """Convert distance and elapsed time into kilometers per hour."""
    if distance_m <= 0:
        raise ValueError("distance_m must be greater than zero")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be greater than zero")
    return (distance_m / elapsed_seconds) * 3.6


def travel_direction(
    line_a_ratio: float,
    line_b_ratio: float,
    line_a_time: float,
    line_b_time: float,
) -> str:
    """Infer image-plane direction from line positions and crossing order."""
    moving_down = (line_b_ratio > line_a_ratio) == (line_b_time > line_a_time)
    return "down" if moving_down else "up"
