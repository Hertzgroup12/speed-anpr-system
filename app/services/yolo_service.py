def estimate_speed(distance_m: float, time_seconds: float, vehicle_length_m: float | None = None) -> float:
    if distance_m <= 0:
        raise ValueError("distance_m must be greater than zero")
    if time_seconds <= 0:
        raise ValueError("time_seconds must be greater than zero")

    speed_mps = distance_m / time_seconds
    speed_kmh = speed_mps * 3.6

    if vehicle_length_m is not None and vehicle_length_m > 0:
        calibration_factor = max(1.0, vehicle_length_m / 4.5)
        speed_kmh *= calibration_factor

    return round(speed_kmh, 2)
