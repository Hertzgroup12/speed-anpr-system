"""Read ISO 6709 GPS coordinates embedded in common video metadata."""

import re
from pathlib import Path

ISO_6709_PATTERN = re.compile(
    rb"([+-]\d{2}(?:\.\d+)?)([+-]\d{3}(?:\.\d+)?)(?:([+-]\d+(?:\.\d+)?))?/"
)
READ_CHUNK_SIZE = 1024 * 1024
PATTERN_OVERLAP = 64


def extract_video_location(
    video_path: str | Path,
) -> dict[str, float] | None:
    """Return embedded ISO 6709 latitude/longitude and optional altitude."""
    carry = b""
    with Path(video_path).open("rb") as video_file:
        while chunk := video_file.read(READ_CHUNK_SIZE):
            data = carry + chunk
            for match in ISO_6709_PATTERN.finditer(data):
                latitude = float(match.group(1))
                longitude = float(match.group(2))
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    continue

                location = {
                    "latitude": latitude,
                    "longitude": longitude,
                }
                if match.group(3) is not None:
                    location["altitude_m"] = float(match.group(3))
                return location
            carry = data[-PATTERN_OVERLAP:]
    return None
