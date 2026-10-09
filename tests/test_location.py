from app.location import extract_video_location


def test_extract_video_location_reads_iso_6709_coordinates(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"ftypisom\x00\x00\x00\x00+37.421998-122.084000+12.5/")

    assert extract_video_location(video_path) == {
        "latitude": 37.421998,
        "longitude": -122.084,
        "altitude_m": 12.5,
    }


def test_extract_video_location_handles_coordinates_across_read_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    import app.location as location

    monkeypatch.setattr(location, "READ_CHUNK_SIZE", 16)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"x" * 20 + b"+37.421998-122.084000/")

    assert extract_video_location(video_path) == {
        "latitude": 37.421998,
        "longitude": -122.084,
    }


def test_extract_video_location_ignores_invalid_coordinates(tmp_path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"+99.000000-200.000000/")

    assert extract_video_location(video_path) is None
