"""Timestamp helpers using GMT (the zero-offset equivalent of UTC)."""

from datetime import datetime, timedelta, timezone

GMT = timezone(timedelta(0), "GMT")


def gmt_now_iso() -> str:
    """Return the current GMT time in ISO 8601 format."""
    return datetime.now(GMT).isoformat()


def gmt_now_label() -> str:
    """Return the current GMT time with an explicit human-readable label."""
    return datetime.now(GMT).strftime("%Y-%m-%d %H:%M:%S GMT")
