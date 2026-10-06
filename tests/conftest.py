"""Isolate the operator authentication environment used by tests."""

from collections.abc import Iterator
import pytest

from app.config import get_settings

TEST_USERNAME = "test-operator"
TEST_PASSWORD = "test-password-123"
TEST_SECRET = "test-session-secret-that-is-long-enough"


@pytest.fixture(autouse=True)
def operator_auth_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("APP_USERNAME", TEST_USERNAME)
    monkeypatch.setenv("APP_PASSWORD", TEST_PASSWORD)
    monkeypatch.setenv("APP_SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("AUTH_COOKIE_SECURE", "false")
    monkeypatch.setenv("FIREBASE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
