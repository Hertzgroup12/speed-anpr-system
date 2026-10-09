"""Tests for signed operator session cookies."""

from app.auth import create_session_token, session_username, verify_session_token
from app.config import Settings


def test_signed_session_is_valid_only_for_matching_secret_and_user() -> None:
    settings = Settings(
        app_username="operator",
        app_password="long-test-password",
        app_secret_key="test-session-secret-that-is-long-enough",
    )
    token = create_session_token(settings)

    assert verify_session_token(token, settings)
    assert session_username(token, settings) == "operator"
    assert not verify_session_token(
        token,
        settings.model_copy(update={"app_secret_key": "different-secret-key-long-enough"}),
    )
    assert not verify_session_token(
        token,
        settings.model_copy(update={"app_username": "different-operator"}),
    )


def test_session_rejects_tampered_malformed_and_missing_tokens() -> None:
    settings = Settings(
        app_username="operator",
        app_password="long-test-password",
        app_secret_key="test-session-secret-that-is-long-enough",
    )
    token = create_session_token(settings)

    assert not verify_session_token(f"{token}tampered", settings)
    assert not verify_session_token("not-a-session-token", settings)
    assert not verify_session_token("%%% .%%%", settings)
    assert not verify_session_token(None, settings)


def test_session_rejects_expired_tokens(monkeypatch) -> None:
    import app.auth as auth

    settings = Settings(
        app_username="operator",
        app_password="long-test-password",
        app_secret_key="test-session-secret-that-is-long-enough",
    )
    monkeypatch.setattr(auth.time, "time", lambda: 1_000)
    token = create_session_token(settings)
    monkeypatch.setattr(auth.time, "time", lambda: 1_000 + auth.SESSION_MAX_AGE_SECONDS)

    assert not verify_session_token(token, settings)


def test_registered_user_session_is_valid_for_non_operator_username() -> None:
    settings = Settings(
        app_username="operator",
        app_password="long-test-password",
        app_secret_key="test-session-secret-that-is-long-enough",
    )
    token = create_session_token(
        settings,
        "registered@example.com",
        registered_user=True,
    )

    assert verify_session_token(token, settings)
