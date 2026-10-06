"""Signed, short-lived browser sessions for the single-operator dashboard."""

import base64
import binascii
import hashlib
import hmac
import secrets
import time

from app.config import Settings

SESSION_COOKIE = "jnrd_session"
SESSION_MAX_AGE_SECONDS = 8 * 60 * 60


def auth_is_configured(settings: Settings) -> bool:
    """Require explicit credentials and a strong signing key; fail closed."""
    return bool(
        settings.app_username
        and settings.app_password
        and len(settings.app_password) >= 12
        and settings.app_secret_key
        and len(settings.app_secret_key) >= 32
    )


def credentials_match(
    username: str,
    password: str,
    settings: Settings,
) -> bool:
    """Compare credentials without leaking whether either value matched."""
    if not auth_is_configured(settings):
        return False
    return secrets.compare_digest(username, settings.app_username or "") and (
        secrets.compare_digest(password, settings.app_password or "")
    )


def create_session_token(settings: Settings) -> str:
    """Create an authenticated cookie token that expires after eight hours."""
    if not auth_is_configured(settings):
        raise ValueError("Authentication is not configured")
    expires_at = int(time.time()) + SESSION_MAX_AGE_SECONDS
    payload = base64.urlsafe_b64encode(
        f"{settings.app_username}\n{expires_at}".encode("utf-8")
    ).rstrip(b"=")
    signature = hmac.new(
        (settings.app_secret_key or "").encode("utf-8"),
        payload,
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{payload.decode('ascii')}.{encoded_signature.decode('ascii')}"


def verify_session_token(token: str | None, settings: Settings) -> bool:
    """Validate the signature, configured user, and expiry of a session."""
    if not token or not auth_is_configured(settings):
        return False
    try:
        encoded_payload, encoded_signature = token.split(".", maxsplit=1)
        payload = encoded_payload.encode("ascii")
        supplied_signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected_signature = hmac.new(
            (settings.app_secret_key or "").encode("utf-8"),
            payload,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return False

        decoded_payload = base64.urlsafe_b64decode(
            encoded_payload + "=" * (-len(encoded_payload) % 4)
        ).decode("utf-8")
        username, expires_at = decoded_payload.rsplit("\n", maxsplit=1)
        return (
            secrets.compare_digest(username, settings.app_username or "")
            and int(expires_at) > int(time.time())
        )
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
