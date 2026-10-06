"""Firestore-backed invite-only accounts for dashboard operators."""

import hashlib
import re
import secrets
from datetime import datetime, timezone

from google.api_core.exceptions import AlreadyExists

from app.config import Settings
from app.firebase_store import _firestore_client

PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s.]+(?:\.[^@\s.]+)+$")
USER_COLLECTION = "operator_users"


def normalize_username(username: str) -> str:
    """Normalize email usernames so account identity is case-insensitive."""
    return username.strip().casefold()


def valid_signup_username(username: str) -> bool:
    """Accept email-shaped sign-in IDs and reject malformed addresses."""
    normalized = normalize_username(username)
    return len(normalized) <= 254 and EMAIL_PATTERN.fullmatch(normalized) is not None


def invite_code_matches(candidate: str, settings: Settings) -> bool:
    """Constant-time compare the shared registration invitation code."""
    configured_code = settings.admin_invite_code
    return bool(
        configured_code
        and len(configured_code) >= 20
        and secrets.compare_digest(candidate, configured_code)
    )


def _user_reference(username: str, settings: Settings):
    """Return a deterministic, non-email Firestore document reference."""
    normalized = normalize_username(username)
    document_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    client = _firestore_client(
        settings.firebase_project_id,
        settings.firebase_credentials,
    )
    return client.collection(USER_COLLECTION).document(document_id)


def register_user(username: str, password: str, settings: Settings) -> None:
    """Atomically create an operator account with a salted PBKDF2 password hash."""
    if not settings.firebase_enabled:
        raise RuntimeError("New account registration requires Firebase to be enabled")
    if not valid_signup_username(username):
        raise ValueError("Enter a valid email address")
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters")

    salt = secrets.token_bytes(32)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    reference = _user_reference(username, settings)
    try:
        reference.create(
            {
                "username": normalize_username(username),
                "password_salt": salt.hex(),
                "password_hash": password_hash.hex(),
                "password_iterations": PASSWORD_ITERATIONS,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except AlreadyExists as error:
        raise FileExistsError("An account with this email already exists") from error


def authenticate_registered_user(
    username: str,
    password: str,
    settings: Settings,
) -> bool:
    """Verify a Firestore user password using constant-time hash comparison."""
    if not settings.firebase_enabled or not valid_signup_username(username):
        return False

    snapshot = _user_reference(username, settings).get()
    if not snapshot.exists:
        return False

    user = snapshot.to_dict()
    try:
        salt = bytes.fromhex(user["password_salt"])
        expected_hash = bytes.fromhex(user["password_hash"])
        iterations = user["password_iterations"]
        if (
            not isinstance(iterations, int)
            or iterations < PASSWORD_ITERATIONS
            or iterations > 2_000_000
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return secrets.compare_digest(actual_hash, expected_hash)
