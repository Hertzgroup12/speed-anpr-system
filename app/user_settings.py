"""Per-account dashboard preferences."""

import threading
from typing import Any

from app.accounts import user_id_for_username
from app.config import Settings
from app.firebase_store import _firestore_client

PREFERENCES_COLLECTION = "operator_preferences"
_PREFERENCES: dict[str, dict[str, Any]] = {}
_PREFERENCES_LOCK = threading.Lock()


def get_user_settings(username: str, settings: Settings) -> dict[str, Any]:
    """Return account preferences overlaid on the server's configured defaults."""
    owner_id = user_id_for_username(username)
    if settings.firebase_enabled:
        reference = (
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            )
            .collection(PREFERENCES_COLLECTION)
            .document(owner_id)
        )
        snapshot = reference.get()
        preferences = snapshot.to_dict() if snapshot.exists else {}
    else:
        with _PREFERENCES_LOCK:
            preferences = _PREFERENCES.get(owner_id, {}).copy()

    return {
        "speed_limit_kmh": preferences.get(
            "speed_limit_kmh",
            settings.speed_limit_kmh,
        ),
        "camera_location": preferences.get(
            "camera_location",
            settings.camera_location,
        ),
    }


def save_user_settings(
    username: str,
    preferences: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Persist supplied preference fields without changing other user values."""
    owner_id = user_id_for_username(username)
    if settings.firebase_enabled:
        reference = (
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            )
            .collection(PREFERENCES_COLLECTION)
            .document(owner_id)
        )
        reference.set(preferences, merge=True)
    else:
        with _PREFERENCES_LOCK:
            _PREFERENCES.setdefault(owner_id, {}).update(preferences)

    return get_user_settings(username, settings)
