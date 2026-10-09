"""Optional Firestore persistence for completed vehicle speed events."""

from functools import lru_cache
from typing import Any

from app.config import Settings
from app.time_utils import gmt_now_iso, gmt_now_label


def _auth_event_data(
    username: str,
    action: str,
    auth_method: str,
) -> dict[str, str]:
    """Build an authentication audit record without credentials or tokens."""
    return {
        "username": username.strip().casefold(),
        "action": action,
        "auth_method": auth_method,
        "timestamp_gmt": gmt_now_label(),
        "created_at": gmt_now_iso(),
    }


@lru_cache
def _firestore_client(project_id: str | None, credentials_path: str | None) -> Any:
    """Create the Firestore client once, using a service account or ADC."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    if credentials_path:
        credential = credentials.Certificate(credentials_path)
        options = {"projectId": project_id} if project_id else None
        app = firebase_admin.initialize_app(credential, options=options)
    else:
        options = {"projectId": project_id} if project_id else None
        app = firebase_admin.initialize_app(options=options)
    return firestore.client(app=app)


def persist_events(
    events: list[dict[str, Any]],
    video_name: str,
    settings: Settings,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Write each analyzed event to Firestore and return the number saved."""
    if not settings.firebase_enabled or not events:
        return 0

    client = _firestore_client(
        settings.firebase_project_id,
        settings.firebase_credentials,
    )
    collection = client.collection(settings.firebase_collection)
    for event in events:
        collection.document().set(
            {
                **event,
                **(metadata or {}),
                "video_name": video_name,
                "created_at": gmt_now_iso(),
            }
        )
    return len(events)


def record_auth_event(
    username: str,
    action: str,
    auth_method: str,
    settings: Settings,
) -> None:
    """Persist a successful authentication event when Firestore is enabled."""
    if not settings.firebase_enabled:
        return

    client = _firestore_client(
        settings.firebase_project_id,
        settings.firebase_credentials,
    )
    client.collection(settings.auth_audit_collection).document().set(
        _auth_event_data(username, action, auth_method)
    )
