"""Optional Firestore persistence for completed vehicle speed events."""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.config import Settings


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
                "video_name": video_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return len(events)
