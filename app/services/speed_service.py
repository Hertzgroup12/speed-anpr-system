from __future__ import annotations

import os
from typing import Any

try:
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import db
except ImportError:  # pragma: no cover
    firebase_admin = None
    credentials = None
    db = None


def initialize_firebase() -> bool:
    if firebase_admin is None:
        return False

    if not firebase_admin._apps:
        credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        database_url = os.getenv("FIREBASE_DATABASE_URL")

        if not credentials_path or not database_url:
            return False

        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})

    return True


def save_violation_record(record: dict[str, Any]) -> dict[str, Any]:
    if db is None:
        return {"status": "simulated", "record": record}

    ref = db.reference("speed_anpr_records")
    new_record = ref.push(record)
    return {"status": "saved", "record_id": new_record.key}
