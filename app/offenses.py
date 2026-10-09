"""Speed-case review, optional Firestore storage, and manual driver notifications."""

import json
import threading
import uuid
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.accounts import user_id_for_username
from app.config import Settings
from app.firebase_store import _firestore_client
from app.time_utils import gmt_now_iso

CaseDecision = Literal["reviewed", "dismissed"]
_CASES: dict[str, dict[str, Any]] = {}
_CASES_LOCK = threading.Lock()


def _default_owner_id(settings: Settings) -> str:
    return user_id_for_username(settings.app_username or "operator")


def _case_belongs_to_user(
    case: dict[str, Any],
    owner_id: str,
    settings: Settings,
) -> bool:
    case_owner = case.get("owner_id")
    if case_owner is None:
        return owner_id == _default_owner_id(settings)
    return case_owner == owner_id


def create_offenses(
    events: list[dict[str, Any]],
    video_name: str,
    location: str,
    detected_at: str,
    settings: Settings,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Create review cases for estimates above the configured speed limit."""
    owner_id = owner_id or _default_owner_id(settings)
    cases: list[dict[str, Any]] = []
    for event in events:
        if event["speed_kmh"] <= settings.speed_limit_kmh:
            continue

        case = {
            "case_id": uuid.uuid4().hex,
            "track_id": event["track_id"],
            "plate_number": event["plate_number"],
            "plate_confidence": event["plate_confidence"],
            "speed_kmh": event["speed_kmh"],
            "speed_limit_kmh": settings.speed_limit_kmh,
            "excess_kmh": round(event["speed_kmh"] - settings.speed_limit_kmh, 2),
            "location": location,
            "detected_at": detected_at,
            "video_timestamp_seconds": event["timestamp_seconds"],
            "vehicle_capture_id": event["vehicle_capture_id"],
            "plate_capture_id": event["plate_capture_id"],
            "video_name": video_name,
            "review_status": "pending_review",
            "notification_status": "not_sent",
            "created_at": gmt_now_iso(),
            "owner_id": owner_id,
        }
        with _CASES_LOCK:
            _CASES[case["case_id"]] = case

        if settings.firebase_enabled:
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            ).collection(settings.offense_collection).document(case["case_id"]).set(case)
        cases.append(case.copy())
    return cases


def list_offenses(
    settings: Settings,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the current user's saved cases, newest first."""
    owner_id = owner_id or _default_owner_id(settings)
    if settings.firebase_enabled:
        documents = (
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            )
            .collection(settings.offense_collection)
            .stream()
        )
        cases = [
            case
            for document in documents
            if (case := document.to_dict()) is not None
            and _case_belongs_to_user(case, owner_id, settings)
        ]
    else:
        with _CASES_LOCK:
            cases = [
                case.copy()
                for case in _CASES.values()
                if _case_belongs_to_user(case, owner_id, settings)
            ]
    return sorted(cases, key=lambda case: case["created_at"], reverse=True)


def update_case_decision(
    case_id: str,
    decision: CaseDecision,
    settings: Settings,
    owner_id: str | None = None,
) -> dict[str, Any] | None:
    """Record a human review decision without changing the measured estimate."""
    owner_id = owner_id or _default_owner_id(settings)
    if settings.firebase_enabled:
        reference = (
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            )
            .collection(settings.offense_collection)
            .document(case_id)
        )
        snapshot = reference.get()
        if not snapshot.exists:
            return None
        case = snapshot.to_dict()
        if case is None or not _case_belongs_to_user(case, owner_id, settings):
            return None
        reviewed_at = gmt_now_iso()
        reference.update(
            {
                "review_status": decision,
                "reviewed_at": reviewed_at,
            }
        )
        case.update(
            {
                "review_status": decision,
                "reviewed_at": reviewed_at,
            }
        )
        return case

    with _CASES_LOCK:
        case = _CASES.get(case_id)
        if case is None or not _case_belongs_to_user(case, owner_id, settings):
            return None
        case["review_status"] = decision
        case["reviewed_at"] = gmt_now_iso()
        return case.copy()


def notify_driver(
    case_id: str,
    settings: Settings,
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Send a reviewed case alert, or record a simulation without contacting anyone."""
    owner_id = owner_id or _default_owner_id(settings)
    if settings.firebase_enabled:
        cases = list_offenses(settings, owner_id)
        case = next((item for item in cases if item["case_id"] == case_id), None)
    else:
        with _CASES_LOCK:
            saved_case = _CASES.get(case_id)
            case = (
                saved_case.copy()
                if saved_case
                and _case_belongs_to_user(saved_case, owner_id, settings)
                else None
            )

    if case is None:
        raise KeyError("Offense case was not found")
    if case["review_status"] != "reviewed":
        raise ValueError("A case must be reviewed before notifying a driver")
    if case["notification_status"] in {"simulated", "sent"}:
        raise ValueError("This case already has a recorded notification")
    plate_number = case.get("plate_number")
    phone_number = settings.driver_directory.get(plate_number or "")
    if not phone_number:
        raise LookupError("No driver phone number is configured for this plate")

    message = (
        f"Speed alert: vehicle {plate_number} was measured at "
        f"{case['speed_kmh']} km/h in a {case['speed_limit_kmh']} km/h zone "
        f"at {case['location']} on {case['detected_at']}. "
        "This is a review notice, not a legal determination."
    )
    if settings.sms_simulation:
        status = "simulated"
    else:
        status = _send_africastalking_sms(phone_number, message, settings)

    update = {
        "notification_status": status,
        "notification_at": gmt_now_iso(),
    }
    if settings.firebase_enabled:
        (
            _firestore_client(
                settings.firebase_project_id,
                settings.firebase_credentials,
            )
            .collection(settings.offense_collection)
            .document(case_id)
            .update(update)
        )
    else:
        with _CASES_LOCK:
            saved_case = _CASES.get(case_id)
            if saved_case is None or not _case_belongs_to_user(
                saved_case,
                owner_id,
                settings,
            ):
                raise KeyError("Offense case was not found")
            saved_case.update(update)

    return {
        "case_id": case_id,
        "notification_status": status,
        "simulation": settings.sms_simulation,
    }


def capture_belongs_to_user(
    capture_id: str,
    settings: Settings,
    owner_id: str,
) -> bool:
    """Check that a capture is referenced by one of this user's review cases."""
    return any(
        capture_id
        in {
            case.get("vehicle_capture_id"),
            case.get("plate_capture_id"),
        }
        for case in list_offenses(settings, owner_id)
    )


def _send_africastalking_sms(
    phone_number: str,
    message: str,
    settings: Settings,
) -> str:
    """Send one explicitly requested SMS through the Africa's Talking API."""
    if not settings.africastalking_username or not settings.africastalking_api_key:
        raise ValueError(
            "Configure Africa's Talking credentials or enable SMS_SIMULATION"
        )

    form_data = {
        "username": settings.africastalking_username,
        "to": phone_number,
        "message": message,
    }
    if settings.africastalking_sender_id:
        form_data["from"] = settings.africastalking_sender_id
    request = Request(
        "https://api.africastalking.com/version1/messaging",
        data=urlencode(form_data).encode("utf-8"),
        headers={
            "apiKey": settings.africastalking_api_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Africa's Talking request failed: {error}") from error

    recipients = result.get("SMSMessageData", {}).get("Recipients", [])
    if not recipients or any(
        recipient.get("status") != "Success" for recipient in recipients
    ):
        raise RuntimeError("Africa's Talking did not confirm successful SMS delivery")
    return "sent"
