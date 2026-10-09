"""Tests for persistent authentication audit records."""

from typing import Any

from app import accounts, firebase_store
from app.config import Settings
from app.time_utils import GMT, gmt_now_iso


class FakeDocument:
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name

    def document(self, identifier: str | None = None) -> FakeDocument:
        return FakeDocument(identifier or f"{self.name}-generated")


class FakeBatch:
    def __init__(self) -> None:
        self.operations: list[tuple[str, FakeDocument, dict[str, Any]]] = []
        self.committed = False

    def create(self, reference: FakeDocument, data: dict[str, Any]) -> None:
        self.operations.append(("create", reference, data))

    def set(self, reference: FakeDocument, data: dict[str, Any]) -> None:
        self.operations.append(("set", reference, data))

    def commit(self) -> None:
        self.committed = True


class FakeFirestore:
    def __init__(self) -> None:
        self.batch_instance = FakeBatch()

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(name)

    def batch(self) -> FakeBatch:
        return self.batch_instance


def test_registration_commits_account_and_gmt_signup_event_together(
    monkeypatch,
) -> None:
    client = FakeFirestore()
    monkeypatch.setattr(accounts, "_firestore_client", lambda *_args: client)
    settings = Settings(firebase_enabled=True, _env_file=None)

    accounts.register_user("New.User@example.com", "a-long-test-password", settings)

    assert client.batch_instance.committed
    create, signup = client.batch_instance.operations
    assert create[0] == "create"
    assert create[1].identifier.startswith(accounts.USER_COLLECTION)
    assert create[2]["username"] == "new.user@example.com"
    assert "a-long-test-password" not in create[2].values()
    assert signup[0] == "set"
    assert signup[1].identifier.startswith(settings.auth_audit_collection)
    assert signup[2]["username"] == "new.user@example.com"
    assert signup[2]["action"] == "signup"
    assert signup[2]["auth_method"] == "invite_code"
    assert signup[2]["timestamp_gmt"].endswith(" GMT")
    assert signup[2]["created_at"].endswith("+00:00")


def test_login_audit_record_is_written_to_configured_collection(
    monkeypatch,
) -> None:
    class RecordingDocument(FakeDocument):
        def set(self, data: dict[str, Any]) -> None:
            writes.append((self.identifier, data))

    class RecordingCollection(FakeCollection):
        def document(self, identifier: str | None = None) -> RecordingDocument:
            return RecordingDocument(identifier or f"{self.name}-generated")

    class RecordingClient(FakeFirestore):
        def collection(self, name: str) -> RecordingCollection:
            return RecordingCollection(name)

    writes: list[tuple[str, dict[str, Any]]] = []
    client = RecordingClient()
    monkeypatch.setattr(firebase_store, "_firestore_client", lambda *_args: client)
    settings = Settings(
        firebase_enabled=True,
        auth_audit_collection="auth_history",
        _env_file=None,
    )

    firebase_store.record_auth_event(
        "Operator@Example.com",
        "login",
        "configured_operator",
        settings,
    )

    assert writes[0][0].startswith("auth_history")
    assert writes[0][1]["username"] == "operator@example.com"
    assert writes[0][1]["action"] == "login"
    assert writes[0][1]["auth_method"] == "configured_operator"
    assert writes[0][1]["timestamp_gmt"].endswith(" GMT")
    assert writes[0][1]["created_at"].endswith("+00:00")


def test_iso_timestamps_keep_the_gmt_zero_offset() -> None:
    timestamp = gmt_now_iso()

    assert timestamp.endswith("+00:00")
    assert GMT.tzname(None) == "GMT"
