"""Tests for the stateless Gemini chat request."""

import json
from io import BytesIO
from urllib.request import Request

import pytest

from app.chatbot import ask_gemini
from app.config import Settings


class FakeResponse(BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_gemini_request_keeps_api_key_server_side_and_disables_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            b'{"output":[{"type":"text","text":"Use the review panel."}]}'
        )

    monkeypatch.setattr("app.chatbot.urlopen", fake_urlopen)
    settings = Settings(gemini_api_key="private-test-key")

    answer = ask_gemini("How do I review a case?", settings)

    request = captured["request"]
    assert answer == "Use the review panel."
    assert isinstance(request, Request)
    assert request.get_header("X-goog-api-key") == "private-test-key"
    assert captured["timeout"] == 30
    assert json.loads(request.data or b"{}")["store"] is False
    assert "private-test-key" not in (request.data or b"").decode()


def test_gemini_requires_an_api_key() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        ask_gemini("Hello", Settings())


def test_gemini_rejects_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chatbot.urlopen",
        lambda *_args, **_kwargs: FakeResponse(b'{"output":[]}'),
    )

    with pytest.raises(RuntimeError, match="no text response"):
        ask_gemini("Hello", Settings(gemini_api_key="private-test-key"))


@pytest.mark.parametrize("body", [b"[]", b'{"output":{}}'])
def test_gemini_rejects_unexpected_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    monkeypatch.setattr(
        "app.chatbot.urlopen",
        lambda *_args, **_kwargs: FakeResponse(body),
    )

    with pytest.raises(RuntimeError, match="invalid response"):
        ask_gemini("Hello", Settings(gemini_api_key="private-test-key"))
