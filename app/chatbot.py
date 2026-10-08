"""Server-side Gemini chat requests; API keys never leave the backend."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import Settings

MAX_MESSAGE_LENGTH = 2000
SYSTEM_INSTRUCTION = (
    "You are the JNRD PRO assistant. Help operators understand the software, "
    "its configuration, and general speed-estimation workflow. Measurements "
    "are estimates, not legal findings. Never claim a reading proves an "
    "offence. Do not request, infer, or repeat personal data or license-plate "
    "numbers. You cannot access the user's camera, cases, database, or system "
    "settings."
)


def ask_gemini(message: str, settings: Settings) -> str:
    """Send one bounded, stateless user message to the configured Gemini model."""
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini chat is not configured")

    payload = json.dumps(
        {
            "model": settings.gemini_model,
            "input": message,
            "system_instruction": SYSTEM_INSTRUCTION,
            "store": False,
            "generation_config": {"thinking_level": "low"},
        }
    ).encode("utf-8")
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except HTTPError as error:
        if error.code == 429:
            raise RuntimeError("Gemini is rate limiting requests; try again shortly") from error
        raise RuntimeError("Gemini could not complete the request") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError("Gemini is temporarily unavailable") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Gemini returned an invalid response") from error

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an invalid response")
    output_text = result.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = result.get("output", [])
    if not isinstance(output, list):
        raise RuntimeError("Gemini returned an invalid response")
    text_parts = [
        item["text"]
        for item in output
        if isinstance(item, dict)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    if not text_parts:
        steps = result.get("steps", [])
        if not isinstance(steps, list):
            raise RuntimeError("Gemini returned an invalid response")
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content", [])
            if not isinstance(content, list):
                raise RuntimeError("Gemini returned an invalid response")
            text_parts.extend(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
    answer = "\n".join(text_parts).strip()
    if not answer:
        raise RuntimeError("Gemini returned no text response")
    return answer
