"""Minimal OpenRouter connectivity check using the configured environment variables."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx.utils.env import load_dotenv

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.2"
DEFAULT_MAX_TOKENS = 128


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
    site_name = os.environ.get("OPENROUTER_SITE_NAME", "").strip()
    max_tokens = coerce_max_tokens(os.environ.get("OPENROUTER_MAX_TOKENS"), default=DEFAULT_MAX_TOKENS)

    if not api_key or api_key.lower() == "your_openrouter_api_key_here":
        print("OpenRouter check failed: OPENROUTER_API_KEY is not configured.")
        return 2

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-OpenRouter-Title"] = site_name

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: OPENROUTER_OK",
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(f"OpenRouter check failed: HTTP {exc.code}")
        print(details)
        return 1
    except urllib.error.URLError as exc:
        print(f"OpenRouter check failed: network error: {exc.reason}")
        return 1

    parsed = json.loads(body)
    text = extract_output_text(parsed)
    print("OpenRouter check succeeded.")
    print(f"Model: {parsed.get('model', model)}")
    print(f"Output: {text or '<empty>'}")
    return 0


def extract_output_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip()


def coerce_max_tokens(raw_value: str | None, default: int) -> int:
    try:
        value = int(str(raw_value).strip()) if raw_value is not None and str(raw_value).strip() else default
    except (TypeError, ValueError):
        return default
    return max(1, value)


if __name__ == "__main__":
    raise SystemExit(main())
