"""Minimal Groq connectivity check using the configured environment variables."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx.utils.env import load_dotenv

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.environ.get("GROQ_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    if not api_key or api_key.lower() == "your_groq_api_key_here":
        print("Groq check failed: GROQ_API_KEY is not configured.")
        return 2

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: GROQ_OK",
            }
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        print(f"Groq check failed: HTTP {exc.code}")
        print(details)
        return 1
    except urllib.error.URLError as exc:
        print(f"Groq check failed: network error: {exc.reason}")
        return 1

    parsed = json.loads(body)
    text = extract_output_text(parsed)
    print("Groq check succeeded.")
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


if __name__ == "__main__":
    raise SystemExit(main())
