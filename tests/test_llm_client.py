from pathlib import Path
import sys
import urllib.error
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx.core.llm_client import GroqClient
from ctx.core.llm_client import GeminiClient
from ctx.core.llm_client import LLMClient
from ctx.core.llm_client import OpenRouterClient


def test_groq_extracts_output_text_from_chat_completion_payload() -> None:
    parsed = {
        "id": "chatcmpl_123",
        "object": "chat.completion",
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Fast models reduce latency.",
                },
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 6,
            "total_tokens": 16,
        },
    }

    assert GroqClient._extract_chat_completion_text(parsed) == "Fast models reduce latency."


def test_groq_stops_retrying_other_models_after_403() -> None:
    client = GroqClient(api_key="test-key")
    response = Mock()
    response.read.return_value = b"error code: 1010"
    response.close = Mock()
    error = urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=403,
        msg="Forbidden",
        hdrs=None,
        fp=response,
    )

    with patch("urllib.request.urlopen", side_effect=[error]) as mocked_urlopen:
        try:
            client.generate(prompt="hello")
        except RuntimeError as exc:
            assert "llama-3.3-70b-versatile: HTTP 403: error code: 1010" in str(exc)
        else:
            raise AssertionError("Expected Groq runtime error")

    assert mocked_urlopen.call_count == 1


def test_gemini_stops_retrying_other_models_after_429() -> None:
    client = GeminiClient(api_key="test-key")
    response = Mock()
    response.read.return_value = b'{"error":{"code":429,"message":"quota exceeded"}}'
    response.close = Mock()
    error = urllib.error.HTTPError(
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=response,
    )

    with patch("urllib.request.urlopen", side_effect=[error]) as mocked_urlopen:
        try:
            client.generate(prompt="hello")
        except RuntimeError as exc:
            assert "gemini-2.5-flash: HTTP 429:" in str(exc)
        else:
            raise AssertionError("Expected Gemini runtime error")

    assert mocked_urlopen.call_count == 1


def test_openrouter_extracts_output_text_from_chat_completion_payload() -> None:
    parsed = {
        "id": "gen-123",
        "model": "openai/gpt-5.2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "OpenRouter is working.",
                },
            }
        ],
    }

    assert GroqClient._extract_chat_completion_text(parsed) == "OpenRouter is working."


def test_openrouter_stops_retrying_other_models_after_401() -> None:
    client = OpenRouterClient(api_key="test-key")
    response = Mock()
    response.read.return_value = b'{"error":{"message":"invalid api key"}}'
    response.close = Mock()
    error = urllib.error.HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=response,
    )

    with patch("urllib.request.urlopen", side_effect=[error]) as mocked_urlopen:
        try:
            client.generate(prompt="hello")
        except RuntimeError as exc:
            assert "openai/gpt-5.2: HTTP 401:" in str(exc)
        else:
            raise AssertionError("Expected OpenRouter runtime error")

    assert mocked_urlopen.call_count == 1


def test_provider_order_supports_openrouter() -> None:
    with patch.dict("os.environ", {"CTX_LLM_PROVIDER_ORDER": "openrouter,gemini,groq"}, clear=False):
        assert LLMClient._resolve_provider_order() == ("openrouter", "gemini", "groq")


def test_openrouter_uses_configurable_max_tokens() -> None:
    with patch.dict("os.environ", {"OPENROUTER_MAX_TOKENS": "256"}, clear=False):
        client = OpenRouterClient(api_key="test-key")

    assert client._max_tokens == 256
