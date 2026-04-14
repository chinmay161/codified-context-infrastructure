"""LLM execution clients with Groq, OpenRouter, and Gemini providers."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..utils.env import load_dotenv

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "openai/gpt-5.2"
_OPENROUTER_DEFAULT_MAX_TOKENS = 1024
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(slots=True, frozen=True)
class LLMResult:
    """LLM completion payload plus metadata."""

    output: str
    metadata: dict[str, Any]


class LLMClient:
    """Provider-aware client with Groq, OpenRouter, and Gemini fallback."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        load_dotenv()
        self._timeout_seconds = timeout_seconds
        self._groq_client = GroqClient(timeout_seconds=timeout_seconds)
        self._openrouter_client = OpenRouterClient(timeout_seconds=timeout_seconds)
        self._gemini_client = GeminiClient(timeout_seconds=timeout_seconds)
        self._provider_order = self._resolve_provider_order()

    @property
    def is_configured(self) -> bool:
        return (
            self._groq_client.is_configured
            or self._openrouter_client.is_configured
            or self._gemini_client.is_configured
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using the configured provider order."""

        provider_errors: list[str] = []
        for provider_name in self._provider_order:
            if provider_name == "groq":
                if not self._groq_client.is_configured:
                    provider_errors.append("groq: not configured")
                    continue
                try:
                    return self._groq_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )
                except RuntimeError as exc:
                    provider_errors.append(f"groq: {exc}")
                    continue

            if provider_name == "gemini":
                if not self._gemini_client.is_configured:
                    provider_errors.append("gemini: not configured")
                    continue
                try:
                    return self._gemini_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )
                except RuntimeError as exc:
                    provider_errors.append(f"gemini: {exc}")
                    continue

            if provider_name == "openrouter":
                if not self._openrouter_client.is_configured:
                    provider_errors.append("openrouter: not configured")
                    continue
                try:
                    return self._openrouter_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )
                except RuntimeError as exc:
                    provider_errors.append(f"openrouter: {exc}")
                    continue

        raise RuntimeError("All configured LLM providers failed: " + " | ".join(provider_errors))

    @staticmethod
    def _resolve_provider_order() -> tuple[str, ...]:
        raw_value = os.environ.get("CTX_LLM_PROVIDER_ORDER", "groq,openrouter,gemini").strip().lower()
        ordered = [item.strip() for item in raw_value.split(",") if item.strip()]
        deduped: list[str] = []
        for item in ordered:
            normalized = "groq" if item in {"grok", "xai"} else item
            if normalized in {"groq", "openrouter", "gemini"} and normalized not in deduped:
                deduped.append(normalized)
        if not deduped:
            return ("groq", "openrouter", "gemini")
        return tuple(deduped)


class GroqClient:
    """Small stdlib-based client for Groq chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        load_dotenv()
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self._model = model or os.environ.get("GROQ_MODEL", _GROQ_DEFAULT_MODEL).strip() or _GROQ_DEFAULT_MODEL
        self._base_url = (base_url or os.environ.get("GROQ_BASE_URL", _GROQ_BASE_URL)).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._fallback_models = self._build_fallback_models(self._model)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and self._api_key.lower() != "your_groq_api_key_here"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using Groq's chat completions API."""

        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        errors: list[str] = []
        for candidate_model in self._fallback_models:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": candidate_model,
                "messages": messages,
                "temperature": temperature,
            }
            request = urllib.request.Request(
                url=f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                errors.append(f"{candidate_model}: HTTP {exc.code}: {details}")
                if _should_stop_after_http_error(exc.code):
                    break
                continue
            except urllib.error.URLError as exc:
                errors.append(f"{candidate_model}: {exc.reason}")
                continue

            parsed = json.loads(body)
            content = self._extract_chat_completion_text(parsed)
            if not content:
                errors.append(f"{candidate_model}: missing message content")
                continue

            usage = parsed.get("usage", {})
            return LLMResult(
                output=content,
                metadata={
                    "provider": "groq",
                    "model": parsed.get("model", candidate_model),
                    "usage": usage if isinstance(usage, dict) else {},
                    "attempted_models": list(self._fallback_models),
                },
            )

        raise RuntimeError("Groq API request failed for all models: " + " | ".join(errors))

    @staticmethod
    def _extract_chat_completion_text(parsed: dict[str, Any]) -> str:
        choices = parsed.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            return ""
        return str(message.get("content", "")).strip()

    @staticmethod
    def _build_fallback_models(primary_model: str) -> tuple[str, ...]:
        ordered = [
            primary_model,
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        deduped: list[str] = []
        for model in ordered:
            normalized = str(model).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return tuple(deduped)


class GeminiClient:
    """Small stdlib-based client for Gemini generateContent."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        load_dotenv()
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        self._model = model or os.environ.get("GEMINI_MODEL", _GEMINI_DEFAULT_MODEL).strip() or _GEMINI_DEFAULT_MODEL
        self._base_url = (base_url or os.environ.get("GEMINI_BASE_URL", _GEMINI_BASE_URL)).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._fallback_models = self._build_fallback_models(self._model)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and self._api_key.lower() != "your_gemini_api_key_here"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using Gemini generateContent."""

        if not self.is_configured:
            raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is not configured.")

        errors: list[str] = []
        for candidate_model in self._fallback_models:
            payload: dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                },
            }
            if system_prompt:
                payload["system_instruction"] = {
                    "parts": [{"text": system_prompt}],
                }

            encoded_model = urllib.parse.quote(candidate_model, safe="")
            request = urllib.request.Request(
                url=f"{self._base_url}/models/{encoded_model}:generateContent",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                errors.append(f"{candidate_model}: HTTP {exc.code}: {details}")
                if _should_stop_after_http_error(exc.code):
                    break
                continue
            except urllib.error.URLError as exc:
                errors.append(f"{candidate_model}: {exc.reason}")
                continue

            parsed = json.loads(body)
            candidates = parsed.get("candidates", [])
            if not candidates:
                errors.append(f"{candidate_model}: missing candidates")
                continue
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [str(part.get("text", "")) for part in parts if str(part.get("text", "")).strip()]
            content = "\n".join(text_parts).strip()
            if not content:
                errors.append(f"{candidate_model}: missing text content")
                continue

            usage = parsed.get("usageMetadata", {})
            return LLMResult(
                output=content,
                metadata={
                    "provider": "gemini",
                    "model": candidate_model,
                    "usage": usage if isinstance(usage, dict) else {},
                    "attempted_models": list(self._fallback_models),
                },
            )

        raise RuntimeError("Gemini API request failed for all models: " + " | ".join(errors))

    @staticmethod
    def _build_fallback_models(primary_model: str) -> tuple[str, ...]:
        ordered = [
            primary_model,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
        deduped: list[str] = []
        for model in ordered:
            normalized = str(model).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return tuple(deduped)


class OpenRouterClient:
    """Small stdlib-based client for OpenRouter chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        load_dotenv()
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._model = (
            model
            or os.environ.get("OPENROUTER_MODEL", _OPENROUTER_DEFAULT_MODEL).strip()
            or _OPENROUTER_DEFAULT_MODEL
        )
        self._base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL", _OPENROUTER_BASE_URL)).rstrip("/")
        self._site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
        self._site_name = os.environ.get("OPENROUTER_SITE_NAME", "").strip()
        self._max_tokens = _coerce_max_tokens(
            os.environ.get("OPENROUTER_MAX_TOKENS"),
            default=_OPENROUTER_DEFAULT_MAX_TOKENS,
        )
        self._timeout_seconds = timeout_seconds
        self._fallback_models = self._build_fallback_models(self._model)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and self._api_key.lower() != "your_openrouter_api_key_here"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using OpenRouter's chat completions API."""

        if not self.is_configured:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        errors: list[str] = []
        for candidate_model in self._fallback_models:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": candidate_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self._max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            if self._site_url:
                headers["HTTP-Referer"] = self._site_url
            if self._site_name:
                headers["X-OpenRouter-Title"] = self._site_name
            request = urllib.request.Request(
                url=f"{self._base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                    body = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                errors.append(f"{candidate_model}: HTTP {exc.code}: {details}")
                if _should_stop_after_http_error(exc.code):
                    break
                continue
            except urllib.error.URLError as exc:
                errors.append(f"{candidate_model}: {exc.reason}")
                continue

            parsed = json.loads(body)
            content = GroqClient._extract_chat_completion_text(parsed)
            if not content:
                errors.append(f"{candidate_model}: missing message content")
                continue

            usage = parsed.get("usage", {})
            return LLMResult(
                output=content,
                metadata={
                    "provider": "openrouter",
                    "model": parsed.get("model", candidate_model),
                    "usage": usage if isinstance(usage, dict) else {},
                    "attempted_models": list(self._fallback_models),
                },
            )

        raise RuntimeError("OpenRouter API request failed for all models: " + " | ".join(errors))

    @staticmethod
    def _build_fallback_models(primary_model: str) -> tuple[str, ...]:
        ordered = [
            primary_model,
            "openai/gpt-5.2",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
        ]
        deduped: list[str] = []
        for model in ordered:
            normalized = str(model).strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return tuple(deduped)


def _should_stop_after_http_error(status_code: int) -> bool:
    """Stop model fallback when the error is auth or quota related."""

    return status_code in {401, 403, 429}


def _coerce_max_tokens(raw_value: str | None, default: int) -> int:
    try:
        value = int(str(raw_value).strip()) if raw_value is not None and str(raw_value).strip() else default
    except (TypeError, ValueError):
        return default
    return max(1, value)
