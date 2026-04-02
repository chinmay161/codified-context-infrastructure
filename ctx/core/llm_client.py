"""LLM execution clients with Grok primary and Gemini fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..utils.env import load_dotenv

_XAI_BASE_URL = "https://api.x.ai/v1"
_XAI_DEFAULT_MODEL = "grok-4-1-fast-non-reasoning"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(slots=True, frozen=True)
class LLMResult:
    """LLM completion payload plus metadata."""

    output: str
    metadata: dict[str, Any]


class LLMClient:
    """Provider-aware client with Grok primary and Gemini fallback."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        load_dotenv()
        self._timeout_seconds = timeout_seconds
        self._xai_client = GrokClient(timeout_seconds=timeout_seconds)
        self._gemini_client = GeminiClient(timeout_seconds=timeout_seconds)
        self._provider_order = self._resolve_provider_order()

    @property
    def is_configured(self) -> bool:
        return self._xai_client.is_configured or self._gemini_client.is_configured

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using the configured provider order."""

        provider_errors: list[str] = []
        for provider_name in self._provider_order:
            if provider_name == "grok":
                if not self._xai_client.is_configured:
                    provider_errors.append("grok: not configured")
                    continue
                try:
                    return self._xai_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )
                except RuntimeError as exc:
                    provider_errors.append(f"grok: {exc}")
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

        raise RuntimeError("All configured LLM providers failed: " + " | ".join(provider_errors))

    @staticmethod
    def _resolve_provider_order() -> tuple[str, ...]:
        raw_value = os.environ.get("CTX_LLM_PROVIDER_ORDER", "grok,gemini").strip().lower()
        ordered = [item.strip() for item in raw_value.split(",") if item.strip()]
        deduped: list[str] = []
        for item in ordered:
            if item in {"grok", "gemini"} and item not in deduped:
                deduped.append(item)
        if not deduped:
            return ("grok", "gemini")
        return tuple(deduped)


class GrokClient:
    """Small stdlib-based client for xAI chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        load_dotenv()
        self._api_key = api_key or os.environ.get("XAI_API_KEY", "").strip()
        self._model = model or os.environ.get("XAI_MODEL", _XAI_DEFAULT_MODEL).strip() or _XAI_DEFAULT_MODEL
        self._base_url = (base_url or os.environ.get("XAI_BASE_URL", _XAI_BASE_URL)).rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._fallback_models = self._build_fallback_models(self._model)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and self._api_key.lower() != "your_xai_api_key_here"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a completion using xAI's chat completions API."""

        if not self.is_configured:
            raise RuntimeError("XAI_API_KEY is not configured.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        errors: list[str] = []
        for candidate_model in self._fallback_models:
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
                continue
            except urllib.error.URLError as exc:
                errors.append(f"{candidate_model}: {exc.reason}")
                continue

            parsed = json.loads(body)
            choices = parsed.get("choices", [])
            if not choices:
                errors.append(f"{candidate_model}: missing choices")
                continue
            message = choices[0].get("message", {})
            content = str(message.get("content", "")).strip()
            if not content:
                errors.append(f"{candidate_model}: missing message content")
                continue

            usage = parsed.get("usage", {})
            return LLMResult(
                output=content,
                metadata={
                    "provider": "xai",
                    "model": parsed.get("model", candidate_model),
                    "usage": usage if isinstance(usage, dict) else {},
                    "attempted_models": list(self._fallback_models),
                },
            )

        raise RuntimeError("xAI API request failed for all models: " + " | ".join(errors))

    @staticmethod
    def _build_fallback_models(primary_model: str) -> tuple[str, ...]:
        ordered = [
            primary_model,
            "grok-4-1-fast-non-reasoning",
            "grok-4-1-fast-reasoning",
            "grok-4.20-0309-non-reasoning",
            "grok-4.20-0309-reasoning",
            "grok-4.20-multi-agent-0309",
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
