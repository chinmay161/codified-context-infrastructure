"""Failure analysis utilities for context-engine evaluation."""

from __future__ import annotations

from typing import Any

from ..core.models import Response


def classify_failure(
    response: Response,
    expected: dict[str, Any],
    retrieval_score: float,
    routing_score: float,
) -> str:
    """Classify the dominant failure mode for an evaluated execution."""

    context_files = list(response.context_used.get("files", []))
    if not context_files:
        return "context_missing"

    expected_agent = str(expected.get("expected_agent", ""))
    if expected_agent and routing_score < 1.0:
        return "routing_failure"

    retrieval_threshold = float(expected.get("retrieval_threshold", 0.45))
    if retrieval_score < retrieval_threshold:
        return "retrieval_failure"

    return "ok"
