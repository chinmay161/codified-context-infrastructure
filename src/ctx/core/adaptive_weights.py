"""Adaptive retrieval weighting for query-aware score fusion."""

from __future__ import annotations

from typing import Any

from .query_analyzer import QueryAnalysis, analyze_query

_BASE_WEIGHTS = {
    "bm25": 0.3,
    "semantic": 0.26,
    "graph": 0.2,
    "boost": 0.14,
    "memory": 0.1,
}


def compute_weights(query: str, context: dict[str, Any] | None = None) -> dict[str, float]:
    """Compute normalized retrieval weights from query and runtime context."""

    context = context or {}
    analysis = _resolve_analysis(query=query, context=context)
    memory_entry = context.get("memory_entry", {}) if isinstance(context.get("memory_entry", {}), dict) else {}
    weights = dict(_BASE_WEIGHTS)

    if analysis.is_keyword_query:
        weights["bm25"] += 0.12
        weights["semantic"] -= 0.05
    if analysis.is_long_query or analysis.is_abstract_query:
        weights["semantic"] += 0.1
        weights["graph"] += 0.06
        weights["bm25"] -= 0.07
    if analysis.is_structural_query:
        weights["graph"] += 0.12
        weights["semantic"] += 0.03
        weights["bm25"] -= 0.04
    if analysis.is_seen_query:
        confidence = float(memory_entry.get("memory_confidence", 0.0))
        weights["memory"] += 0.08 + min(confidence, 0.15)
        weights["boost"] += 0.02
    if analysis.intent == "optimize":
        weights["semantic"] += 0.03
        weights["graph"] += 0.02
    if analysis.intent == "debug":
        weights["bm25"] += 0.03
        weights["memory"] += 0.02

    feedback_adjustments = context.get("feedback_adjustments", {})
    if isinstance(feedback_adjustments, dict):
        for key in weights:
            weights[key] += float(feedback_adjustments.get(key, 0.0))

    return _normalize_weights(weights)


def update_weights_from_feedback(query: str, performance: dict[str, Any]) -> dict[str, float]:
    """Return deterministic weight deltas informed by retrieval performance."""

    analysis = analyze_query(query=query, context={"memory_entry": performance.get("memory_entry", {})})
    previous_score = float(performance.get("previous_score", performance.get("baseline_score", 0.0)) or 0.0)
    current_score = float(performance.get("retrieval_score", 0.0) or 0.0)
    improvement = round(current_score - previous_score, 4)

    adjustments = {key: 0.0 for key in _BASE_WEIGHTS}
    if improvement >= 0.05:
        if analysis.is_keyword_query:
            adjustments["bm25"] += 0.02
        if analysis.is_structural_query:
            adjustments["graph"] += 0.02
        if analysis.is_long_query or analysis.is_abstract_query:
            adjustments["semantic"] += 0.02
        adjustments["memory"] += 0.01
    elif improvement <= -0.05:
        if analysis.is_keyword_query:
            adjustments["semantic"] += 0.01
            adjustments["bm25"] -= 0.01
        if analysis.is_structural_query:
            adjustments["graph"] -= 0.015
            adjustments["semantic"] += 0.015
        adjustments["boost"] -= 0.005
        adjustments["memory"] -= 0.005

    return {key: round(value, 4) for key, value in adjustments.items()}


def _resolve_analysis(query: str, context: dict[str, Any]) -> QueryAnalysis:
    existing = context.get("query_analysis")
    if isinstance(existing, QueryAnalysis):
        return existing
    return analyze_query(query=query, context=context)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(value, 0.01) for key, value in weights.items()}
    total = sum(clipped.values()) or 1.0
    normalized = {key: round(value / total, 6) for key, value in clipped.items()}

    # Keep the sum stable at exactly 1.0 after rounding.
    delta = round(1.0 - sum(normalized.values()), 6)
    if delta != 0:
        dominant = max(normalized, key=normalized.get)
        normalized[dominant] = round(normalized[dominant] + delta, 6)
    return normalized
