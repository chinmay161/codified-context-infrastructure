"""Deterministic evaluation metrics for the context engine."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from ..core.models import Response, RetrievedContext, Task

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _extract_task_terms(task: Task, keywords: list[str] | None = None) -> set[str]:
    terms = set(_tokenize(task.query))
    for tag in task.context_tags or []:
        terms.update(_tokenize(tag))
    for file_path in task.files_changed or []:
        terms.update(_tokenize(Path(file_path).stem))
    for keyword in keywords or []:
        terms.update(_tokenize(keyword))
    return terms


def _keyword_hits_from_context(context: RetrievedContext) -> set[str]:
    keyword_hits = context.metadata.get("keyword_hits", {})
    if not isinstance(keyword_hits, dict):
        return set()

    hits: set[str] = set()
    for values in keyword_hits.values():
        if isinstance(values, list):
            hits.update(str(value).lower() for value in values)
    return hits


def retrieval_relevance_score(
    context: RetrievedContext,
    task: Task,
    keywords: list[str] | None = None,
) -> float:
    """Measure how well retrieved context aligns with task and expected keywords."""

    task_terms = _extract_task_terms(task, keywords)
    if not task_terms:
        return 0.0

    hits = _keyword_hits_from_context(context)
    if not hits:
        for document in context.docs:
            hits.update(_tokenize(document))

    score = len(task_terms.intersection(hits)) / len(task_terms)
    return round(min(max(score, 0.0), 1.0), 4)


def retrieval_precision_at_k(
    response: Response,
    expected_keywords: list[str] | None = None,
) -> float:
    """Measure how many retrieved top-k documents contain expected keywords."""

    keywords = {_keyword.lower() for _keyword in (expected_keywords or []) if str(_keyword).strip()}
    if not keywords:
        return 0.0

    metadata = dict(response.context_used.get("metadata", {}))
    top_k = list(metadata.get("top_k", []))
    keyword_hits = dict(metadata.get("keyword_hits", {}))
    if not top_k:
        return 0.0

    relevant_docs = 0
    for path in top_k:
        hits = {str(hit).lower() for hit in keyword_hits.get(path, [])}
        if keywords.intersection(hits):
            relevant_docs += 1

    return round(relevant_docs / len(top_k), 4)


def retrieval_miss_penalty(
    context: RetrievedContext,
    task: Task,
    keywords: list[str] | None = None,
) -> float:
    """Compute a penalty when expected retrieval terms are not present."""

    expected_terms = _extract_task_terms(task, keywords)
    if not expected_terms:
        return 0.0

    observed_terms = _keyword_hits_from_context(context)
    missing_terms = expected_terms.difference(observed_terms)
    penalty = len(missing_terms) / len(expected_terms)
    return round(min(max(penalty, 0.0), 1.0), 4)


def agent_selection_accuracy(response: Response, expected_agent: str) -> float:
    """Return 1.0 when the selected agent matches the expected agent, else 0.0."""

    if not expected_agent.strip():
        return 0.0
    return 1.0 if response.agent_used == expected_agent else 0.0


def routing_confidence_score(response: Response) -> float:
    """Estimate routing confidence from the spread between route scores."""

    trace_metadata = dict(response.trace.get("metadata", {}))
    agent_selection = dict(trace_metadata.get("agent_selection", {}))
    route_scores = agent_selection.get("route_scores", {})
    if not isinstance(route_scores, dict) or not route_scores:
        return 0.0

    ordered_scores = sorted((float(score) for score in route_scores.values()), reverse=True)
    if len(ordered_scores) == 1:
        return 1.0

    winner = ordered_scores[0]
    runner_up = ordered_scores[1]
    if winner <= 0:
        return 0.0
    return round((winner - runner_up) / winner, 4)


def trajectory_consistency_score(
    response: Response,
    expected_flow: list[str] | None = None,
) -> float:
    """Score whether the observed execution steps match the expected trajectory."""

    expected_steps = list(expected_flow or [])
    actual_steps = list(response.trace.get("steps", []))
    if not expected_steps:
        return 1.0 if actual_steps else 0.0
    if not actual_steps:
        return 0.0

    matched = sum(1 for index, step in enumerate(expected_steps) if index < len(actual_steps) and actual_steps[index] == step)
    return round(matched / len(expected_steps), 4)


def context_utilization_score(response: Response) -> float:
    """Estimate whether the execution used retrieved context in a meaningful way."""

    trace = response.trace
    trace_steps = trace.get("steps", [])
    trace_metadata = trace.get("metadata", {})
    retrieval_metadata = trace_metadata.get("retrieval", {})
    context_files = response.context_used.get("files", [])
    docs = response.context_used.get("docs", [])

    signals = [
        1.0 if "retrieve_context" in trace_steps else 0.0,
        1.0 if "build_prompt" in trace_steps else 0.0,
        1.0 if retrieval_metadata.get("retrieval_hits", 0) > 0 else 0.0,
        1.0 if context_files and len(context_files) == len(trace.get("files_used", [])) else 0.0,
        1.0 if retrieval_metadata.get("keyword_hits") else 0.0,
        1.0 if any(isinstance(doc, str) and doc.strip() for doc in docs) else 0.0,
    ]
    return round(sum(signals) / len(signals), 4)


def latency_score(response: Response, target_latency_ms: float = 250.0) -> float:
    """Score latency on a smooth 0-1 curve, with 1.0 at or below target latency."""

    latency_ms = max(response.latency_ms, 0.0)
    if latency_ms <= target_latency_ms:
        return 1.0

    penalty = (latency_ms - target_latency_ms) / max(target_latency_ms, 1.0)
    score = math.exp(-penalty)
    return round(min(max(score, 0.0), 1.0), 4)


def overall_score(
    retrieval_score: float,
    routing_score: float,
    trajectory_score: float,
    context_score: float,
    latency: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Combine individual metrics into a single weighted overall score."""

    metric_weights = weights or {
        "retrieval": 0.3,
        "routing": 0.25,
        "trajectory": 0.2,
        "context": 0.15,
        "latency": 0.15,
    }
    weighted_total = (
        retrieval_score * metric_weights["retrieval"]
        + routing_score * metric_weights["routing"]
        + trajectory_score * metric_weights["trajectory"]
        + context_score * metric_weights["context"]
        + latency * metric_weights["latency"]
    )
    return round(min(max(weighted_total, 0.0), 1.0), 4)


def failure_mode_classifier(
    response: Response,
    expected: dict[str, Any],
    retrieval_score: float,
    routing_score: float,
) -> str:
    """Classify failures based on retrieval and routing outcomes."""

    from .analysis import classify_failure

    return classify_failure(
        response=response,
        expected=expected,
        retrieval_score=retrieval_score,
        routing_score=routing_score,
    )


def build_retrieved_context_from_response(response: Response) -> RetrievedContext:
    """Reconstruct a RetrievedContext object from response payloads."""

    return RetrievedContext(
        docs=list(response.context_used.get("docs", [])),
        files=list(response.context_used.get("files", [])),
        metadata=dict(response.context_used.get("metadata", {})),
    )
