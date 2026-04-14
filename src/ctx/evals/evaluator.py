"""Evaluation orchestration for context engine responses."""

from __future__ import annotations

from typing import Any

from ..core.models import Response, Task
from .metrics import (
    agent_selection_accuracy,
    build_retrieved_context_from_response,
    context_utilization_score,
    failure_mode_classifier,
    latency_score,
    overall_score,
    retrieval_miss_penalty,
    retrieval_precision_at_k,
    retrieval_relevance_score,
    routing_confidence_score,
    trajectory_consistency_score,
)


class Evaluator:
    """Evaluate context-engine executions across retrieval, routing, and efficiency."""

    def evaluate(self, response: Response, expected: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a response against an expected routing and retrieval profile."""

        task = Task(
            query=str(expected.get("task", "")),
            files_changed=list(expected.get("files_changed", []) or []) or None,
            context_tags=list(expected.get("context_tags", []) or []) or None,
        )
        retrieved_context = build_retrieved_context_from_response(response)
        retrieval_score = retrieval_relevance_score(
            context=retrieved_context,
            task=task,
            keywords=list(expected.get("expected_keywords", []) or []),
        )
        routing_score = agent_selection_accuracy(
            response=response,
            expected_agent=str(expected.get("expected_agent", "")),
        )
        routing_confidence = routing_confidence_score(response)
        trajectory_score = trajectory_consistency_score(
            response=response,
            expected_flow=list(expected.get("expected_flow", []) or []),
        )
        precision_at_k = retrieval_precision_at_k(
            response=response,
            expected_keywords=list(expected.get("expected_keywords", []) or []),
        )
        retrieval_penalty = retrieval_miss_penalty(
            context=retrieved_context,
            task=task,
            keywords=list(expected.get("expected_keywords", []) or []),
        )
        context_score = context_utilization_score(response)
        latency = latency_score(response)
        failure_mode = failure_mode_classifier(
            response=response,
            expected=expected,
            retrieval_score=retrieval_score,
            routing_score=routing_score,
        )
        retrieval_composite = max(
            0.0,
            min(1.0, round((0.6 * retrieval_score) + (0.4 * precision_at_k) - (0.3 * retrieval_penalty), 4)),
        )
        overall = overall_score(
            retrieval_score=retrieval_composite,
            routing_score=round((0.7 * routing_score) + (0.3 * routing_confidence), 4),
            trajectory_score=trajectory_score,
            context_score=context_score,
            latency=latency,
        )

        return {
            "task": task.query,
            "trace_id": response.trace_id,
            "expected_agent": expected.get("expected_agent", ""),
            "expected_keywords": list(expected.get("expected_keywords", []) or []),
            "selected_agent": response.agent_used,
            "retrieval_score": retrieval_score,
            "retrieval_precision_at_k": precision_at_k,
            "retrieval_miss_penalty": retrieval_penalty,
            "observed_keywords": sorted(
                {
                    str(keyword).lower()
                    for keywords in response.context_used.get("metadata", {}).get("keyword_hits", {}).values()
                    for keyword in keywords
                }
            ),
            "routing_score": routing_score,
            "routing_confidence": routing_confidence,
            "trajectory_score": trajectory_score,
            "context_utilization": context_score,
            "latency_ms": round(response.latency_ms, 3),
            "latency_score": latency,
            "failure_mode": failure_mode,
            "overall_score": overall,
            "boost_effect": response.context_used.get("metadata", {}).get("boost_effect", []),
            "boosted_docs": response.context_used.get("metadata", {}).get("boosted_docs", []),
            "memory_hit": response.context_used.get("metadata", {}).get("memory_hit", False),
            "memory_confidence": response.context_used.get("metadata", {}).get("memory_confidence", 0.0),
            "query_type": response.context_used.get("metadata", {}).get("query_type", "keyword"),
            "weight_distribution": response.context_used.get("metadata", {}).get("weight_distribution", {}),
            "rerank_effect": response.context_used.get("metadata", {}).get("rerank_effect", []),
            "trace_steps": list(response.trace.get("steps", [])),
        }
