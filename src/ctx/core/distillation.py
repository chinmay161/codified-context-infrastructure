"""Deterministic context distillation and compile-confidence helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import ContextDistillation, RetrievedContext, Task
from .query_analyzer import QueryAnalysis

_RULE_PATTERN = re.compile(
    r"\b(must|should|always|never|required|do not|don't|only|avoid)\b",
    re.IGNORECASE,
)
_API_PATTERN = re.compile(
    r"(`[^`]+`|\b(GET|POST|PUT|PATCH|DELETE)\b|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|\b[a-zA-Z_][a-zA-Z0-9_]*:\s+\S+)",
    re.IGNORECASE,
)
_RELATION_PATTERN = re.compile(
    r"\b(depends on|uses|calls|publishes|subscribes to|flows into|maps to|returns|writes to|reads from)\b",
    re.IGNORECASE,
)


def distill_context(
    task: Task,
    retrieved_context: RetrievedContext,
    query_analysis: QueryAnalysis,
) -> ContextDistillation:
    """Produce deterministic, compact context artifacts for prompt building."""

    docs = list(zip(retrieved_context.files, retrieved_context.docs))
    priority_order = _priority_order(retrieved_context)
    evidence_snippets = _extract_evidence_snippets(docs, priority_order)
    rule_extractions = _extract_rules(docs)
    api_summary = _extract_api_summary(docs)
    relationship_map = _extract_relationships(docs, retrieved_context.metadata)
    distillation_confidence = _compute_distillation_confidence(
        docs=docs,
        priority_order=priority_order,
        rule_extractions=rule_extractions,
        api_summary=api_summary,
        relationship_map=relationship_map,
    )

    compression_notes: list[str] = []
    if len(retrieved_context.docs) > 3:
        compression_notes.append("Prioritized top-ranked documents before prompt assembly.")
    if not rule_extractions and not api_summary and not relationship_map:
        compression_notes.append("Distillation fell back to evidence snippets because structured signals were sparse.")
    if query_analysis.is_long_query:
        compression_notes.append("Long query detected; favored higher-signal summaries over raw context.")

    return ContextDistillation(
        rule_extractions=rule_extractions,
        api_summary=api_summary,
        relationship_map=relationship_map,
        priority_order=priority_order,
        compression_notes=compression_notes,
        budget_decisions={},
        distillation_confidence=distillation_confidence,
        evidence_snippets=evidence_snippets,
    )


def compute_compile_confidence(
    retrieval_metadata: dict[str, Any],
    distillation: ContextDistillation,
) -> dict[str, Any]:
    """Compute compile-time confidence used to shape prompt scaffolding."""

    ranking = retrieval_metadata.get("ranking", [])
    top_scores = [
        float(item.get("score", 0.0))
        for item in ranking
        if isinstance(item, dict)
    ][:3]
    ranking_strength = sum(top_scores) / len(top_scores) if top_scores else 0.0
    ranking_agreement = 1.0
    if len(top_scores) >= 2:
        ranking_agreement = max(0.0, 1.0 - abs(top_scores[0] - top_scores[-1]))
    memory_bonus = 0.1 if retrieval_metadata.get("memory_hit") else 0.0
    score = max(
        0.0,
        min(
            1.0,
            round(
                0.5 * ranking_strength
                + 0.2 * ranking_agreement
                + 0.2 * distillation.distillation_confidence
                + memory_bonus,
                4,
            ),
        ),
    )
    if score >= 0.75:
        band = "high"
    elif score >= 0.45:
        band = "medium"
    else:
        band = "low"
    return {
        "score": score,
        "band": band,
        "signals": {
            "ranking_strength": round(ranking_strength, 4),
            "ranking_agreement": round(ranking_agreement, 4),
            "memory_hit": bool(retrieval_metadata.get("memory_hit")),
            "distillation_confidence": round(distillation.distillation_confidence, 4),
        },
    }


def _priority_order(retrieved_context: RetrievedContext) -> list[str]:
    ranked = [
        _priority_label(str(item.get("path", "")).strip())
        for item in retrieved_context.metadata.get("ranking", [])
        if isinstance(item, dict) and str(item.get("path", "")).strip()
    ]
    if ranked:
        return ranked
    return [_priority_label(path) for path in retrieved_context.files]


def _extract_rules(docs: list[tuple[str, str]]) -> list[str]:
    items: list[str] = []
    for path, content in docs:
        filename = Path(path).name
        for line in _candidate_lines(content):
            if _RULE_PATTERN.search(line):
                items.append(f"{filename}: {_clean_line(line)}")
            if len(items) >= 6:
                return items
    return items


def _extract_api_summary(docs: list[tuple[str, str]]) -> list[str]:
    items: list[str] = []
    for path, content in docs:
        filename = Path(path).name
        for line in _candidate_lines(content):
            if _API_PATTERN.search(line):
                items.append(f"{filename}: {_clean_line(line)}")
            if len(items) >= 6:
                return items
    return items


def _extract_relationships(docs: list[tuple[str, str]], metadata: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for path, content in docs:
        filename = Path(path).name
        for line in _candidate_lines(content):
            if _RELATION_PATTERN.search(line):
                items.append(f"{filename}: {_clean_line(line)}")
            if len(items) >= 4:
                return items

    top_files = [Path(path).name for path in metadata.get("top_k", []) if str(path).strip()]
    if len(top_files) >= 2:
        items.append(f"{top_files[0]} is prioritized ahead of {top_files[1]} for this task.")
    return items


def _extract_evidence_snippets(
    docs: list[tuple[str, str]],
    priority_order: list[str],
) -> list[str]:
    ordered = sorted(
        docs,
        key=lambda item: (
            priority_order.index(_priority_label(item[0]))
            if _priority_label(item[0]) in priority_order
            else len(priority_order)
        ),
    )
    snippets: list[str] = []
    for path, content in ordered[:3]:
        snippet = _extract_sentence_snippet(content)
        if not snippet:
            continue
        snippets.append(f"{Path(path).name}: {snippet}")
    return snippets


def _compute_distillation_confidence(
    docs: list[tuple[str, str]],
    priority_order: list[str],
    rule_extractions: list[str],
    api_summary: list[str],
    relationship_map: list[str],
) -> float:
    if not docs:
        return 0.0
    coverage = min(len(priority_order) / max(len(docs), 1), 1.0)
    signal_density = min(
        (len(rule_extractions) + len(api_summary) + len(relationship_map)) / 8.0,
        1.0,
    )
    return round(min(1.0, 0.45 * coverage + 0.55 * signal_density), 4)


def _candidate_lines(content: str) -> list[str]:
    candidates: list[str] = []
    for block in content.splitlines():
        line = block.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("<!--")
            or _is_table_row(line)
            or _is_table_divider(line)
        ):
            continue
        candidates.append(line)
    return candidates


def _clean_line(line: str) -> str:
    collapsed = re.sub(r"\s+", " ", line).strip(" -\t")
    return collapsed[:220]


def _priority_label(path: str) -> str:
    return Path(path).name if path else path


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|")


def _is_table_divider(line: str) -> bool:
    compact = line.replace(" ", "")
    return compact.startswith("|-") and compact.endswith("|")


def _extract_sentence_snippet(content: str, limit: int = 200) -> str:
    paragraphs = [
        _clean_line(line)
        for line in content.splitlines()
        if line.strip()
        and not line.strip().startswith("#")
        and not line.strip().startswith("<!--")
        and not _is_table_row(line.strip())
        and not _is_table_divider(line.strip())
    ]
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            return paragraph
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        chosen: list[str] = []
        total = 0
        for sentence in sentences:
            if not sentence:
                continue
            addition = len(sentence) + (1 if chosen else 0)
            if total + addition > limit:
                break
            chosen.append(sentence)
            total += addition
        if chosen:
            return " ".join(chosen)
        shortened = paragraph[:limit].rstrip()
        last_space = shortened.rfind(" ")
        if last_space > 80:
            shortened = shortened[:last_space].rstrip()
        return f"{shortened}..."
    return ""
