"""Dynamic score boosts for retrieval ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .retrieval_v2 import tokenize_text

_KEYWORD_RELEVANCE_THRESHOLD = 0.2
_SEMANTIC_RELEVANCE_THRESHOLD = 0.18
_IRRELEVANT_GENERATED_PENALTY = -0.2
_MEMORY_CONFIDENCE_WEIGHT = 0.3


@dataclass(slots=True, frozen=True)
class BoostDecision:
    """Represents boost adjustments applied to a document."""

    total_boost: float
    components: dict[str, float]


def extract_document_metadata(doc_path: str, content: str, knowledge_base_root: str | Path) -> dict[str, Any]:
    """Extract deterministic metadata used for retrieval boosting."""

    path = Path(doc_path).resolve()
    knowledge_root = Path(knowledge_base_root).resolve()
    try:
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except OSError:
        modified_at = datetime.fromtimestamp(0, tz=timezone.utc)

    origin_task = _extract_origin_task(content)
    is_generated = knowledge_root in path.parents
    is_recent = modified_at >= datetime.now(timezone.utc) - timedelta(days=14)
    return {
        "source": "generated" if is_generated else "static",
        "is_recent": is_recent,
        "origin_task": origin_task,
        "modified_at": modified_at.isoformat(),
    }


def apply_boosts(
    doc_path: str,
    query: str,
    metadata: dict[str, Any],
    keyword_overlap: float = 0.0,
    semantic_similarity: float = 0.0,
) -> BoostDecision:
    """Apply deterministic boost rules based on document provenance and query alignment."""

    query_terms = set(tokenize_text(query))
    origin_terms = set(tokenize_text(str(metadata.get("origin_task", ""))))
    overlap = len(query_terms.intersection(origin_terms)) / len(query_terms) if query_terms and origin_terms else 0.0
    relevant = is_relevant(keyword_overlap=keyword_overlap, semantic_similarity=semantic_similarity)
    is_generated = metadata.get("source") == "generated"

    components = {
        "generated": 0.15 if is_generated and relevant and overlap > 0 else 0.0,
        "recent": 0.1 if is_generated and metadata.get("is_recent") and relevant and overlap > 0 else 0.0,
        "origin_task": 0.0,
        "irrelevant_generated": _IRRELEVANT_GENERATED_PENALTY if is_generated and not relevant else 0.0,
    }
    if relevant and query.strip() and query.lower() in str(metadata.get("origin_task", "")).lower():
        components["origin_task"] = 0.5
    elif relevant and overlap > 0:
        components["origin_task"] = round(0.5 * overlap, 4)

    total = round(sum(components.values()), 4)
    return BoostDecision(total_boost=total, components=components)


def apply_memory_boost(doc_path: str, query: str, memory_entry: dict[str, Any]) -> BoostDecision:
    """Apply retrieval-memory-based score adjustments."""

    successful_docs = set(memory_entry.get("successful_docs", []))
    failed_docs = set(memory_entry.get("failed_docs", []))
    boosted_docs = set(memory_entry.get("boosted_docs", []))
    memory_confidence = float(memory_entry.get("memory_confidence", 0.0))

    components = {
        "memory_success": 0.4 if doc_path in successful_docs else 0.0,
        "memory_failure": -0.2 if doc_path in failed_docs else 0.0,
        "memory_boosted": 0.15 if doc_path in boosted_docs else 0.0,
        "memory_confidence": round(memory_confidence * _MEMORY_CONFIDENCE_WEIGHT, 4)
        if doc_path in successful_docs
        else 0.0,
    }
    total = round(sum(components.values()), 4)
    return BoostDecision(total_boost=total, components=components)


def is_relevant(keyword_overlap: float, semantic_similarity: float) -> bool:
    """Return whether a generated document is sufficiently relevant for positive boosting."""

    return (
        keyword_overlap >= _KEYWORD_RELEVANCE_THRESHOLD
        or semantic_similarity >= _SEMANTIC_RELEVANCE_THRESHOLD
    )


def _extract_origin_task(content: str) -> str:
    marker = "task '"
    lower = content.lower()
    index = lower.find(marker)
    if index == -1:
        return ""
    start = index + len(marker)
    end = content.find("'", start)
    if end == -1:
        return ""
    return content[start:end].strip()
