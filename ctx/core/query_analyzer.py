"""Deterministic query understanding helpers for adaptive retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .retrieval_v2 import tokenize_text

_STRUCTURAL_TERMS = {
    "architecture",
    "chain",
    "connected",
    "dependency",
    "edge",
    "flow",
    "graph",
    "hop",
    "integration",
    "linked",
    "path",
    "pipeline",
    "relationship",
    "route",
    "routing",
    "structure",
    "system",
    "traversal",
}
_INTENT_TERMS = {
    "debug": {"bug", "debug", "error", "failure", "fix", "issue", "regression", "trace"},
    "explain": {"describe", "explain", "how", "overview", "summary", "understand", "why"},
    "implement": {"add", "build", "create", "implement", "introduce", "make", "support", "upgrade"},
    "optimize": {"boost", "improve", "optimize", "performance", "quality", "rank", "stability"},
}
_ENTITY_PREFIXES = ("ctx", "api", "db", "ui", "v1", "v2", "v3")


@dataclass(slots=True, frozen=True)
class QueryAnalysis:
    """Structured query understanding used by retrieval optimization."""

    query_type: str
    entities: tuple[str, ...]
    intent: str
    tokens: tuple[str, ...]
    unique_tokens: tuple[str, ...]
    token_count: int
    is_seen_query: bool
    is_long_query: bool
    is_abstract_query: bool
    is_keyword_query: bool
    is_structural_query: bool
    concept_terms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "entities": list(self.entities),
            "intent": self.intent,
            "tokens": list(self.tokens),
            "unique_tokens": list(self.unique_tokens),
            "token_count": self.token_count,
            "is_seen_query": self.is_seen_query,
            "is_long_query": self.is_long_query,
            "is_abstract_query": self.is_abstract_query,
            "is_keyword_query": self.is_keyword_query,
            "is_structural_query": self.is_structural_query,
            "concept_terms": list(self.concept_terms),
        }


def analyze_query(query: str, context: dict[str, Any] | None = None) -> QueryAnalysis:
    """Analyze a query with simple deterministic heuristics."""

    context = context or {}
    tokens = tuple(tokenize_text(query))
    unique_tokens = tuple(sorted(set(tokens)))
    token_count = len(tokens)
    memory_entry = context.get("memory_entry", {}) if isinstance(context.get("memory_entry", {}), dict) else {}
    graph_hint_terms = {
        str(term).lower()
        for term in context.get("graph_terms", [])
        if str(term).strip()
    }

    structural_hits = {
        token for token in unique_tokens if token in _STRUCTURAL_TERMS or token in graph_hint_terms
    }
    keyword_like = token_count <= 4 and len(structural_hits) == 0
    long_query = token_count >= 9
    abstract_query = _is_abstract_query(unique_tokens)
    structural_query = bool(structural_hits) or _contains_structural_pattern(query)

    if structural_query:
        query_type = "structural"
    elif keyword_like and not abstract_query:
        query_type = "keyword"
    else:
        query_type = "semantic"

    return QueryAnalysis(
        query_type=query_type,
        entities=_extract_entities(query, unique_tokens),
        intent=_infer_intent(unique_tokens),
        tokens=tokens,
        unique_tokens=unique_tokens,
        token_count=token_count,
        is_seen_query=bool(memory_entry.get("total_uses", 0)),
        is_long_query=long_query,
        is_abstract_query=abstract_query,
        is_keyword_query=keyword_like,
        is_structural_query=structural_query,
        concept_terms=tuple(sorted(structural_hits)),
    )


def _extract_entities(query: str, unique_tokens: tuple[str, ...]) -> tuple[str, ...]:
    entities: list[str] = []
    for raw_term in query.replace("/", " ").replace("-", " ").split():
        cleaned = raw_term.strip(".,:;()[]{}<>\"'`")
        if not cleaned:
            continue
        if cleaned.lower().startswith(_ENTITY_PREFIXES) or any(char.isupper() for char in cleaned[1:]):
            entities.append(cleaned.lower())
    if not entities:
        entities.extend(token for token in unique_tokens if "_" in token or any(char.isdigit() for char in token))
    return tuple(sorted(dict.fromkeys(entities)))


def _infer_intent(unique_tokens: tuple[str, ...]) -> str:
    scores: dict[str, int] = {}
    query_set = set(unique_tokens)
    for intent, terms in _INTENT_TERMS.items():
        scores[intent] = len(query_set.intersection(terms))
    best_intent = max(scores, key=lambda key: (scores[key], key)) if scores else "explore"
    return best_intent if scores.get(best_intent, 0) > 0 else "explore"


def _is_abstract_query(unique_tokens: tuple[str, ...]) -> bool:
    abstract_terms = {
        "accuracy",
        "adapt",
        "adaptive",
        "behavior",
        "benefit",
        "concept",
        "context",
        "optimize",
        "quality",
        "ranking",
        "reasoning",
        "relevance",
        "semantic",
        "stability",
    }
    return len(set(unique_tokens).intersection(abstract_terms)) >= 2


def _contains_structural_pattern(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in ("->", "between", "depends on", "connected to", "related to", "path from")
    )
