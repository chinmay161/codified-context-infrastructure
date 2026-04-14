"""Deterministic multi-stage reranking helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .query_analyzer import QueryAnalysis, analyze_query
from .retrieval_v2 import tokenize_text

_EMBEDDING_DIMENSION = 96


@dataclass(slots=True, frozen=True)
class RerankResult:
    """Reranked document payload with observability details."""

    path: str
    score: float
    before_rank: int
    after_rank: int
    delta: float
    signal_breakdown: dict[str, float]


def rerank(
    docs: list[dict[str, Any]],
    query: str,
    query_analysis: QueryAnalysis | None = None,
) -> tuple[list[dict[str, Any]], list[RerankResult]]:
    """Apply a deterministic fine-grained reranking pass over candidate docs."""

    analysis = query_analysis or analyze_query(query)
    query_tokens = tokenize_text(query)
    query_embedding = compute_embedding(query)
    reranked_docs: list[dict[str, Any]] = []
    before_ranks = {doc["path"]: index for index, doc in enumerate(docs, start=1)}

    for doc in docs:
        signal_breakdown = _fine_grained_score(
            doc=doc,
            query=query,
            query_tokens=query_tokens,
            query_embedding=query_embedding,
            query_analysis=analysis,
        )
        rerank_delta = round(sum(signal_breakdown.values()), 6)
        updated_doc = {
            **doc,
            "rerank_delta": rerank_delta,
            "rerank_signals": signal_breakdown,
            "score": round(float(doc["score"]) + rerank_delta, 6),
        }
        reranked_docs.append(updated_doc)

    reranked_docs.sort(key=lambda item: (-float(item["score"]), item["path"]))
    effects: list[RerankResult] = []
    for index, doc in enumerate(reranked_docs, start=1):
        effects.append(
            RerankResult(
                path=str(doc["path"]),
                score=round(float(doc["score"]), 6),
                before_rank=before_ranks[str(doc["path"])],
                after_rank=index,
                delta=round(float(doc.get("rerank_delta", 0.0)), 6),
                signal_breakdown=dict(doc.get("rerank_signals", {})),
            )
        )
    return reranked_docs, effects


def _fine_grained_score(
    doc: dict[str, Any],
    query: str,
    query_tokens: list[str],
    query_embedding: list[float],
    query_analysis: QueryAnalysis,
) -> dict[str, float]:
    content = str(doc.get("content", ""))
    path = str(doc.get("path", ""))
    doc_tokens = tokenize_text(f"{Path(path).stem} {content}")
    doc_token_set = set(doc_tokens)
    query_token_set = set(query_tokens)
    overlap = query_token_set.intersection(doc_token_set)

    semantic_precision = compute_cosine_similarity(query_embedding, compute_embedding(f"{Path(path).stem} {content}"))
    graph_distance = float(doc.get("graph_distance", 99.0))
    graph_distance_score = 0.0 if graph_distance >= 99.0 else 1.0 / (graph_distance + 1.0)
    keyword_density = len(overlap) / max(len(doc_tokens), 1)
    concept_overlap = len(set(query_analysis.concept_terms).intersection(doc_token_set)) / max(
        len(query_analysis.concept_terms),
        1,
    ) if query_analysis.concept_terms else 0.0

    semantic_weight = 0.05 if query_analysis.query_type == "keyword" else 0.08
    graph_weight = 0.05 if query_analysis.is_structural_query else 0.025
    keyword_weight = 0.045 if query_analysis.is_keyword_query else 0.03
    concept_weight = 0.05 if query_analysis.is_structural_query else 0.025

    return {
        "semantic_similarity": round(semantic_precision * semantic_weight, 6),
        "graph_distance": round(graph_distance_score * graph_weight, 6),
        "keyword_density": round(keyword_density * keyword_weight, 6),
        "concept_overlap": round(concept_overlap * concept_weight, 6),
    }


def compute_embedding(text: str) -> list[float]:
    """Compute the same deterministic lightweight embedding used for reranking."""

    vector = [0.0] * _EMBEDDING_DIMENSION
    tokens = tokenize_text(text)
    if not tokens:
        return vector

    for token in tokens:
        token_weight = 1.0 + min(len(token), 12) / 12.0
        for ngram_size in (3, 4):
            if len(token) < ngram_size:
                ngrams = [token]
            else:
                ngrams = [token[index : index + ngram_size] for index in range(len(token) - ngram_size + 1)]

            for ngram in ngrams:
                bucket = _stable_hash(f"{ngram_size}:{ngram}") % _EMBEDDING_DIMENSION
                sign = 1.0 if (_stable_hash(f"sign:{ngram}") % 2 == 0) else -1.0
                vector[bucket] += sign * token_weight / len(ngrams)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""

    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(left * right for left, right in zip(vec1, vec2))
    norm_left = math.sqrt(sum(value * value for value in vec1))
    norm_right = math.sqrt(sum(value * value for value in vec2))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot_product / (norm_left * norm_right)


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
