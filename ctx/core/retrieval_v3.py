"""Adaptive hybrid retrieval with query-aware fusion and reranking."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adaptive_weights import compute_weights
from .graph_retrieval import GraphRetriever
from .models import Agent, Task
from .query_analyzer import QueryAnalysis, analyze_query
from .reranker import rerank
from .retrieval_boost import apply_boosts, apply_memory_boost, extract_document_metadata
from .retrieval_memory import RetrievalMemory
from .retrieval_v2 import RetrievalScore, rank_documents, tokenize_text
from ..utils.file_loader import MarkdownDocument

_EMBEDDING_DIMENSION = 96
_RRF_K = 60
_CANDIDATE_LIMIT = 20


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def compute_embedding(text: str) -> list[float]:
    """Compute a deterministic lightweight embedding vector for local semantic search."""

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


@dataclass(slots=True, frozen=True)
class SemanticDocument:
    """A document enriched with a cached embedding vector."""

    path: str
    content: str
    embedding: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class SemanticScore:
    """Semantic retrieval score for a document."""

    path: str
    content: str
    similarity: float
    rank: int


class HybridRetriever:
    """Adaptive cached retriever that combines lexical, semantic, graph, boost, and memory signals."""

    def __init__(
        self,
        documents: list[MarkdownDocument],
        knowledge_base_root: str | Path,
        default_top_k: int = 5,
        rrf_k: int = _RRF_K,
        memory: RetrievalMemory | None = None,
    ) -> None:
        self._documents = list(documents)
        self._default_top_k = default_top_k
        self._rrf_k = rrf_k
        self._knowledge_base_root = Path(knowledge_base_root).resolve()
        self._memory = memory or RetrievalMemory()
        self._semantic_documents = self._build_semantic_index(self._documents)
        self._graph_retriever = GraphRetriever(self._documents)
        self._document_map = {document.path: document.content for document in self._documents}
        self._document_metadata = {
            document.path: extract_document_metadata(
                doc_path=document.path,
                content=document.content,
                knowledge_base_root=self._knowledge_base_root,
            )
            for document in self._documents
        }

    def retrieve(
        self,
        task: Task,
        agent: Agent,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalScore], dict[str, object]]:
        """Run adaptive multi-stage retrieval and return deterministic results plus metadata."""

        effective_top_k = top_k or self._default_top_k
        candidate_depth = max(min(effective_top_k * 4, _CANDIDATE_LIMIT), 10)
        memory_entry = self._memory.get(task.query)
        query_analysis = analyze_query(
            query=task.query,
            context={
                "memory_entry": memory_entry,
                "graph_terms": list(self._graph_retriever.graph.concept_index.keys()),
            },
        )
        expanded_task, expansion_terms, graph_expansion_terms = self._expand_task(task, query_analysis)
        adaptive_weights = compute_weights(
            query=task.query,
            context={
                "memory_entry": memory_entry,
                "query_analysis": query_analysis,
                "feedback_adjustments": self._memory.weight_adjustments(task.query),
            },
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            lexical_future = executor.submit(
                rank_documents,
                self._documents,
                expanded_task,
                agent,
                candidate_depth,
            )
            semantic_future = executor.submit(
                self._semantic_rank_documents,
                expanded_task,
                candidate_depth,
            )
            graph_future = executor.submit(
                self._graph_retriever.retrieve,
                expanded_task.query,
                expanded_task,
                candidate_depth,
            )
            lexical_results, lexical_metadata = lexical_future.result()
            semantic_results = semantic_future.result()
            graph_result = graph_future.result()

        lexical_rank_map = {result.path: index for index, result in enumerate(lexical_results, start=1)}
        semantic_rank_map = {result.path: result.rank for result in semantic_results}
        lexical_result_map = {result.path: result for result in lexical_results}
        semantic_result_map = {result.path: result for result in semantic_results}
        graph_score_map = {path: result.score for path, result in graph_result.scores.items()}
        graph_result_map = graph_result.scores

        candidate_paths = set(lexical_rank_map).union(semantic_rank_map).union(graph_score_map)
        fused_scores: dict[str, float] = {}
        scoring_breakdown: dict[str, dict[str, Any]] = {}
        keyword_hits: dict[str, list[str]] = {}
        boost_events: list[dict[str, object]] = []
        candidate_docs: list[dict[str, Any]] = []

        normalized_lexical = _normalize_component_scores(
            {path: result.total_score for path, result in lexical_result_map.items()}
        )
        normalized_semantic = _normalize_component_scores(
            {path: result.similarity for path, result in semantic_result_map.items()}
        )
        normalized_graph = _normalize_component_scores(graph_score_map)

        for path in candidate_paths:
            lexical_result = lexical_result_map.get(path)
            semantic_result = semantic_result_map.get(path)
            graph_result_item = graph_result_map.get(path)
            document_metadata = self._document_metadata.get(path, {})

            bm25_signal = normalized_lexical.get(path, 0.0)
            semantic_signal = normalized_semantic.get(path, 0.0)
            graph_signal = normalized_graph.get(path, 0.0)
            static_boost = apply_boosts(
                path,
                task.query,
                document_metadata,
                keyword_overlap=lexical_result.keyword_score if lexical_result else 0.0,
                semantic_similarity=semantic_result.similarity if semantic_result else 0.0,
            )
            memory_boost = apply_memory_boost(path, task.query, memory_entry)
            boost_signal = min(max(static_boost.total_boost, 0.0), 1.0)
            memory_signal = min(max(memory_boost.total_boost, 0.0), 1.0)

            weighted_score = round(
                adaptive_weights["bm25"] * bm25_signal
                + adaptive_weights["semantic"] * semantic_signal
                + adaptive_weights["graph"] * graph_signal
                + adaptive_weights["boost"] * boost_signal
                + adaptive_weights["memory"] * memory_signal,
                8,
            )
            fused_scores[path] = weighted_score
            keyword_hits[path] = list(lexical_result.matched_terms if lexical_result else ())

            scoring_breakdown[path] = {
                "lexical_rank": lexical_rank_map.get(path),
                "semantic_rank": semantic_rank_map.get(path),
                "bm25_signal": round(bm25_signal, 6),
                "semantic_signal": round(semantic_signal, 6),
                "graph_signal": round(graph_signal, 6),
                "boost_signal": round(boost_signal, 6),
                "memory_signal": round(memory_signal, 6),
                "lexical_total": lexical_result.total_score if lexical_result else 0.0,
                "keyword_overlap": lexical_result.keyword_score if lexical_result else 0.0,
                "semantic_similarity": semantic_result.similarity if semantic_result else 0.0,
                "graph_score": round(graph_score_map.get(path, 0.0), 8),
                "graph_distance": graph_result_item.distance if graph_result_item else None,
                "graph_connections": graph_result_item.connections if graph_result_item else 0,
                "rrf_score": round(self._rrf_score(lexical_rank_map.get(path), semantic_rank_map.get(path)), 6),
                "boost_score": static_boost.total_boost,
                "memory_boost": memory_boost.total_boost,
                "memory_confidence": memory_entry.get("memory_confidence", 0.0),
                "weight_distribution": adaptive_weights,
                "weighted_score": weighted_score,
            }
            candidate_docs.append(
                {
                    "path": path,
                    "content": self._document_map[path],
                    "score": weighted_score,
                    "graph_distance": graph_result_item.distance if graph_result_item else 99,
                }
            )
            if static_boost.total_boost or memory_boost.total_boost:
                boost_events.append(
                    {
                        "path": path,
                        "boost_value": round(static_boost.total_boost + memory_boost.total_boost, 4),
                        "static_components": static_boost.components,
                        "memory_components": memory_boost.components,
                    }
                )

        pre_rerank_paths = sorted(candidate_paths, key=lambda path: (-fused_scores[path], path))
        pre_rerank_ranks = {path: index for index, path in enumerate(pre_rerank_paths, start=1)}
        reranked_docs, rerank_effects = rerank(
            docs=sorted(candidate_docs, key=lambda item: (-float(item["score"]), item["path"])),
            query=expanded_task.query,
            query_analysis=query_analysis,
        )
        ranked_paths = [str(doc["path"]) for doc in reranked_docs]
        post_rerank_ranks = {path: index for index, path in enumerate(ranked_paths, start=1)}
        top_paths = ranked_paths[:effective_top_k]

        final_results: list[RetrievalScore] = []
        for path in top_paths:
            lexical_result = lexical_result_map.get(path)
            semantic_result = semantic_result_map.get(path)
            reranked_score = next(doc["score"] for doc in reranked_docs if doc["path"] == path)
            scoring_breakdown[path]["rerank_delta"] = next(
                float(doc.get("rerank_delta", 0.0)) for doc in reranked_docs if doc["path"] == path
            )
            scoring_breakdown[path]["rerank_signals"] = next(
                dict(doc.get("rerank_signals", {})) for doc in reranked_docs if doc["path"] == path
            )
            scoring_breakdown[path]["final_score"] = reranked_score
            final_results.append(
                RetrievalScore(
                    path=path,
                    content=self._document_map[path],
                    total_score=round(reranked_score, 6),
                    bm25_score=round(lexical_result.bm25_score if lexical_result else 0.0, 6),
                    tfidf_score=round(lexical_result.tfidf_score if lexical_result else 0.0, 6),
                    keyword_score=round(lexical_result.keyword_score if lexical_result else 0.0, 6),
                    filename_score=round(lexical_result.filename_score if lexical_result else 0.0, 6),
                    domain_score=round(semantic_result.similarity if semantic_result else 0.0, 6),
                    matched_terms=tuple(keyword_hits[path]),
                )
            )

        rerank_effect = [
            {
                "path": effect.path,
                "before": effect.before_rank,
                "after": effect.after_rank,
                "delta": effect.delta,
                "signals": effect.signal_breakdown,
            }
            for effect in rerank_effects[:candidate_depth]
            if effect.before_rank != effect.after_rank or effect.delta > 0
        ]
        metadata = {
            "strategy": "hybrid_retrieval_v3_adaptive",
            "fusion_method": "adaptive_weighted_fusion",
            "query_terms": sorted(set(tokenize_text(task.query))),
            "expanded_query_terms": expansion_terms,
            "graph_expansion_terms": graph_expansion_terms,
            "files_considered": [document.path for document in self._documents],
            "scores": {
                path: round(next(doc["score"] for doc in reranked_docs if doc["path"] == path), 6)
                for path in top_paths
            },
            "ranking": [
                {
                    "rank": index,
                    "path": path,
                    "score": round(next(doc["score"] for doc in reranked_docs if doc["path"] == path), 6),
                }
                for index, path in enumerate(top_paths, start=1)
            ],
            "top_k": top_paths,
            "keyword_hits": {path: keyword_hits[path] for path in top_paths},
            "scoring_breakdown": {path: scoring_breakdown[path] for path in top_paths},
            "bm25_ranking": [
                {"rank": index, "path": result.path, "score": result.total_score}
                for index, result in enumerate(lexical_results[:candidate_depth], start=1)
            ],
            "semantic_ranking": [
                {"rank": result.rank, "path": result.path, "score": round(result.similarity, 6)}
                for result in semantic_results[:candidate_depth]
            ],
            "graph_ranking": graph_result.metadata.get("graph_scores", []),
            "fusion_scores": [{"path": path, "score": round(fused_scores[path], 6)} for path in top_paths],
            "final_ranking": [
                {
                    "rank": index,
                    "path": path,
                    "score": round(next(doc["score"] for doc in reranked_docs if doc["path"] == path), 6),
                }
                for index, path in enumerate(top_paths, start=1)
            ],
            "embedding_similarity": {
                path: round(semantic_result_map[path].similarity, 6)
                for path in top_paths
                if path in semantic_result_map
            },
            "semantic_cache_size": len(self._semantic_documents),
            "lexical_weights": lexical_metadata.get("weights", {}),
            "bm25_params": lexical_metadata.get("bm25_params", {}),
            "boosted_docs": boost_events,
            "boost_effect": [
                {
                    "path": path,
                    "before_rank": pre_rerank_ranks.get(path),
                    "after_rank": post_rerank_ranks.get(path),
                }
                for path in top_paths
                if pre_rerank_ranks.get(path) != post_rerank_ranks.get(path)
            ],
            "memory_hit": bool(memory_entry.get("successful_docs") or memory_entry.get("failed_docs")),
            "memory_confidence": memory_entry.get("memory_confidence", 0.0),
            "graph_nodes": graph_result.metadata.get("graph_nodes", []),
            "graph_paths": graph_result.metadata.get("graph_paths", []),
            "graph_scores": graph_result.metadata.get("graph_scores", []),
            "traversal_depth": graph_result.metadata.get("traversal_depth", 0),
            "graph_score_weight": self._graph_retriever.graph_score_weight,
            "graph_size": graph_result.metadata.get("graph_size", {}),
            "graph_example": graph_result.metadata.get("graph_example", {}),
            "query_type": query_analysis.query_type,
            "query_analysis": query_analysis.as_dict(),
            "intent": query_analysis.intent,
            "weight_distribution": adaptive_weights,
            "candidate_count": len(candidate_paths),
            "rerank_effect": rerank_effect,
            "rerank_top_changes": rerank_effect[:effective_top_k],
        }
        return final_results, metadata

    def update_feedback(
        self,
        task: Task,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Update retrieval memory after evaluation feedback."""

        self._memory.update(
            query=task.query,
            selected_docs=selected_docs,
            retrieval_score=retrieval_score,
            observed_keywords=observed_keywords,
        )

    @staticmethod
    def _build_semantic_index(documents: list[MarkdownDocument]) -> list[SemanticDocument]:
        semantic_documents: list[SemanticDocument] = []
        for document in documents:
            semantic_documents.append(
                SemanticDocument(
                    path=document.path,
                    content=document.content,
                    embedding=tuple(compute_embedding(f"{Path(document.path).stem} {document.content}")),
                )
            )
        return semantic_documents

    def _semantic_rank_documents(self, task: Task, top_k: int) -> list[SemanticScore]:
        query_parts = [task.query, *(task.context_tags or []), *(Path(path).stem for path in (task.files_changed or []))]
        query_embedding = compute_embedding(" ".join(part for part in query_parts if part).strip())

        scores: list[SemanticScore] = []
        for document in self._semantic_documents:
            similarity = compute_cosine_similarity(query_embedding, list(document.embedding))
            if similarity <= 0:
                continue
            scores.append(
                SemanticScore(
                    path=document.path,
                    content=document.content,
                    similarity=round(similarity, 8),
                    rank=0,
                )
            )

        scores.sort(key=lambda item: (-item.similarity, item.path))
        ranked_scores: list[SemanticScore] = []
        for index, item in enumerate(scores[:top_k], start=1):
            ranked_scores.append(
                SemanticScore(
                    path=item.path,
                    content=item.content,
                    similarity=item.similarity,
                    rank=index,
                )
            )
        return ranked_scores

    def _expand_task(
        self,
        task: Task,
        query_analysis: QueryAnalysis,
    ) -> tuple[Task, list[str], list[str]]:
        memory_terms = self._memory.expand_query(task.query)
        graph_terms = self._expand_from_graph(query_analysis)
        expansion_terms = _dedupe_terms(memory_terms + graph_terms)
        if not expansion_terms:
            return task, [], []
        expanded_query = f"{task.query} {' '.join(expansion_terms)}".strip()
        return (
            Task(
                query=expanded_query,
                files_changed=task.files_changed,
                context_tags=task.context_tags,
            ),
            memory_terms,
            graph_terms,
        )

    def _expand_from_graph(self, query_analysis: QueryAnalysis) -> list[str]:
        if not (query_analysis.is_structural_query or query_analysis.is_long_query or query_analysis.is_abstract_query):
            return []
        related_terms: list[str] = []
        for term in query_analysis.unique_tokens:
            concept_paths = self._graph_retriever.graph.concept_index.get(term, ())
            if not concept_paths:
                continue
            for path in concept_paths[:3]:
                related_terms.extend(tokenize_text(Path(path).stem))
        return _dedupe_terms(related_terms)[:4]

    def _rrf_score(self, lexical_rank: int | None, semantic_rank: int | None) -> float:
        score = 0.0
        if lexical_rank is not None:
            score += 1.0 / (self._rrf_k + lexical_rank)
        if semantic_rank is not None:
            score += 1.0 / (self._rrf_k + semantic_rank)
        return score


def _normalize_component_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: round(value / max_value, 6) for key, value in values.items()}


def _dedupe_terms(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
