"""Selective hybrid retrieval optimised for prompt compilation.

Replaces exhaustive multi-stage retrieval with a focused three-signal
pipeline (lexical BM25, lightweight semantic embeddings, graph proximity)
fused via Reciprocal Rank Fusion (RRF).

Design constraints
------------------
- Single retrieval pass — no recursive expansion loops.
- Query expansion is static (synonym table + agent role terms only).
  No iterative feedback during a single compile() call.
- Top-k cap enforced early; no re-scoring beyond the candidate set.
- Memory and boost signals are additive adjustments — they do NOT trigger
  re-retrieval or additional passes.
- All returned metadata is for internal observability only.

The public surface used by the engine is:
    HybridRetriever.retrieve(query, agent, top_k) -> (results, metadata)
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Task
from .query_analyzer import QueryAnalysis, analyze_query
from .retrieval_boost import apply_boosts, apply_memory_boost, extract_document_metadata
from .retrieval_memory import RetrievalMemory
from .retrieval_v2 import RetrievalScore, rank_documents, tokenize_text
from .graph_retrieval import GraphRetriever
from ..utils.file_loader import MarkdownDocument

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EMBEDDING_DIMENSION = 96
_RRF_K = 60
# Hard ceiling on candidate documents considered before ranking.
# Keeps token budgets predictable and avoids noise from low-signal docs.
_CANDIDATE_LIMIT = 15
_MAX_EXPANSION_TERMS = 6
_EXPANSION_SIMILARITY_THRESHOLD = 0.2
_LOW_FREQUENCY_TERM_THRESHOLD = 1
_MEMORY_BOOST_CAP = 0.35

_GENERIC_GRAPH_TERMS = {
    "behavior", "concept", "connected", "context", "dependency",
    "design", "flow", "meaning", "overview", "path", "quality",
    "relationship", "relevance", "retrieve", "state",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "use", "using", "what", "when", "with",
}

_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "bug":         ["debug", "failure", "fix", "regression"],
    "debug":       ["bug", "failure", "trace", "fix"],
    "error":       ["failure", "bug", "exception"],
    "failure":     ["error", "bug", "regression"],
    "fix":         ["debug", "repair", "stability"],
    "graph":       ["relationship", "dependency", "connected", "path"],
    "improve":     ["optimize", "boost", "quality"],
    "optimize":    ["improve", "boost", "performance", "quality"],
    "performance": ["optimize", "latency", "throughput"],
    "rank":        ["ranking", "relevance", "retrieve"],
    "retrieval":   ["retrieve", "ranking", "relevance", "context"],
    "semantic":    ["meaning", "intent", "concept"],
    "summary":     ["overview", "explain"],
    "sync":        ["state", "network", "client"],
}

# Per-agent retrieval weight profiles (semantic / graph / keyword).
# Used only to adjust fusion weights — no additional retrieval passes.
AGENT_RETRIEVAL_CONFIG: dict[str, dict[str, float]] = {
    "planner_agent":    {"semantic": 0.6, "graph": 0.3, "keyword": 0.1},
    "retrieval_agent":  {"semantic": 0.7, "graph": 0.2, "keyword": 0.1},
    "draft_agent":      {"semantic": 0.8, "graph": 0.1, "keyword": 0.1},
    "review_agent":     {"semantic": 0.5, "graph": 0.4, "keyword": 0.1},
    "validation_agent": {"semantic": 0.4, "graph": 0.5, "keyword": 0.1},
}
DEFAULT_RETRIEVAL_CONFIG: dict[str, float] = {
    "semantic": 0.5, "graph": 0.2, "keyword": 0.3,
}


# ---------------------------------------------------------------------------
# Lightweight static query expansion
# ---------------------------------------------------------------------------

def expand_query_terms(query: str, agent: dict[str, object] | None) -> list[str]:
    """Return a deduplicated list of expansion terms for the query.

    Expansion is static: synonym table + agent role keywords only.
    No LLM calls, no recursive passes.
    """
    tokens = [t for t in tokenize_text(query) if t not in _STOPWORDS]
    synonyms: list[str] = []
    for token in tokens:
        synonyms.extend(_QUERY_EXPANSIONS.get(token, []))

    role_terms: list[str] = []
    if agent:
        role = str(agent.get("role", "")).lower()
        if "code" in role or "architect" in role:
            role_terms.extend(["architecture", "dependency", "module"])
        elif "debug" in role or "issue" in role:
            role_terms.extend(["bug", "error", "trace", "fix"])

    all_terms = _dedupe_terms(tokens + synonyms + role_terms)
    return all_terms[:_MAX_EXPANSION_TERMS]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def compute_embedding(text: str) -> list[float]:
    """Deterministic lightweight character-n-gram embedding (no external model)."""
    vector = [0.0] * _EMBEDDING_DIMENSION
    tokens = _semantic_tokens(text)
    if not tokens:
        return vector

    for token in tokens:
        token_weight = 1.0 + min(len(token), 12) / 12.0
        for ngram_size in (3, 4):
            ngrams = (
                [token] if len(token) < ngram_size
                else [token[i: i + ngram_size] for i in range(len(token) - ngram_size + 1)]
            )
            for ngram in ngrams:
                bucket = _stable_hash(f"{ngram_size}:{ngram}") % _EMBEDDING_DIMENSION
                sign = 1.0 if (_stable_hash(f"sign:{ngram}") % 2 == 0) else -1.0
                vector[bucket] += sign * token_weight / len(ngrams)

    for left, right in zip(tokens, tokens[1:]):
        bucket = _stable_hash(f"bigram:{left}_{right}") % _EMBEDDING_DIMENSION
        vector[bucket] += 1.35

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [round(v / norm, 8) for v in vector]


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(v * v for v in vec1))
    n2 = math.sqrt(sum(v * v for v in vec2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class SemanticDocument:
    path: str
    content: str
    embedding: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class SemanticScore:
    path: str
    content: str
    similarity: float
    rank: int


# ---------------------------------------------------------------------------
# HybridRetriever — the single public class used by the engine
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Selective hybrid retriever combining lexical, semantic, and graph signals.

    Single-pass design:
      1. Expand query terms (static synonyms + agent role keywords).
      2. Run lexical (BM25), semantic (embedding cosine), and graph signals
         in parallel — one pass each.
      3. Normalize and fuse with RRF + agent-weight adjustment.
      4. Apply memory and static boost as additive score corrections.
      5. Return top-k results.

    No retry loops.  No recursive expansion.  No LLM calls.
    """

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
        self._document_map = {doc.path: doc.content for doc in self._documents}
        self._document_terms = {
            doc.path: set(tokenize_text(f"{Path(doc.path).stem} {doc.content}"))
            for doc in self._documents
        }
        self._term_doc_freq = self._build_term_document_frequency()
        self._document_metadata = {
            doc.path: extract_document_metadata(
                doc_path=doc.path,
                content=doc.content,
                knowledge_base_root=self._knowledge_base_root,
            )
            for doc in self._documents
        }

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def retrieve(
        self,
        query: str | Task,
        agent: dict[str, object] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalScore], dict[str, object]]:
        """Run selective hybrid retrieval and return (results, metadata).

        Single pass:
          expand → parallel signals → normalize → fuse → boost → top-k
        """
        effective_top_k = top_k or self._default_top_k
        actual_query = query.query if isinstance(query, Task) else str(query)

        # 1. Static query expansion (no feedback loop).
        expansion_terms = expand_query_terms(actual_query, agent)
        expanded_query = (
            f"{actual_query} {' '.join(expansion_terms)}".strip()
            if expansion_terms else actual_query
        )
        retrieval_task = Task(
            query=expanded_query,
            files_changed=query.files_changed if isinstance(query, Task) else None,
            context_tags=query.context_tags if isinstance(query, Task) else None,
        )

        candidate_depth = max(min(effective_top_k * 3, _CANDIDATE_LIMIT), 8)

        # 2. Run BM25, semantic, graph in parallel — one pass.
        lex_results, lex_meta, sem_results, graph_result = self._run_signals(
            retrieval_task, candidate_depth
        )

        # 3. Normalize component scores to [0, 1].
        n_lex = _normalize({r.path: r.total_score for r in lex_results})
        n_sem = _normalize({r.path: r.similarity for r in sem_results})
        n_grp = _normalize({p: s.score for p, s in graph_result.scores.items()})

        # 4. Fuse with agent-profile weights + RRF ranks.
        agent_name = str(agent.get("name", "")) if agent else ""
        weights = AGENT_RETRIEVAL_CONFIG.get(agent_name, DEFAULT_RETRIEVAL_CONFIG)

        all_paths = set(n_lex) | set(n_sem) | set(n_grp)
        lex_rank = {r.path: i + 1 for i, r in enumerate(lex_results)}
        sem_rank = {r.path: i + 1 for i, r in enumerate(sem_results)}

        candidates: list[dict[str, Any]] = []
        memory_entry = self._memory.retrieve_with_memory(actual_query, agent)
        query_analysis = analyze_query(actual_query, context={"memory_entry": memory_entry})

        for path in all_paths:
            l_result = next((r for r in lex_results if r.path == path), None)
            s_result = next((r for r in sem_results if r.path == path), None)

            k_overlap = l_result.keyword_score if l_result else 0.0
            sem_sim = s_result.similarity if s_result else 0.0

            rrf = (
                (1.0 / (self._rrf_k + lex_rank[path]) if path in lex_rank else 0.0)
                + (1.0 / (self._rrf_k + sem_rank[path]) if path in sem_rank else 0.0)
            )
            weighted = (
                weights["semantic"] * n_sem.get(path, 0.0)
                + weights["graph"] * n_grp.get(path, 0.0)
                + weights["keyword"] * n_lex.get(path, 0.0)
            )
            base_score = 0.6 * weighted + 0.4 * rrf

            # 5. Additive boost corrections (memory + static).
            static_boost = apply_boosts(
                path, actual_query,
                self._document_metadata.get(path, {}),
                keyword_overlap=k_overlap,
                semantic_similarity=sem_sim,
            )
            memory_boost = apply_memory_boost(path, actual_query, memory_entry)
            boost_delta = min(
                static_boost.total_boost + memory_boost.total_boost, _MEMORY_BOOST_CAP
            )

            final_score = min(base_score + boost_delta, 1.0)

            candidates.append({
                "path": path,
                "content": self._document_map[path],
                "score": round(final_score, 6),
                "keyword_score": round(n_lex.get(path, 0.0), 6),
                "semantic_score": round(n_sem.get(path, 0.0), 6),
                "graph_score": round(n_grp.get(path, 0.0), 6),
                "matched_terms": tuple(l_result.matched_terms) if l_result else (),
                "lexical_result": l_result,
                "semantic_result": s_result,
            })

        candidates.sort(key=lambda x: -x["score"])
        top_candidates = candidates[:effective_top_k]

        # Convert to RetrievalScore list for engine compatibility.
        final_results: list[RetrievalScore] = []
        for doc in top_candidates:
            lr = doc["lexical_result"]
            final_results.append(
                RetrievalScore(
                    path=doc["path"],
                    content=doc["content"],
                    total_score=doc["score"],
                    bm25_score=round(lr.bm25_score if lr else 0.0, 6),
                    tfidf_score=round(lr.tfidf_score if lr else 0.0, 6),
                    keyword_score=round(doc["keyword_score"], 6),
                    filename_score=round(lr.filename_score if lr else 0.0, 6),
                    domain_score=round(doc["semantic_score"], 6),
                    matched_terms=doc["matched_terms"],
                )
            )

        metadata: dict[str, object] = {
            "query": actual_query,
            "agent": agent_name or "none",
            "expansion_terms": expansion_terms,
            "expanded_query_terms": expansion_terms,
            "expanded_query": expanded_query,
            "weights": weights,
            "candidate_count": len(candidates),
            "top_k": [doc["path"] for doc in top_candidates],
            "files_considered": [doc["path"] for doc in candidates],
            "ranking": [
                {"path": doc["path"], "score": doc["score"]} for doc in top_candidates
            ],
            "scoring_breakdown": {
                doc["path"]: {
                    "bm25_signal": doc["keyword_score"],
                    "semantic_signal": doc["semantic_score"],
                    "graph_signal": doc["graph_score"],
                }
                for doc in top_candidates
            },
            "memory_hit": any(
                apply_memory_boost(doc["path"], actual_query, memory_entry).total_boost > 0
                for doc in top_candidates
            ),
            "memory_match_type": memory_entry.get("memory_match_type", "none"),
            "memory_confidence": float(memory_entry.get("memory_confidence", 0.0)),
            "query_type": query_analysis.query_type,
            "task_mode": query_analysis.task_mode,
            "intent": query_analysis.intent,
            "query_analysis": query_analysis.as_dict(),
            "boost_signal": round(
                sum(
                    min(
                        apply_boosts(
                            doc["path"],
                            actual_query,
                            self._document_metadata.get(doc["path"], {}),
                            keyword_overlap=(
                                doc["lexical_result"].keyword_score
                                if doc["lexical_result"] is not None
                                else 0.0
                            ),
                            semantic_similarity=(
                                doc["semantic_result"].similarity
                                if doc["semantic_result"] is not None
                                else 0.0
                            ),
                        ).total_boost
                        + apply_memory_boost(doc["path"], actual_query, memory_entry).total_boost,
                        _MEMORY_BOOST_CAP,
                    )
                    for doc in top_candidates
                ),
                6,
            ),
            "graph_nodes": graph_result.metadata.get("graph_nodes", []),
            "graph_paths": graph_result.metadata.get("graph_paths", []),
            "graph_scores": graph_result.metadata.get("graph_scores", []),
            "graph_size": graph_result.metadata.get("graph_size", {}),
        }

        return final_results, metadata

    def update_feedback(
        self,
        task: Task,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Persist retrieval feedback into memory for future signal boosting."""
        self._memory.update(
            query=task.query,
            selected_docs=selected_docs,
            retrieval_score=retrieval_score,
            observed_keywords=observed_keywords,
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _run_signals(
        self,
        task: Task,
        candidate_depth: int,
    ) -> tuple[list[RetrievalScore], dict[str, Any], list[SemanticScore], Any]:
        """Run lexical, semantic, and graph retrieval in parallel — one pass each."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            lex_future = executor.submit(rank_documents, self._documents, task, None, candidate_depth)
            sem_future = executor.submit(self._semantic_rank, task, candidate_depth)
            grp_future = executor.submit(self._graph_retriever.retrieve, task.query, task, candidate_depth)
            lex_results, lex_meta = lex_future.result()
            sem_results = sem_future.result()
            graph_result = grp_future.result()
        return lex_results, lex_meta, sem_results, graph_result

    def _semantic_rank(self, task: Task, top_k: int) -> list[SemanticScore]:
        query_parts = [
            task.query,
            *(task.context_tags or []),
            *(Path(p).stem for p in (task.files_changed or [])),
        ]
        query_embedding = compute_embedding(" ".join(p for p in query_parts if p).strip())

        scores: list[SemanticScore] = []
        for doc in self._semantic_documents:
            similarity = compute_cosine_similarity(query_embedding, list(doc.embedding))
            if similarity > 0:
                scores.append(SemanticScore(
                    path=doc.path,
                    content=doc.content,
                    similarity=round(similarity, 8),
                    rank=0,
                ))

        scores.sort(key=lambda item: (-item.similarity, item.path))
        return [
            SemanticScore(path=item.path, content=item.content, similarity=item.similarity, rank=i + 1)
            for i, item in enumerate(scores[:top_k])
        ]

    @staticmethod
    def _build_semantic_index(documents: list[MarkdownDocument]) -> list[SemanticDocument]:
        return [
            SemanticDocument(
                path=doc.path,
                content=doc.content,
                embedding=tuple(compute_embedding(f"{Path(doc.path).stem} {doc.content}")),
            )
            for doc in documents
        ]

    def _build_term_document_frequency(self) -> dict[str, int]:
        tdf: dict[str, int] = {}
        for terms in self._document_terms.values():
            for term in terms:
                tdf[term] = tdf.get(term, 0) + 1
        return tdf


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_val = max(values.values())
    if max_val <= 0:
        return {k: 0.0 for k in values}
    return {k: round(v / max_val, 6) for k, v in values.items()}


def _dedupe_terms(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = str(value).strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def _semantic_tokens(text: str) -> list[str]:
    filtered = [t for t in tokenize_text(text) if t not in _STOPWORDS]
    return filtered or tokenize_text(text)


# Legacy aliases kept for any import sites that reference these names directly.
expand_query = expand_query_terms
filter_docs = lambda docs, agent: docs  # noqa: E731  (no-op filter — removal over modification)
rank_docs = lambda docs, agent: docs    # noqa: E731  (ranking handled inside HybridRetriever)
