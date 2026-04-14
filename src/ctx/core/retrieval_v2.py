"""Advanced retrieval and ranking utilities for the context engine."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .models import Agent, Task
from ..utils.file_loader import MarkdownDocument

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase terms."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(slots=True, frozen=True)
class IndexedDocument:
    """Preprocessed markdown document ready for ranking."""

    path: str
    content: str
    filename: str
    tokens: tuple[str, ...]
    term_frequencies: dict[str, int]
    length: int


@dataclass(slots=True, frozen=True)
class CorpusStats:
    """Corpus-level statistics used by TF-IDF and BM25 scoring."""

    document_count: int
    average_document_length: float
    document_frequencies: dict[str, int]


@dataclass(slots=True, frozen=True)
class RetrievalScore:
    """Multi-signal score for a ranked document."""

    path: str
    content: str
    total_score: float
    bm25_score: float
    tfidf_score: float
    keyword_score: float
    filename_score: float
    domain_score: float
    matched_terms: tuple[str, ...]


def index_documents(documents: list[MarkdownDocument]) -> tuple[list[IndexedDocument], CorpusStats]:
    """Build indexed documents and corpus statistics for retrieval."""

    indexed_documents: list[IndexedDocument] = []
    document_frequencies: Counter[str] = Counter()
    total_length = 0

    for document in documents:
        filename = Path(document.path).stem.lower()
        tokens = tuple(tokenize_text(f"{filename} {document.content}"))
        term_frequencies = Counter(tokens)
        unique_terms = set(term_frequencies)
        document_frequencies.update(unique_terms)
        total_length += len(tokens)
        indexed_documents.append(
            IndexedDocument(
                path=document.path,
                content=document.content,
                filename=filename,
                tokens=tokens,
                term_frequencies=dict(term_frequencies),
                length=len(tokens),
            )
        )

    document_count = len(indexed_documents)
    average_document_length = total_length / document_count if document_count else 0.0
    return indexed_documents, CorpusStats(
        document_count=document_count,
        average_document_length=average_document_length,
        document_frequencies=dict(document_frequencies),
    )


def compute_tfidf_score(
    doc_tokens: tuple[str, ...],
    query_tokens: list[str],
    corpus_stats: CorpusStats,
) -> float:
    """Compute a simple TF-IDF score for a document/query pair."""

    if not doc_tokens or not query_tokens or corpus_stats.document_count == 0:
        return 0.0

    term_frequencies = Counter(doc_tokens)
    document_length = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        term_frequency = term_frequencies.get(term, 0)
        if term_frequency == 0:
            continue
        document_frequency = corpus_stats.document_frequencies.get(term, 0)
        idf = math.log((corpus_stats.document_count + 1) / (document_frequency + 1)) + 1.0
        tf = term_frequency / document_length
        score += tf * idf
    return score


def compute_bm25_score(
    doc: IndexedDocument,
    query_tokens: list[str],
    corpus_stats: CorpusStats,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a document/query pair."""

    if not doc.tokens or not query_tokens or corpus_stats.document_count == 0:
        return 0.0

    avg_doc_length = corpus_stats.average_document_length or 1.0
    score = 0.0
    for term in set(query_tokens):
        term_frequency = doc.term_frequencies.get(term, 0)
        if term_frequency == 0:
            continue
        document_frequency = corpus_stats.document_frequencies.get(term, 0)
        idf = math.log(
            1 + ((corpus_stats.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        )
        numerator = term_frequency * (k1 + 1)
        denominator = term_frequency + k1 * (1 - b + b * (doc.length / avg_doc_length))
        score += idf * (numerator / denominator)
    return score


def build_query_tokens(task: Task) -> list[str]:
    """Build a tokenized query profile from task fields."""

    tokens = tokenize_text(task.query)
    for tag in task.context_tags or []:
        tokens.extend(tokenize_text(tag))
    for file_path in task.files_changed or []:
        tokens.extend(tokenize_text(Path(file_path).stem))
    return tokens


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: round(value / max_value, 6) for key, value in values.items()}


def rank_documents(
    documents: list[MarkdownDocument],
    task: Task,
    agent: Agent | dict[str, Any] | None,
    top_k: int = 5,
    bm25_weight: float = 0.35,
    keyword_weight: float = 0.2,
    filename_weight: float = 0.15,
    domain_weight: float = 0.15,
    tfidf_weight: float = 0.15,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[RetrievalScore], dict[str, object]]:
    """Rank documents using BM25, TF-IDF, keyword overlap, filename, and domain signals."""

    indexed_documents, corpus_stats = index_documents(documents)
    query_tokens = build_query_tokens(task)
    query_terms = set(query_tokens)
    
    if agent is None:
        domain_terms = set()
    elif isinstance(agent, dict):
        domain_terms = {term.lower() for term in tokenize_text(str(agent.get("role", "")))}
    else:
        domain_terms = {term.lower() for term in agent.domain}

    files_considered = [document.path for document in indexed_documents]

    if not indexed_documents or not query_terms:
        return [], {
            "strategy": "advanced_ranked_retrieval_v2",
            "query_terms": sorted(query_terms),
            "files_considered": files_considered,
            "scores": {},
            "ranking": [],
            "top_k": [],
            "keyword_hits": {},
            "scoring_breakdown": {},
            "bm25_params": {"k1": k1, "b": b},
        }

    raw_bm25: dict[str, float] = {}
    raw_tfidf: dict[str, float] = {}
    raw_keyword: dict[str, float] = {}
    raw_filename: dict[str, float] = {}
    raw_domain: dict[str, float] = {}
    matched_terms_by_path: dict[str, tuple[str, ...]] = {}

    for document in indexed_documents:
        document_terms = set(document.tokens)
        matched_terms = tuple(sorted(query_terms.intersection(document_terms)))
        matched_terms_by_path[document.path] = matched_terms

        raw_bm25[document.path] = compute_bm25_score(
            doc=document,
            query_tokens=query_tokens,
            corpus_stats=corpus_stats,
            k1=k1,
            b=b,
        )
        raw_tfidf[document.path] = compute_tfidf_score(
            doc_tokens=document.tokens,
            query_tokens=query_tokens,
            corpus_stats=corpus_stats,
        )
        raw_keyword[document.path] = len(matched_terms) / len(query_terms)
        raw_filename[document.path] = len(query_terms.intersection(set(tokenize_text(document.filename)))) / len(query_terms)
        raw_domain[document.path] = len(domain_terms.intersection(document_terms)) / len(domain_terms or {""})

    normalized_bm25 = _normalize_scores(raw_bm25)
    normalized_tfidf = _normalize_scores(raw_tfidf)
    normalized_keyword = _normalize_scores(raw_keyword)
    normalized_filename = _normalize_scores(raw_filename)
    normalized_domain = _normalize_scores(raw_domain)

    results: list[RetrievalScore] = []
    for document in indexed_documents:
        path = document.path
        total_score = (
            bm25_weight * normalized_bm25[path]
            + tfidf_weight * normalized_tfidf[path]
            + keyword_weight * normalized_keyword[path]
            + filename_weight * normalized_filename[path]
            + domain_weight * normalized_domain[path]
        )
        if total_score <= 0:
            continue
        results.append(
            RetrievalScore(
                path=path,
                content=document.content,
                total_score=round(total_score, 6),
                bm25_score=round(normalized_bm25[path], 6),
                tfidf_score=round(normalized_tfidf[path], 6),
                keyword_score=round(normalized_keyword[path], 6),
                filename_score=round(normalized_filename[path], 6),
                domain_score=round(normalized_domain[path], 6),
                matched_terms=matched_terms_by_path[path],
            )
        )

    results.sort(key=lambda item: (-item.total_score, item.path))
    top_results = results[:top_k]

    metadata = {
        "strategy": "advanced_ranked_retrieval_v2",
        "query_terms": sorted(query_terms),
        "files_considered": files_considered,
        "scores": {result.path: result.total_score for result in top_results},
        "ranking": [
            {"rank": index, "path": result.path, "score": result.total_score}
            for index, result in enumerate(top_results, start=1)
        ],
        "top_k": [result.path for result in top_results],
        "keyword_hits": {result.path: list(result.matched_terms) for result in top_results},
        "scoring_breakdown": {
            result.path: {
                "bm25": result.bm25_score,
                "tfidf": result.tfidf_score,
                "keyword": result.keyword_score,
                "filename": result.filename_score,
                "domain": result.domain_score,
                "total": result.total_score,
            }
            for result in top_results
        },
        "bm25_params": {"k1": k1, "b": b},
        "weights": {
            "bm25": bm25_weight,
            "tfidf": tfidf_weight,
            "keyword": keyword_weight,
            "filename": filename_weight,
            "domain": domain_weight,
        },
    }
    return top_results, metadata
