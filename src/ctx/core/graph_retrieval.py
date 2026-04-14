"""Graph-based retrieval utilities for multi-hop context discovery."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Task
from .query_analyzer import QueryAnalysis, analyze_query
from .retrieval_v2 import build_query_tokens, tokenize_text
from ..utils.file_loader import MarkdownDocument

_MIN_CONCEPT_LENGTH = 3
_MAX_CONCEPTS_PER_DOC = 24
_MAX_DOC_EDGES_PER_DOC = 12
_CONTAINS_WEIGHT = 0.35
_SHARED_KEYWORD_BASE_WEIGHT = 0.25
_SHARED_KEYWORD_SCALE = 0.08
_FILE_REFERENCE_WEIGHT = 0.45
_IMPORT_REFERENCE_WEIGHT = 0.6
_GRAPH_SCORE_WEIGHT = 0.2
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
}


@dataclass(slots=True, frozen=True)
class GraphEdge:
    """A weighted edge in the retrieval graph."""

    source: str
    target: str
    relation: str
    weight: float


@dataclass(slots=True, frozen=True)
class GraphScore:
    """A graph-ranked document result."""

    path: str
    score: float
    distance: int
    connections: int
    paths: tuple[tuple[str, ...], ...]
    matched_concepts: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class TraversalResult:
    """Traversal output capturing discovered nodes and paths."""

    visited_nodes: tuple[str, ...]
    distances: dict[str, int]
    paths: dict[str, tuple[str, ...]]
    edge_paths: dict[str, tuple[dict[str, Any], ...]]
    path_scores: dict[str, float]
    path_counts: dict[str, int]
    traversal_depth: int


@dataclass(slots=True, frozen=True)
class GraphRetrievalResult:
    """Graph retrieval output plus observability metadata."""

    scores: dict[str, GraphScore]
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class DocumentGraph:
    """In-memory graph of documents and concepts."""

    nodes: dict[str, dict[str, Any]]
    edges: tuple[GraphEdge, ...]
    adjacency: dict[str, tuple[GraphEdge, ...]]
    concept_index: dict[str, tuple[str, ...]]
    document_nodes: tuple[str, ...]
    edge_lookup: dict[tuple[str, str], tuple[GraphEdge, ...]]

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable graph view."""

        sampled_nodes = {
            node_id: _compact_node_payload(payload)
            for node_id, payload in list(sorted(self.nodes.items()))[:8]
        }
        return {
            "nodes": sampled_nodes,
            "edges": [
                (edge.source, edge.target, edge.relation, edge.weight)
                for edge in self.edges[:16]
            ],
        }


def build_graph(docs: list[MarkdownDocument]) -> DocumentGraph:
    """Build a lightweight graph from markdown documents."""

    nodes: dict[str, dict[str, Any]] = {}
    adjacency_map: dict[str, list[GraphEdge]] = defaultdict(list)
    edge_lookup: dict[tuple[str, str], list[GraphEdge]] = defaultdict(list)
    concepts_by_doc: dict[str, tuple[str, ...]] = {}
    references_by_doc: dict[str, set[str]] = {}
    import_refs_by_doc: dict[str, set[str]] = {}
    stem_to_path = {Path(doc.path).stem.lower(): doc.path for doc in docs}

    for doc in docs:
        path = doc.path
        doc_concepts = _extract_concepts(doc.content, path)
        concepts_by_doc[path] = doc_concepts
        references_by_doc[path] = _extract_file_references(doc.content, stem_to_path)
        import_refs_by_doc[path] = _extract_import_references(doc.content, stem_to_path)
        nodes[path] = {
            "type": "document",
            "path": path,
            "content": doc.content,
            "filename": Path(path).stem.lower(),
            "concepts": list(doc_concepts),
        }

    concept_index: dict[str, list[str]] = defaultdict(list)
    edges: list[GraphEdge] = []

    for path, concepts in concepts_by_doc.items():
        for concept in concepts:
            concept_id = _concept_node_id(concept)
            concept_index[concept].append(path)
            if concept_id not in nodes:
                nodes[concept_id] = {
                    "type": "concept",
                    "concept": concept,
                }
            _add_bidirectional_edge(
                edges=edges,
                adjacency_map=adjacency_map,
                edge_lookup=edge_lookup,
                source=path,
                target=concept_id,
                relation="contains",
                weight=_CONTAINS_WEIGHT,
            )

    doc_paths = sorted(concepts_by_doc)
    for index, left_path in enumerate(doc_paths):
        left_concepts = set(concepts_by_doc[left_path])
        for right_path in doc_paths[index + 1 :]:
            shared = left_concepts.intersection(concepts_by_doc[right_path])
            if not shared:
                continue
            weight = round(
                _SHARED_KEYWORD_BASE_WEIGHT
                + min(len(shared), 5) * _SHARED_KEYWORD_SCALE,
                4,
            )
            _add_bidirectional_edge(
                edges=edges,
                adjacency_map=adjacency_map,
                edge_lookup=edge_lookup,
                source=left_path,
                target=right_path,
                relation="shared_keyword",
                weight=weight,
            )

    for path, referenced_paths in references_by_doc.items():
        for referenced_path in sorted(referenced_paths):
            _add_bidirectional_edge(
                edges=edges,
                adjacency_map=adjacency_map,
                edge_lookup=edge_lookup,
                source=path,
                target=referenced_path,
                relation="references",
                weight=_FILE_REFERENCE_WEIGHT,
            )

    for path, imported_paths in import_refs_by_doc.items():
        for imported_path in sorted(imported_paths):
            _add_bidirectional_edge(
                edges=edges,
                adjacency_map=adjacency_map,
                edge_lookup=edge_lookup,
                source=path,
                target=imported_path,
                relation="imports",
                weight=_IMPORT_REFERENCE_WEIGHT,
            )

    adjacency = {
        node_id: tuple(
            sorted(
                node_edges,
                key=lambda edge: (-edge.weight, edge.relation, edge.target),
            )
        )
        for node_id, node_edges in adjacency_map.items()
    }
    lookup = {
        pair: tuple(
            sorted(
                pair_edges,
                key=lambda edge: (-edge.weight, edge.relation, edge.target),
            )
        )
        for pair, pair_edges in edge_lookup.items()
    }
    return DocumentGraph(
        nodes=nodes,
        edges=tuple(edges),
        adjacency=adjacency,
        concept_index={concept: tuple(sorted(paths)) for concept, paths in concept_index.items()},
        document_nodes=tuple(sorted(doc_paths)),
        edge_lookup=lookup,
    )


def map_query_to_nodes(query: str, graph: DocumentGraph, task: Task | None = None) -> tuple[str, ...]:
    """Map a query to concept and document nodes in the graph."""

    query_tokens = build_query_tokens(task) if task is not None else tokenize_text(query)
    ranked_concepts = _rank_query_concepts(query_tokens, graph)
    start_nodes: list[str] = []
    for concept in ranked_concepts:
        concept_id = _concept_node_id(concept)
        if concept_id in graph.nodes:
            start_nodes.append(concept_id)

    if not start_nodes:
        fallback_docs = _rank_document_starts(query_tokens, graph)
        start_nodes.extend(fallback_docs[:3])

    return tuple(dict.fromkeys(start_nodes))


def traverse_graph(
    graph: DocumentGraph,
    start_nodes: tuple[str, ...] | list[str],
    depth: int = 2,
) -> TraversalResult:
    """Traverse the graph with BFS and collect node paths."""

    normalized_depth = max(depth, 0)
    queue: deque[tuple[str, int]] = deque()
    distances: dict[str, int] = {}
    paths: dict[str, tuple[str, ...]] = {}
    edge_paths: dict[str, tuple[dict[str, Any], ...]] = {}
    path_scores: dict[str, float] = {}
    path_counts: dict[str, int] = defaultdict(int)

    for node in start_nodes:
        if node not in graph.nodes or node in distances:
            continue
        distances[node] = 0
        paths[node] = (node,)
        edge_paths[node] = ()
        path_scores[node] = 1.0
        path_counts[node] = 1
        queue.append((node, 0))

    while queue:
        current, distance = queue.popleft()
        if distance >= normalized_depth:
            continue
        for edge in graph.adjacency.get(current, ()):
            next_distance = distance + 1
            if next_distance > normalized_depth:
                continue
            candidate_score = round(path_scores.get(current, 1.0) + float(edge.weight), 6)
            path_counts[edge.target] += 1
            if edge.target in distances and distances[edge.target] <= next_distance:
                path_scores[edge.target] = max(path_scores.get(edge.target, 0.0), candidate_score)
                continue
            distances[edge.target] = next_distance
            paths[edge.target] = (*paths[current], edge.target)
            edge_paths[edge.target] = (
                *edge_paths[current],
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                },
            )
            path_scores[edge.target] = candidate_score
            queue.append((edge.target, next_distance))

    ordered_nodes = tuple(sorted(distances, key=lambda node: (distances[node], node)))
    return TraversalResult(
        visited_nodes=ordered_nodes,
        distances=distances,
        paths=paths,
        edge_paths=edge_paths,
        path_scores=path_scores,
        path_counts={node: int(count) for node, count in path_counts.items()},
        traversal_depth=normalized_depth,
    )


def score_graph_nodes(
    graph: DocumentGraph,
    traversal: TraversalResult,
    start_nodes: tuple[str, ...] | list[str],
    query: str,
    task: Task | None = None,
    query_analysis: QueryAnalysis | None = None,
) -> dict[str, GraphScore]:
    """Score document nodes discovered during traversal."""

    analysis = query_analysis or analyze_query(query)
    query_tokens = set(build_query_tokens(task) if task is not None else tokenize_text(query))
    start_set = set(start_nodes)
    scores: dict[str, GraphScore] = {}

    for node in traversal.visited_nodes:
        node_data = graph.nodes.get(node, {})
        if node_data.get("type") != "document":
            continue

        distance = traversal.distances.get(node, 0)
        path_edges = traversal.edge_paths.get(node, ())
        edge_weight_total = sum(float(edge["weight"]) for edge in path_edges)
        connection_weight = round(
            edge_weight_total / max(len(path_edges), 1),
            6,
        )
        matched_concepts = tuple(
            sorted(
                concept
                for concept in node_data.get("concepts", [])
                if concept in query_tokens
            )
        )
        direct_match_bonus = 0.15 if node in start_set else 0.0
        matched_concept_bonus = 0.05 * min(len(matched_concepts), 4)
        connection_bonus = 0.02 * min(len(graph.adjacency.get(node, ())), 5)
        centrality_bonus = 0.015 * min(len(graph.adjacency.get(node, ())), 8)
        multi_path_bonus = 0.03 * min(max(traversal.path_counts.get(node, 1) - 1, 0), 3)
        path_score_bonus = 0.04 * min(traversal.path_scores.get(node, 0.0), 3.0)
        structural_bonus = 0.08 if analysis.is_structural_query and matched_concepts else 0.0
        score = round(
            (1.0 / (distance + 1))
            + connection_weight
            + direct_match_bonus
            + matched_concept_bonus
            + connection_bonus,
            8,
        )
        score = round(
            score
            + centrality_bonus
            + multi_path_bonus
            + path_score_bonus
            + structural_bonus,
            8,
        )
        scores[node] = GraphScore(
            path=node,
            score=score,
            distance=distance,
            connections=len(graph.adjacency.get(node, ())),
            paths=(traversal.paths.get(node, (node,)),),
            matched_concepts=matched_concepts,
        )

    return scores


class GraphRetriever:
    """Cached graph retriever for lightweight GraphRAG-style expansion."""

    def __init__(
        self,
        documents: list[MarkdownDocument],
        traversal_depth: int = 2,
        graph_score_weight: float = _GRAPH_SCORE_WEIGHT,
    ) -> None:
        self._documents = list(documents)
        self._document_map = {document.path: document.content for document in self._documents}
        self._traversal_depth = max(traversal_depth, 0)
        self._graph_score_weight = max(graph_score_weight, 0.0)
        self._graph = build_graph(self._documents)

    @property
    def graph(self) -> DocumentGraph:
        return self._graph

    @property
    def graph_score_weight(self) -> float:
        return self._graph_score_weight

    def retrieve(
        self,
        query: str,
        task: Task | None = None,
        top_k: int = 5,
    ) -> GraphRetrievalResult:
        """Retrieve graph-ranked documents and observability metadata."""

        query_analysis = analyze_query(
            query=query,
            context={"graph_terms": list(self._graph.concept_index.keys())},
        )
        start_nodes = map_query_to_nodes(query=query, graph=self._graph, task=task)
        traversal_depth = self.adaptive_traverse(query_analysis)
        traversal = traverse_graph(
            graph=self._graph,
            start_nodes=start_nodes,
            depth=traversal_depth,
        )
        scores = score_graph_nodes(
            graph=self._graph,
            traversal=traversal,
            start_nodes=start_nodes,
            query=query,
            task=task,
            query_analysis=query_analysis,
        )
        ranked = sorted(scores.values(), key=lambda item: (-item.score, item.distance, item.path))
        selected = ranked[:top_k]

        metadata = {
            "graph_nodes": list(start_nodes),
            "graph_paths": [
                {
                    "path": result.path,
                    "nodes": list(result.paths[0]),
                    "edges": list(traversal.edge_paths.get(result.path, ())),
                    "distance": result.distance,
                    "path_score": round(traversal.path_scores.get(result.path, 0.0), 6),
                    "path_count": traversal.path_counts.get(result.path, 0),
                }
                for result in selected
            ],
            "graph_scores": [
                {
                    "path": result.path,
                    "score": round(result.score, 6),
                    "distance": result.distance,
                    "connections": result.connections,
                    "matched_concepts": list(result.matched_concepts),
                    "path_score": round(traversal.path_scores.get(result.path, 0.0), 6),
                    "path_count": traversal.path_counts.get(result.path, 0),
                }
                for result in selected
            ],
            "traversal_depth": traversal_depth,
            "graph_size": {
                "nodes": len(self._graph.nodes),
                "edges": len(self._graph.edges),
                "documents": len(self._graph.document_nodes),
                "concepts": sum(1 for node in self._graph.nodes.values() if node.get("type") == "concept"),
            },
            "graph_example": self._graph.as_dict(),
            "query_type": query_analysis.query_type,
        }
        return GraphRetrievalResult(
            scores={result.path: result for result in selected},
            metadata=metadata,
        )

    def adaptive_traverse(self, query_analysis: QueryAnalysis) -> int:
        """Choose traversal depth from query characteristics."""

        if query_analysis.is_structural_query:
            return min(max(self._traversal_depth + 1, 2), 3)
        if query_analysis.is_long_query or query_analysis.is_abstract_query:
            return min(max(self._traversal_depth, 1), 2)
        return min(self._traversal_depth, 1) if self._traversal_depth > 0 else 0


def _extract_concepts(content: str, path: str) -> tuple[str, ...]:
    tokens = tokenize_text(f"{Path(path).stem} {content}")
    counts = Counter(
        token
        for token in tokens
        if len(token) >= _MIN_CONCEPT_LENGTH and not token.isdigit() and token not in _STOPWORDS
    )
    concepts = [
        token
        for token, _ in counts.most_common(_MAX_CONCEPTS_PER_DOC)
    ]
    return tuple(sorted(set(concepts)))


def _extract_file_references(content: str, stem_to_path: dict[str, str]) -> set[str]:
    references: set[str] = set()
    for raw_match in tokenize_text(content):
        stem = raw_match.lower()
        if stem in stem_to_path:
            references.add(stem_to_path[stem])
    return references


def _extract_import_references(content: str, stem_to_path: dict[str, str]) -> set[str]:
    references: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not (lowered.startswith("import ") or lowered.startswith("from ") or "::" in stripped):
            continue
        for token in tokenize_text(stripped):
            if token in stem_to_path:
                references.add(stem_to_path[token])
    return references


def _rank_query_concepts(query_tokens: list[str], graph: DocumentGraph) -> list[str]:
    concept_scores: dict[str, float] = {}
    query_set = {token for token in query_tokens if token not in _STOPWORDS}
    for concept, paths in graph.concept_index.items():
        if concept not in query_set:
            continue
        concept_scores[concept] = len(paths)
    return sorted(concept_scores, key=lambda concept: (-concept_scores[concept], concept))


def _rank_document_starts(query_tokens: list[str], graph: DocumentGraph) -> list[str]:
    query_set = {token for token in query_tokens if token not in _STOPWORDS}
    scored: list[tuple[float, str]] = []
    for path in graph.document_nodes:
        concepts = set(graph.nodes.get(path, {}).get("concepts", []))
        overlap = len(query_set.intersection(concepts))
        if overlap <= 0:
            continue
        scored.append((overlap, path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, path in scored]


def _concept_node_id(concept: str) -> str:
    return f"concept::{concept}"


def _add_bidirectional_edge(
    edges: list[GraphEdge],
    adjacency_map: dict[str, list[GraphEdge]],
    edge_lookup: dict[tuple[str, str], list[GraphEdge]],
    source: str,
    target: str,
    relation: str,
    weight: float,
) -> None:
    if source == target:
        return
    for left, right in ((source, target), (target, source)):
        edge = GraphEdge(
            source=left,
            target=right,
            relation=relation,
            weight=round(weight, 4),
        )
        edges.append(edge)
        adjacency_map[left].append(edge)
        edge_lookup[(left, right)].append(edge)


def _compact_node_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    content = compact.get("content")
    if isinstance(content, str):
        compact["content"] = content[:120]
    return compact
