"""Minimal skill guidance for the prompt compiler."""

from __future__ import annotations

import functools
import re
from pathlib import Path

from .query_analyzer import QueryAnalysis

_CORE_SKILLS = (
    "context-fundamentals",
    "context-degradation",
    "prompt-engineering",
    "systematic-debugging",
    "architecture",
)

_OPTIONAL_SKILLS = {
    "build": ("ai-engineer",),
    "backend": ("cc-skill-backend-patterns",),
    "database": ("database-design",),
}

_GRAPH_SKILL_RULES = (
    ({"api", "auth", "authentication", "authorization", "backend", "endpoint", "jwt", "login", "oauth", "role", "session", "token"}, "backend"),
    ({"data", "database", "db", "schema", "sql", "table"}, "database"),
    ({"build", "implement", "support", "upgrade"}, "build"),
)

_FILE_SKILL_RULES = (
    ({"api", "auth", "backend"}, "backend"),
    ({"database", "db", "schema", "sql"}, "database"),
)

_QUERY_SKILL_RULES = (
    ({"build", "create", "implement", "support", "upgrade"}, "build"),
    ({"api", "auth", "authentication", "backend", "endpoint", "jwt", "login", "role", "session", "token"}, "backend"),
    ({"data", "database", "db", "schema", "sql", "table"}, "database"),
)

_MAX_HEURISTICS = 5
_MAX_HEURISTIC_CHARS = 120


def select_skill_guidance(
    analysis: QueryAnalysis,
    retrieval_metadata: dict[str, object] | None = None,
) -> list[str]:
    """Return compact compiler guidance with graph > files > query priority."""

    retrieval_metadata = retrieval_metadata or {}
    selected_skills: list[str] = list(_CORE_SKILLS)

    graph_terms = _extract_graph_terms(retrieval_metadata)
    file_terms = _extract_file_terms(retrieval_metadata)
    query_terms = set(analysis.unique_tokens)

    selected_skills.extend(_skills_from_terms(graph_terms, _GRAPH_SKILL_RULES))
    selected_skills.extend(_skills_from_terms(file_terms, _FILE_SKILL_RULES))
    selected_skills.extend(_skills_from_terms(query_terms, _QUERY_SKILL_RULES))

    if analysis.task_mode == "debug":
        selected_skills = [skill for skill in selected_skills if skill != "ai-engineer"] + ["systematic-debugging"]

    heuristics: list[str] = []
    seen: set[str] = set()
    for skill_name in _dedupe(selected_skills):
        for item in _extract_skill_heuristics(skill_name):
            normalized = item.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            heuristics.append(item)
            if len(heuristics) >= _MAX_HEURISTICS:
                return heuristics
    return heuristics


def _skills_from_terms(terms: set[str], rules: tuple[tuple[set[str], str], ...]) -> list[str]:
    skills: list[str] = []
    for triggers, optional_key in rules:
        if terms.intersection(triggers):
            skills.extend(_OPTIONAL_SKILLS[optional_key])
    return skills


@functools.lru_cache(maxsize=None)
def _extract_skill_heuristics(skill_name: str) -> tuple[str, ...]:
    skill_path = _resolve_skill_path(skill_name)
    if skill_path is None or not skill_path.is_file():
        return ()

    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError:
        return ()

    lines = [line.strip() for line in content.splitlines()]
    candidates: list[str] = []
    capture = False
    for line in lines:
        if line.startswith("## ") and any(
            marker in line.lower()
            for marker in ("instructions", "guidelines", "core principle", "overview", "safety")
        ):
            capture = True
            continue
        if line.startswith("## "):
            capture = False
        if not capture:
            continue
        if line.startswith("- "):
            normalized = _normalize_heuristic(line[2:])
            if normalized:
                candidates.append(normalized)
        elif line and not line.startswith("#") and len(candidates) < 2:
            normalized = _normalize_heuristic(line)
            if normalized:
                candidates.append(normalized)

    if not candidates:
        description = _frontmatter_description(content)
        if description:
            candidates.append(description)

    return tuple(candidates[:2])


def _resolve_skill_path(skill_name: str) -> Path | None:
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        skill_file = parent / ".skills" / skill_name / "SKILL.md"
        if skill_file.is_file():
            return skill_file
    return None


def _normalize_heuristic(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -*\t")
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    sentence = sentence[:_MAX_HEURISTIC_CHARS].rstrip()
    last_space = sentence.rfind(" ")
    if len(sentence) == _MAX_HEURISTIC_CHARS and last_space > 60:
        sentence = sentence[:last_space].rstrip()
    sentence = sentence.rstrip(".,;: ")
    return sentence


def _frontmatter_description(content: str) -> str:
    match = re.search(r"^description:\s*\"?(.*?)\"?$", content, flags=re.MULTILINE)
    return _normalize_heuristic(match.group(1)) if match else ""


def _extract_graph_terms(retrieval_metadata: dict[str, object]) -> set[str]:
    terms: set[str] = set()
    for item in retrieval_metadata.get("graph_scores", []):
        if not isinstance(item, dict):
            continue
        for concept in item.get("matched_concepts", []):
            normalized = str(concept).strip().lower()
            if normalized:
                terms.add(normalized)
    return terms


def _extract_file_terms(retrieval_metadata: dict[str, object]) -> set[str]:
    terms: set[str] = set()
    for raw_path in retrieval_metadata.get("top_k", []):
        stem = Path(str(raw_path)).stem.lower()
        for token in re.split(r"[^a-z0-9]+", stem):
            token = token.strip()
            if token:
                terms.add(token)
    return terms


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped
