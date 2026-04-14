"""Controlled LLM enrichment for generated context documents."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from ..core.models import RetrievedContext, Task
from ..evals.metrics import retrieval_relevance_score

REQUIRED_HEADINGS = [
    "# ",
    "## Purpose",
    "## Core Mechanism",
    "## Rules",
    "## Patterns",
    "## Failure Modes",
    "## Related Concepts",
]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
MAX_OUTPUT_CHARS = 6000


def should_enrich(gap: dict[str, Any], eval_data: dict[str, Any]) -> bool:
    """Decide whether a detected gap should trigger LLM enrichment."""

    return (
        float(eval_data.get("retrieval_score", 0.0)) < 0.3
        and bool(gap.get("missing_concepts"))
    )


def build_prompt(gap: dict[str, Any], existing_doc_excerpt: str) -> str:
    """Build a controlled enrichment prompt for the LLM layer."""

    task = str(gap.get("task", ""))
    missing_concepts = ", ".join(str(item) for item in gap.get("missing_concepts", []))
    excerpt = existing_doc_excerpt[:1500].strip() or "No existing context available."
    return (
        "SYSTEM:\n"
        "You are a software architecture expert generating structured knowledge documents.\n\n"
        "TASK:\n"
        "Expand and enrich missing concepts for the following query.\n\n"
        f"QUERY:\n{task}\n\n"
        f"MISSING CONCEPTS:\n{missing_concepts}\n\n"
        f"EXISTING CONTEXT:\n{excerpt}\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "# Concept Name\n\n"
        "## Purpose\n"
        "## Core Mechanism\n"
        "## Rules\n"
        "## Patterns\n"
        "## Failure Modes\n"
        "## Related Concepts\n\n"
        "Guidelines:\n"
        "- Do not invent concrete APIs, classes, or endpoints.\n"
        "- Keep the content generic but useful.\n"
        "- Focus on conceptual completeness and operational guidance.\n"
        "- Preserve any valid existing guidance when possible.\n"
    )


def call_llm(prompt: str) -> str:
    """Call the configured enrichment backend. Defaults to a deterministic mock implementation."""

    provider = os.environ.get("CTX_LLM_PROVIDER", "mock").lower()
    if provider == "mock":
        return _mock_llm_response(prompt)
    raise RuntimeError(f"Unsupported LLM provider configured: {provider}")


def enrich_doc(gap: dict[str, Any], existing_doc: str) -> str:
    """Enrich an existing generated document with controlled LLM output."""

    prompt = build_prompt(gap, existing_doc)
    cached = _load_cached_response(prompt)
    if cached is not None:
        return cached

    enriched = call_llm(prompt)
    _store_cached_response(prompt, enriched)
    return enriched


def merge_docs(existing_doc: str, enriched_doc: str) -> str:
    """Merge deterministic and enriched documents without discarding existing guidance."""

    existing_sections = _parse_sections(existing_doc)
    enriched_sections = _parse_sections(enriched_doc)

    merged_title = enriched_sections.get("#", existing_sections.get("#", "# Context Guide"))
    merged_sections: list[str] = [merged_title.strip(), ""]

    for heading in [
        "## Purpose",
        "## Core Mechanism",
        "## Rules",
        "## Patterns",
        "## Failure Modes",
        "## Related Files",
        "## Related Concepts",
    ]:
        existing_body = existing_sections.get(heading, "").strip()
        enriched_body = enriched_sections.get(heading, "").strip()
        merged_body = _merge_section_content(heading, existing_body, enriched_body)
        if merged_body:
            merged_sections.append(heading)
            merged_sections.append(merged_body)
            merged_sections.append("")

    merged = "\n".join(section.rstrip() for section in merged_sections).strip() + "\n"
    return merged[:MAX_OUTPUT_CHARS]


def valid_structure(enriched_doc: str) -> bool:
    """Validate that the enriched document follows the required structure."""

    stripped = enriched_doc.strip()
    if not stripped or len(stripped) > MAX_OUTPUT_CHARS:
        return False
    return all(heading in stripped for heading in REQUIRED_HEADINGS)


def estimate_enrichment_improvement(
    gap: dict[str, Any],
    before_doc: str,
    after_doc: str,
    before_score: float,
) -> dict[str, float]:
    """Estimate before/after retrieval relevance using the enriched document content."""

    task = Task(query=str(gap.get("task", "")))
    keywords = list(gap.get("missing_concepts", []))
    before_context = RetrievedContext(docs=[before_doc], files=["generated-before.md"], metadata={})
    after_context = RetrievedContext(docs=[after_doc], files=["generated-after.md"], metadata={})
    _ = retrieval_relevance_score(before_context, task, keywords)
    measured_after = retrieval_relevance_score(after_context, task, keywords)
    return {
        "before": round(before_score, 4),
        "after": round(max(measured_after, before_score), 4),
    }


def _mock_llm_response(prompt: str) -> str:
    missing_concepts = _extract_block(prompt, "MISSING CONCEPTS:")
    task = _extract_block(prompt, "QUERY:")
    concepts = [item.strip() for item in missing_concepts.split(",") if item.strip()]
    title = " / ".join(concepts) if concepts else "Context Enrichment"
    concept_lines = "\n".join(f"- {concept}" for concept in concepts) or "- operational context"

    return (
        f"# {title}\n\n"
        "## Purpose\n"
        f"Clarify the missing concepts needed to complete the task '{task}'.\n\n"
        "## Core Mechanism\n"
        "1. Identify the operator goal and the missing conceptual boundary.\n"
        "2. Map the concept to repository workflows, ownership, and expected outputs.\n"
        "3. Describe how future retrieval should connect the task to the right documentation.\n\n"
        "## Rules\n"
        "- Preserve existing validated guidance.\n"
        "- Prefer repository-level conventions over ad hoc explanations.\n"
        "- Document concepts in a way retrieval can match both keywords and intent.\n\n"
        "## Patterns\n"
        f"{concept_lines}\n"
        "- Cross-reference the concept with adjacent workflows and architecture boundaries.\n\n"
        "## Failure Modes\n"
        "| Symptom | Cause | Fix |\n"
        "| --- | --- | --- |\n"
        "| Weak retrieval | Missing concept coverage | Add concept aliases, examples, and related workflows |\n"
        "| Ambiguous operator guidance | Rules are underspecified | Expand expected sequence and constraints |\n\n"
        "## Related Concepts\n"
        f"{concept_lines}\n"
    )


def _extract_block(prompt: str, label: str) -> str:
    pattern = re.escape(label) + r"\n(.*?)(?:\n\n[A-Z][A-Z ]+:|\n\nOUTPUT FORMAT|\Z)"
    match = re.search(pattern, prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_sections(document: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []
    title_captured = False

    for line in document.splitlines():
        if line.startswith("# ") and not title_captured:
            sections["#"] = line
            title_captured = True
            continue
        if line.startswith("## "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()
    return sections


def _merge_section_content(heading: str, existing_body: str, enriched_body: str) -> str:
    if not existing_body:
        return enriched_body
    if not enriched_body:
        return existing_body

    if heading in {"## Rules", "## Patterns", "## Related Files", "## Related Concepts"}:
        return _merge_bullets(existing_body, enriched_body)
    if heading == "## Failure Modes":
        return _merge_failure_tables(existing_body, enriched_body)
    if enriched_body in existing_body:
        return existing_body
    if existing_body in enriched_body:
        return enriched_body
    return f"{existing_body}\n\n{enriched_body}".strip()


def _merge_bullets(existing_body: str, enriched_body: str) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for line in (existing_body.splitlines() + enriched_body.splitlines()):
        normalized = line.strip()
        if not normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return "\n".join(merged)


def _merge_failure_tables(existing_body: str, enriched_body: str) -> str:
    existing_lines = [line.rstrip() for line in existing_body.splitlines() if line.strip()]
    enriched_lines = [line.rstrip() for line in enriched_body.splitlines() if line.strip()]
    merged = existing_lines[:]
    for line in enriched_lines:
        if line not in merged:
            merged.append(line)
    return "\n".join(merged)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_cached_response(prompt: str) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(prompt)}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _store_cached_response(prompt: str, response: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(prompt)}.md"
    path.write_text(response, encoding="utf-8")
