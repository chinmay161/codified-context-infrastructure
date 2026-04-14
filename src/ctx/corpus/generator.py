"""Context-document generation from detected corpus gaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .templates import BASE_TEMPLATE, get_domain_template


def generate_context_doc(gap: dict[str, Any], code_context: dict[str, Any]) -> str:
    """Generate a structured markdown context document for a detected gap."""

    title = _build_title(gap)
    domain = str(gap.get("expected_domain", "general"))
    template_hints = get_domain_template(domain)
    related_files = _select_related_files(gap, code_context)
    module_patterns = code_context.get("patterns", [])

    purpose = (
        f"{template_hints['purpose']} This document closes the retrieval gap for the task "
        f"'{gap.get('task', '')}'."
    )
    core_mechanism = "\n".join(
        [
            "1. Identify the triggering task or user intent that requires this knowledge.",
            "2. Map the concept to the relevant modules, files, and documented system boundaries.",
            "3. Explain the expected flow, key rules, and recovery guidance for future retrieval.",
        ]
    )
    rules = "\n".join(f"- {rule}" for rule in template_hints["rules"])
    patterns = "\n".join(
        [
            f"- Missing concepts: {', '.join(gap.get('missing_concepts', []))}",
            f"- Known repository patterns: {', '.join(module_patterns) if module_patterns else 'No dominant patterns detected'}",
            f"- Expected domain: {domain}",
        ]
    )
    failure_modes = "\n".join(
        [
            "| Retrieval failure | Missing or weak context coverage | Add or refresh this document with concept-specific examples |",
            "| Routing uncertainty | Domain context is underspecified | Clarify ownership, boundaries, and related modules |",
            "| Outdated implementation guidance | Code and docs drifted apart | Version this document and update related file references |",
        ]
    )
    related_files_section = "\n".join(f"- {file_path}" for file_path in related_files) or "- No related files identified"

    return BASE_TEMPLATE.format(
        title=title,
        purpose=purpose,
        core_mechanism=core_mechanism,
        rules=rules,
        patterns=patterns,
        failure_modes=failure_modes,
        related_files=related_files_section,
    )


def _build_title(gap: dict[str, Any]) -> str:
    missing_concepts = gap.get("missing_concepts", [])
    if missing_concepts:
        return f"{' / '.join(str(concept) for concept in missing_concepts)} Context Guide"
    return f"{gap.get('task', 'Context')} Guide"


def _select_related_files(gap: dict[str, Any], code_context: dict[str, Any], limit: int = 5) -> list[str]:
    concept_terms = {
        token.lower()
        for concept in gap.get("missing_concepts", [])
        for token in str(concept).replace("-", " ").split()
        if token
    }
    scored_files: list[tuple[int, str]] = []
    for module in code_context.get("modules", []):
        path = str(module.get("path", ""))
        haystack = " ".join(
            [
                path,
                " ".join(module.get("functions", [])),
                " ".join(module.get("classes", [])),
                " ".join(module.get("imports", [])),
            ]
        ).lower()
        score = sum(1 for term in concept_terms if term in haystack)
        if score > 0:
            scored_files.append((score, path))

    scored_files.sort(key=lambda item: (-item[0], item[1]))
    selected = [path for _, path in scored_files[:limit]]
    if selected:
        return selected

    markdown_files = [str(path) for path in code_context.get("markdown_files", [])]
    return markdown_files[:limit]
