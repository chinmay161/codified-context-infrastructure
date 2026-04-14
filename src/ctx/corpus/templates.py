"""Reusable templates for generated context documents."""

from __future__ import annotations

BASE_TEMPLATE = """# {title}

## Purpose
{purpose}

## Core Mechanism
{core_mechanism}

## Rules
{rules}

## Patterns
{patterns}

## Failure Modes
| Symptom | Cause | Fix |
| --- | --- | --- |
{failure_modes}

## Related Files
{related_files}
"""

DOMAIN_HINTS = {
    "auth": {
        "purpose": "Explain authentication flows, security boundaries, and token lifecycle behavior.",
        "rules": [
            "Validate identity inputs before issuing or refreshing tokens.",
            "Do not bypass authorization checks when debugging failures.",
            "Keep credential, token, and session behavior explicitly documented.",
        ],
    },
    "database": {
        "purpose": "Explain persistence behavior, schema ownership, and transactional constraints.",
        "rules": [
            "Document write ordering and consistency assumptions.",
            "Do not change schema contracts without updating migration and access patterns.",
            "Capture failure handling for partial writes and retry behavior.",
        ],
    },
    "workflow": {
        "purpose": "Explain operational flow, handoffs, and repository-level working conventions.",
        "rules": [
            "Document the sequence of steps from trigger to completion.",
            "Do not assume implicit operator knowledge for cross-file workflows.",
            "Capture entry points, outputs, and coordination rules clearly.",
        ],
    },
    "architecture": {
        "purpose": "Explain component boundaries, control flow, and cross-module coordination.",
        "rules": [
            "Document ownership boundaries between layers and modules.",
            "Do not rely on hidden coupling between retrieval, routing, and execution layers.",
            "Capture extension seams for future infrastructure changes.",
        ],
    },
    "general": {
        "purpose": "Explain the missing concept in terms of intent, behavior, and implementation guidance.",
        "rules": [
            "Document the concept with concrete behavior and constraints.",
            "Do not leave critical system assumptions implicit.",
            "Include related files and failure recovery guidance.",
        ],
    },
}


def get_domain_template(domain: str) -> dict[str, object]:
    """Return domain-specific template hints."""

    normalized = domain.lower()
    return DOMAIN_HINTS.get(normalized, DOMAIN_HINTS["general"])
