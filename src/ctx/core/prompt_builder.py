"""Prompt compilation for external LLMs (Copilot / Codex)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from .models import ContextDistillation

_OUTPUT_FORMAT_BLOCK = """\
OUTPUT FORMAT (MANDATORY)
Return a single valid JSON object — no markdown fences, no extra text.
Required keys:

{
  "answer": "<string — primary response content>",
  "confidence": <float 0.0-1.0>,
  "sources": ["<file path or doc identifier used>", ...]
}

Responses missing any required key will be treated as invalid."""

_BUDGET_CONFIG = {
    "persona_chars": 1400,
    "template_chars": 900,
    "distillation_chars": 2800,
    "evidence_chars": 2400,
}

_TASK_MODE_INSTRUCTIONS = {
    "debug": (
        "Prompt Strategy: debug\n"
        "Work step-by-step. Start from the strongest evidence, identify likely causes, "
        "and end with a concrete verification path."
    ),
    "explain": (
        "Prompt Strategy: explain\n"
        "Structure the answer around concepts, flow, and examples that are explicitly present in context."
    ),
    "build": (
        "Prompt Strategy: build\n"
        "Prioritize constraints, required inputs, boundaries, and acceptance criteria before proposing work."
    ),
    "refactor": (
        "Prompt Strategy: refactor\n"
        "Compare current and target states, call out tradeoffs, and highlight regression risks."
    ),
    "explore": (
        "Prompt Strategy: explore\n"
        "Provide the best grounded synthesis available and keep uncertainty explicit where context is incomplete."
    ),
}

_CONFIDENCE_INSTRUCTIONS = {
    "high": "Confidence Mode: high\nUse compact scaffolding and minimal examples; rely on the strongest distilled evidence.",
    "medium": "Confidence Mode: medium\nUse structured sections and cite distilled evidence before drawing conclusions.",
    "low": "Confidence Mode: low\nBe extra explicit about uncertainty, grounding limits, and example output structure.",
}


def build_prompt(query: str, context: str, agent: dict[str, object]) -> str:
    """Backward-compatible prompt builder wrapper."""

    prompt, _ = build_adaptive_prompt(
        query=query,
        context=context,
        agent=agent,
        query_analysis={},
        task_mode="explore",
        distillation=ContextDistillation(),
        compile_confidence={"band": "medium", "score": 0.5, "signals": {}},
        skill_guidance=[],
    )
    return prompt


def build_adaptive_prompt(
    query: str,
    context: str,
    agent: dict[str, object],
    query_analysis: dict[str, Any],
    task_mode: str,
    distillation: ContextDistillation,
    compile_confidence: dict[str, Any],
    skill_guidance: list[str],
) -> tuple[str, dict[str, Any]]:
    """Compile an adaptive prompt and return prompt plus budget metadata."""

    if not isinstance(agent, dict):
        raise TypeError(f"agent must be a dict, got {type(agent).__name__}.")
    _validate_agent_keys(agent)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    agent_skills = list(agent["skills"])  # type: ignore[arg-type]
    agent_constraints = list(agent["constraints"])  # type: ignore[arg-type]
    decision_strategy = str(agent["decision_strategy"])
    failure_mode = str(agent["failure_mode"])

    template_block = _clip_section(
        (
            f"{_TASK_MODE_INSTRUCTIONS.get(task_mode, _TASK_MODE_INSTRUCTIONS['explore'])}\n"
            f"{_CONFIDENCE_INSTRUCTIONS.get(str(compile_confidence.get('band', 'medium')).lower(), _CONFIDENCE_INSTRUCTIONS['medium'])}\n"
            f"Task Entities: {', '.join(query_analysis.get('entities', [])) or 'none'}\n"
            f"Task Intent: {query_analysis.get('intent', 'explore')}\n"
            f"Task Mode: {task_mode}\n"
            f"Reasoning Policy: {decision_strategy}\n"
            f"Fallback Policy: {failure_mode}\n"
            f"High-Signal Constraints: {'; '.join(agent_constraints[:2]) if agent_constraints else 'none'}\n"
            f"Relevant Specialties: {', '.join(agent_skills[:2]) if agent_skills else 'none'}"
            + (
                "\nGuiding Heuristics:\n" + "\n".join(f"- {item}" for item in skill_guidance[:5])
                if skill_guidance else ""
            )
        ),
        _BUDGET_CONFIG["template_chars"],
    )

    distillation_block = _build_distillation_block(distillation)
    distillation_block = _clip_section(distillation_block, _BUDGET_CONFIG["distillation_chars"])

    evidence_block, evidence_meta = _budget_evidence(
        context=context,
        evidence_snippets=distillation.evidence_snippets,
        priority_order=distillation.priority_order,
    )

    prompt = (
        f"{_OUTPUT_FORMAT_BLOCK}\n\n"
        "---\n\n"
        f"{template_block}\n\n"
        "---\n\n"
        "DISTILLED CONTEXT\n"
        f"{distillation_block}\n\n"
        "---\n\n"
        "EVIDENCE\n"
        f"{evidence_block}\n\n"
        "---\n\n"
        "TASK\n"
        f"{query.strip()}"
    )
    budget = {
        "config": dict(_BUDGET_CONFIG),
        "original_context_chars": len(context),
        "final_prompt_chars": len(prompt),
        "distillation_chars_used": len(distillation_block),
        "evidence_chars_used": evidence_meta["chars_used"],
        "compression_applied": evidence_meta["compression_applied"] or len(distillation_block) < len(_build_distillation_block(distillation)),
        "dropped_docs": evidence_meta["dropped_docs"],
        "preserved_evidence": evidence_meta["preserved_evidence"],
    }
    return prompt, budget


def build_system_prompt(agent: dict[str, object]) -> str:
    """Construct the system-prompt string for an agent-governed LLM call."""

    if not isinstance(agent, dict):
        raise TypeError(f"agent must be a dict, got {type(agent).__name__}.")
    _validate_agent_keys(agent)

    constitution = load_constitution()
    agent_block = (
        f"You are {agent['name']}. {agent['role']} "
        "Answer using only the supplied context documents. "
        "Be precise and concise. "
        "Return a single valid JSON object conforming to the output format in the prompt. "
        "Do NOT include agent_trace in your response."
    )
    if constitution:
        return f"{constitution}\n\n---\n\n{agent_block}"
    return agent_block


@functools.lru_cache(maxsize=1)
def load_constitution() -> str:
    """Load and cache the contents of ``.Constitution/Constitution.md``."""

    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        constitution_file = candidate / ".Constitution" / "Constitution.md"
        if constitution_file.is_file():
            try:
                return constitution_file.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return ""


_REQUIRED_AGENT_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "role",
        "rules",
        "workflow",
        "tools",
        "skills",
        "constraints",
        "decision_strategy",
        "failure_mode",
    }
)


def _validate_agent_keys(agent: dict[str, object]) -> None:
    missing = _REQUIRED_AGENT_KEYS - agent.keys()
    if missing:
        raise ValueError(
            f"Agent dict is missing required keys: {sorted(missing)}. "
            f"Expected all of: {sorted(_REQUIRED_AGENT_KEYS)}."
        )


def _build_distillation_block(distillation: ContextDistillation) -> str:
    sections = [
        ("Rules", distillation.rule_extractions or ["No explicit rules extracted."]),
        ("API Summary", distillation.api_summary or ["No API-like structures found."]),
        ("Relationships", distillation.relationship_map or ["No explicit relationships extracted."]),
        ("Priority Order", distillation.priority_order or ["No ranked documents were available."]),
        ("Compression Notes", distillation.compression_notes or ["No compression heuristics were needed."]),
    ]
    lines: list[str] = []
    for title, items in sections:
        lines.append(f"{title}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _budget_evidence(
    context: str,
    evidence_snippets: list[str],
    priority_order: list[str],
) -> tuple[str, dict[str, Any]]:
    evidence_lines = list(evidence_snippets)
    dropped_docs: list[str] = []
    compression_applied = False

    while evidence_lines and len("\n".join(evidence_lines)) > _BUDGET_CONFIG["evidence_chars"]:
        compression_applied = True
        removed = evidence_lines.pop()
        dropped_docs.append(removed.split(":", 1)[0])

    if not evidence_lines:
        fallback = context.strip() if context.strip() else "No evidence available."
        evidence_lines = [_clip_section(fallback, _BUDGET_CONFIG["evidence_chars"])]

    if priority_order and len(priority_order) > len(evidence_lines):
        for path in priority_order[len(evidence_lines):]:
            dropped_docs.append(Path(path).name)

    return (
        "\n".join(evidence_lines),
        {
            "chars_used": len("\n".join(evidence_lines)),
            "compression_applied": compression_applied,
            "dropped_docs": list(dict.fromkeys(dropped_docs)),
            "preserved_evidence": list(evidence_lines),
        },
    )


def _clip_section(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: max(limit - 3, 0)].rstrip()}..."
