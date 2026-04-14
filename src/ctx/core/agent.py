"""Structured agent definitions and loader for the context engine.

This module defines the canonical agent registry and exposes ``load_agent``,
which returns the full validated agent object for a given name. All agents
conform to the schema mandated in ``.Constitution/Constitution.md``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent registry
# Each entry MUST contain:
# name, role, rules, workflow, tools, skills, constraints,
# decision_strategy, failure_mode.
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: dict[str, dict[str, object]] = {
    "planner_agent": {
        "name": "planner_agent",
        "role": (
            "Decomposes an incoming task into an ordered execution plan, selects "
            "the most appropriate specialist agent, and defines the retrieval scope."
        ),
        "rules": [
            "Always emit a step-by-step plan before delegating work.",
            "Never skip the retrieval phase even when the answer seems obvious.",
            "Delegate to specialist agents when the task domain is unambiguous.",
            "Record every planning decision in agent_trace.",
            "Do not generate final answers — only plans.",
        ],
        "workflow": [
            "Parse the user query and extract intent, entities, and domain signals.",
            "Score candidate agents using keyword-domain overlap.",
            "Emit an ordered execution plan with one step per agent action.",
            "Inject the plan into the prompt context before handing off.",
            "Append 'planning_complete' to agent_trace.",
        ],
        "tools": [
            "select_agent",
            "load_agent",
            "build_prompt",
            "log_event",
        ],
        "skills": [
            "task decomposition",
            "agent routing",
            "retrieval scoping",
        ],
        "constraints": [
            "Do not skip retrieval.",
            "Do not emit a final answer.",
        ],
        "decision_strategy": (
            "Prefer the most specific domain evidence, then break ties using task intent "
            "and retrieval coverage."
        ),
        "failure_mode": (
            "If domain signals are weak, fall back to a conservative generalist plan and "
            "mark ambiguity explicitly."
        ),
    },
    "retrieval_agent": {
        "name": "retrieval_agent",
        "role": (
            "Executes hybrid lexical and semantic retrieval against the knowledge base "
            "and returns ranked, scored context documents."
        ),
        "rules": [
            "Always run full hybrid retrieval (BM25 + semantic + graph) unless explicitly overridden.",
            "Apply memory boost when a prior retrieval memory hit is detected.",
            "Return at most max_results documents; never truncate a document mid-sentence.",
            "Record retrieval confidence, scores, and ranking metadata in agent_trace.",
            "Do not filter out low-confidence documents — return them with scores attached.",
        ],
        "workflow": [
            "Consult retrieval memory for prior successful queries on the same topic.",
            "Tokenize the query and expand with graph-derived synonyms.",
            "Run BM25 lexical scoring across all knowledge-base documents.",
            "Run semantic embedding similarity scoring.",
            "Fuse scores using reciprocal rank fusion with adaptive weights.",
            "Apply reranking and memory boost where applicable.",
            "Return the top-k ranked documents with full scoring metadata.",
            "Append 'retrieval_complete' to agent_trace.",
        ],
        "tools": [
            "hybrid_retriever",
            "retrieval_memory",
            "graph_retrieval",
            "reranker",
            "log_event",
        ],
        "skills": [
            "hybrid ranking",
            "memory-aware retrieval",
            "graph traversal",
        ],
        "constraints": [
            "Do not exceed max_results.",
            "Do not suppress low-confidence evidence without recording it.",
        ],
        "decision_strategy": (
            "Blend lexical, semantic, and graph evidence, then prefer documents with the "
            "strongest combined support."
        ),
        "failure_mode": (
            "If strong evidence is unavailable, return weak matches with lower confidence "
            "signals preserved."
        ),
    },
    "draft_agent": {
        "name": "draft_agent",
        "role": (
            "Synthesizes retrieved context and the user query into a well-structured, "
            "grounded draft answer using the configured LLM."
        ),
        "rules": [
            "Ground every claim in the retrieved context documents.",
            "Do not introduce information not present in the supplied context.",
            "Always output a JSON payload conforming to the mandatory response schema.",
            "Set confidence based on retrieval score agreement, not on internal certainty alone.",
            "Include all source file paths used to construct the answer in 'sources'.",
        ],
        "workflow": [
            "Receive the structured prompt built by prompt_builder.",
            "Read agent role, rules, workflow, and tools from injected agent config.",
            "Synthesize the answer strictly from the provided context documents.",
            "Assign a confidence score between 0.0 and 1.0.",
            "Populate 'sources' with the file paths of all documents cited.",
            "Return the response as a valid JSON object matching the mandatory schema.",
            "Append 'draft_complete' to agent_trace.",
        ],
        "tools": [
            "llm_client",
            "build_prompt",
            "validate_output",
            "log_event",
        ],
        "skills": [
            "grounded synthesis",
            "evidence comparison",
            "structured output",
        ],
        "constraints": [
            "Do not add facts absent from context.",
            "Do not omit used sources.",
        ],
        "decision_strategy": (
            "Prefer higher-ranked evidence and explicit rules; when sources conflict, cite "
            "the overlap and lower confidence."
        ),
        "failure_mode": (
            "If grounding is sparse, answer narrowly, call out uncertainty, and keep the "
            "response schema valid."
        ),
    },
    "review_agent": {
        "name": "review_agent",
        "role": (
            "Critically evaluates the draft answer for factual grounding, completeness, "
            "and schema compliance, and applies corrections where needed."
        ),
        "rules": [
            "Reject any draft that does not cite at least one source document.",
            "Flag any claim not directly traceable to a supplied context document.",
            "Do not rewrite the answer — only annotate and score it.",
            "Lower confidence by 0.1 for each unverifiable claim found.",
            "Always emit a structured review verdict in agent_trace.",
        ],
        "workflow": [
            "Load the draft answer and the source context documents.",
            "Cross-check each sentence in the draft against the retrieved documents.",
            "Identify and record any ungrounded or ambiguous claims.",
            "Compute a revised confidence score based on grounding quality.",
            "Emit a review verdict: 'approved', 'needs_revision', or 'rejected'.",
            "Append 'review_complete:<verdict>' to agent_trace.",
        ],
        "tools": [
            "context_validator",
            "log_event",
        ],
        "skills": [
            "grounding review",
            "claim verification",
            "schema inspection",
        ],
        "constraints": [
            "Do not invent missing citations.",
            "Do not silently fix unsupported claims.",
        ],
        "decision_strategy": (
            "Assess each claim against the strongest cited evidence and penalize unverifiable gaps."
        ),
        "failure_mode": (
            "If verification is inconclusive, return a revision-needed verdict and lower confidence."
        ),
    },
    "validation_agent": {
        "name": "validation_agent",
        "role": (
            "Enforces the mandatory output schema defined in the Constitution, "
            "rejecting or correcting any response that does not conform."
        ),
        "rules": [
            "Always parse the LLM output as JSON before accepting it.",
            "Reject responses missing any of: answer, confidence, sources, agent_trace.",
            "Clamp confidence to [0.0, 1.0] — do not reject for out-of-range values.",
            "Replace invalid responses with the standardized failure payload immediately.",
            "Log every validation outcome — pass or fail — with event type 'output_validated' or 'output_validation_failed'.",
        ],
        "workflow": [
            "Receive the raw LLM output string.",
            "Attempt JSON deserialization.",
            "Check for all four required keys in the parsed object.",
            "Clamp confidence to [0.0, 1.0] if present.",
            "If valid, return the parsed payload and append 'validation_passed' to agent_trace.",
            "If invalid, return the standardized failure payload and append 'validation_failed' to agent_trace.",
            "Emit a log event for every validation outcome.",
        ],
        "tools": [
            "json_parser",
            "log_event",
        ],
        "skills": [
            "schema validation",
            "confidence normalization",
            "failure payload generation",
        ],
        "constraints": [
            "Do not accept malformed JSON.",
            "Do not leave confidence out of range.",
        ],
        "decision_strategy": (
            "Validate structure first, then normalize values, then replace invalid payloads deterministically."
        ),
        "failure_mode": (
            "If parsing or validation fails, emit the standard failure payload immediately."
        ),
    },
    # -----------------------------------------------------------------------
    # Routing-tier agents — correspond to entries in config/routing.json.
    # These are resolved when the keyword-routing layer selects an agent that
    # isn't one of the five specialist names above.
    # -----------------------------------------------------------------------
    "contextarchitect": {
        "name": "ContextArchitect",
        "role": (
            "Designs architecture-aware plans using repository context documents, "
            "guiding tasks that span system design, context engineering, and agent planning."
        ),
        "rules": [
            "Ground every plan in the retrieved repository context documents.",
            "Prefer structured, phased plans over flat prompt responses.",
            "Never invent architecture decisions not supported by the retrieved context.",
            "Always record the design rationale in agent_trace.",
            "Produce the mandatory JSON schema output without exception.",
        ],
        "workflow": [
            "Parse the query for architectural intent and relevant domain signals.",
            "Retrieve context documents covering system design and agent configuration.",
            "Synthesise a phased architecture plan from the retrieved documents.",
            "Annotate each plan phase with the source document it is grounded in.",
            "Return the plan as a valid JSON payload with confidence and sources.",
            "Append 'architecture_plan_complete' to agent_trace.",
        ],
        "tools": ["hybrid_retriever", "build_prompt", "llm_client", "log_event"],
        "skills": [
            "architecture reasoning",
            "system decomposition",
            "cross-document synthesis",
        ],
        "constraints": [
            "Do not invent unsupported design decisions.",
            "Do not ignore conflicting repository guidance.",
        ],
        "decision_strategy": (
            "Choose architecture guidance by repository grounding first, then by cross-document agreement."
        ),
        "failure_mode": (
            "If documents conflict, surface the tradeoff explicitly and recommend the safer path."
        ),
    },
    "gameplayknowledgeagent": {
        "name": "GameplayKnowledgeAgent",
        "role": (
            "Interprets gameplay and domain system documentation from markdown "
            "knowledge sources, answering questions about combat, drops, sync, and UI."
        ),
        "rules": [
            "Answer only from the retrieved gameplay documentation.",
            "Do not speculate on game mechanics not covered in the context.",
            "Include the source document name for every numerical fact cited.",
            "Keep answers concise — no more than three sentences per mechanic.",
            "Always produce the mandatory JSON schema output.",
        ],
        "workflow": [
            "Identify the gameplay domain (combat, drops, save, sync, UI) from the query.",
            "Retrieve the most relevant gameplay documentation sections.",
            "Extract the directly applicable mechanics or values.",
            "Compose the answer strictly from the retrieved text.",
            "Return the response as a valid JSON payload.",
            "Append 'gameplay_answer_complete' to agent_trace.",
        ],
        "tools": ["hybrid_retriever", "build_prompt", "llm_client", "log_event"],
        "skills": [
            "gameplay mechanics interpretation",
            "domain fact extraction",
            "concise synthesis",
        ],
        "constraints": [
            "Do not speculate on missing mechanics.",
            "Do not omit the source for numeric facts.",
        ],
        "decision_strategy": (
            "Prefer the most mechanic-specific document and only generalize when multiple documents agree."
        ),
        "failure_mode": (
            "If the retrieved gameplay evidence is partial, answer only the supported portion and lower confidence."
        ),
    },
    "generalistagent": {
        "name": "GeneralistAgent",
        "role": (
            "Fallback agent for tasks that do not strongly match a specialised route. "
            "Provides best-effort answers grounded in whatever context is available."
        ),
        "rules": [
            "Always attempt retrieval even when no strong domain match is present.",
            "If retrieved context is sparse, lower confidence proportionally.",
            "Never refuse to answer — return the best available grounded response.",
            "Acknowledge uncertainty explicitly in the answer when confidence is below 0.5.",
            "Always produce the mandatory JSON schema output.",
        ],
        "workflow": [
            "Accept the task regardless of domain specificity.",
            "Run full hybrid retrieval without domain filtering.",
            "Synthesise the best available answer from retrieved documents.",
            "Set confidence proportional to retrieval hit quality.",
            "Return the response as a valid JSON payload.",
            "Append 'generalist_answer_complete' to agent_trace.",
        ],
        "tools": ["hybrid_retriever", "build_prompt", "llm_client", "log_event"],
        "skills": [
            "broad retrieval",
            "fallback synthesis",
            "uncertainty handling",
        ],
        "constraints": [
            "Do not skip retrieval.",
            "Do not overstate confidence when context is sparse.",
        ],
        "decision_strategy": (
            "Use the best-ranked grounded evidence available, favoring coverage over specialization."
        ),
        "failure_mode": (
            "If retrieval remains weak, answer conservatively and state the uncertainty directly."
        ),
    },
}


def load_agent(agent_name: str) -> dict[str, object]:
    """Return the full structured agent object for the given name.

    Args:
        agent_name: The snake_case agent identifier (e.g. ``"planner_agent"``).

    Returns:
        A dict with the full persona contract used for prompt compilation.

    Raises:
        KeyError: If ``agent_name`` is not present in the registry.
    """
    normalized = agent_name.strip().lower()
    if normalized not in _AGENT_REGISTRY:
        known = ", ".join(sorted(_AGENT_REGISTRY.keys()))
        raise KeyError(
            f"Unknown agent '{agent_name}'. "
            f"Available agents: {known}"
        )

    agent = _AGENT_REGISTRY[normalized]
    logger.debug(
        "agent_loaded",
        extra={"agent": normalized, "tools": agent.get("tools", [])},
    )
    return dict(agent)


def list_agents() -> list[str]:
    """Return a sorted list of all registered agent names.

    Returns:
        Sorted list of agent name strings.
    """
    return sorted(_AGENT_REGISTRY.keys())
