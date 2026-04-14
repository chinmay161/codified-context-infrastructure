# CONSTITUTION — Codified Context Infrastructure

> Version: 1.0.0
> Scope: Agent execution, prompt construction, retrieval, output validation
> Authority: This document governs all agent behavior in the system. Any implementation
>             that contradicts this document is invalid and must be corrected.

---

## ARTICLE I — CODING STANDARDS

### 1.1 Language and Style
- All implementation code MUST be written in Python 3.11+.
- All functions MUST have type annotations on every parameter and return value.
- All public functions and classes MUST have docstrings describing purpose, parameters, and return values.
- Line length MUST NOT exceed 120 characters.
- No wildcard imports (`from module import *`) are permitted anywhere in the codebase.
- All string formatting MUST use f-strings; `.format()` and `%` substitution are prohibited.

### 1.2 Module Organization
- Agent definitions MUST live in `src/ctx/core/agent.py`.
- Prompt construction logic MUST live in `src/ctx/core/prompt_builder.py`.
- The execution engine entry point MUST remain `src/ctx/core/engine.py`.
- No business logic may be placed in `__init__.py` files.
- Utility functions shared across modules MUST be placed in `src/ctx/utils/`.

### 1.3 Configuration
- No values that vary per deployment (API keys, paths, thresholds) may be hardcoded.
- All configurable values MUST be read from environment variables or JSON config files.
- Default values for configurable constants are permitted in module-level constants marked with `_DEFAULT_` prefix.

### 1.4 Error Handling
- All exceptions raised within agent or prompt modules MUST be typed (not bare `raise Exception`).
- All external I/O (file reads, HTTP calls) MUST be wrapped in try/except with specific exception types.
- Agent execution failures MUST NOT propagate as unhandled exceptions to the caller.
  The caller MUST always receive a structured `AgentResponse` — even on failure.

---

## ARTICLE II — OUTPUT FORMAT CONSTRAINTS

### 2.1 Mandatory Response Schema
Every agent response MUST conform to the following JSON schema:

```json
{
  "answer": "<string — the primary content of the agent response>",
  "confidence": "<float between 0.0 and 1.0 — agent's self-assessed certainty>",
  "sources": ["<list of file paths or document identifiers used>"],
  "agent_trace": ["<ordered list of string step labels describing execution path>"]
}
```

### 2.2 Enforcement Rules
- Any LLM response that does not parse as valid JSON MUST be rejected and replaced with a failure payload.
- Any parsed response missing any of the four required keys (`answer`, `confidence`, `sources`, `agent_trace`) MUST be rejected.
- `confidence` values outside [0.0, 1.0] MUST be clamped, not rejected.
- `sources` MUST be a list even if empty; `null` is not acceptable.
- `agent_trace` MUST contain at least one entry — the agent name — even when the trace is minimal.

### 2.3 Failure Payload Format
When output validation fails, the system MUST return:

```json
{
  "answer": "Agent response could not be validated.",
  "confidence": 0.0,
  "sources": [],
  "agent_trace": ["validation_failed"]
}
```

---

## ARTICLE III — AGENT TRIGGERS

### 3.1 Agent Selection
- Agent selection MUST use keyword-overlap scoring against the agent's `keywords` list.
- When no keywords match (score = 0), the fallback `GeneralistAgent` MUST be selected.
- The fallback agent MUST always be present in the routing config and MUST have an empty `keywords` list.
- Agent selection decisions MUST be logged with the event type `agent_selected`.

### 3.2 Agent Activation
- An agent is activated when `load_agent(agent_name)` returns successfully.
- If an agent name is unknown, `load_agent` MUST raise a `KeyError` with a descriptive message.
- Each agent execution MUST begin by appending `"agent_activated:<agent_name>"` to the `agent_trace`.

### 3.3 Structured Agent Schema
Every agent definition MUST include all five fields:

| Field      | Type          | Description                                         |
|------------|---------------|-----------------------------------------------------|
| `name`     | `str`         | Unique identifier (snake_case)                      |
| `role`     | `str`         | Single-sentence description of the agent's purpose  |
| `rules`    | `list[str]`   | Non-negotiable constraints on agent behavior        |
| `workflow` | `list[str]`   | Ordered sequence of steps the agent follows         |
| `tools`    | `list[str]`   | Identifiers of tools/capabilities the agent may use |

---

## ARTICLE IV — WORKFLOW RULES

### 4.1 Execution Pipeline Order
All executions MUST follow this order. Steps may not be reordered or skipped:

1. `select_agent` — Choose the agent based on task routing.
2. `load_agent` — Load the structured agent object from the registry.
3. `retrieve_context` — Run hybrid retrieval for the task.
4. `build_prompt` — Construct the injected prompt using agent and context.
5. `generate_response` — Call the LLM (or use retrieval-only mode).
6. `validate_output` — Enforce the mandatory response schema.
7. `emit_trace` — Append trace entries and log all events.

### 4.2 Memory Constraints
- Retrieval memory MUST be consulted before hybrid retrieval is triggered.
- Retrieval memory MUST be updated after each successful execution via `record_retrieval_feedback`.
- Memory features MUST NOT be removed or disabled by any agent-level code.

### 4.3 LLM Usage Policy
- LLM MUST only be invoked when retrieval confidence falls below `_LLM_SKIP_CONFIDENCE`.
- LLM MAY be force-invoked when `CTX_FORCE_LLM=1` or the task contains the `[force-llm]` tag.
- When LLM is skipped, a retrieval-only payload MUST be returned using `_build_retrieval_only_response`.

---

## ARTICLE V — FAILURE HANDLING

### 5.1 LLM Failure
- LLM failures MUST be caught and returned as structured failure outputs (not exceptions).
- The failure output MUST include the error reason and the LLM invocation mode.
- After LLM failure, the system MUST fall back to the retrieval-only response.

### 5.2 Retrieval Failure
- If hybrid retrieval returns zero results, the system MUST still return a valid `AgentResponse`.
- `sources` and `docs` MAY be empty lists in this case.
- `confidence` MUST be set to `0.0` when retrieval returns no hits.

### 5.3 Agent Load Failure
- If `load_agent` raises a `KeyError`, execution MUST stop and a failure payload returned immediately.
- The `agent_trace` in the failure payload MUST record `"agent_load_failed:<agent_name>"`.

### 5.4 Output Validation Failure
- If the LLM response fails schema validation, the system MUST NOT propagate the invalid output.
- A standardized failure payload (per Article II §2.3) MUST be returned instead.
- The event MUST be logged with type `output_validation_failed`.

---

## ARTICLE VI — LOGGING STANDARDS

- All log events MUST use the `log_event` utility from `src/ctx/utils/logger.py`.
- Required event types: `agent_selected`, `agent_activated`, `retrieval_complete`,
  `prompt_built`, `llm_response_generated`, `output_validated`, `output_validation_failed`, `execution_complete`.
- Every log event MUST include a `trace_id` that is consistent across all events for one execution.
- Log events MUST NOT contain raw API keys, full prompt text, or user PII.
