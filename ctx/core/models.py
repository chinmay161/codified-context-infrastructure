"""Strongly typed data contracts for the context engine core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _validate_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _validate_string_list(values: list[str] | None, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list of strings.")

    normalized: list[str] = []
    for value in values:
        normalized.append(_validate_non_empty_string(value, field_name))
    return normalized


@dataclass(slots=True, frozen=True)
class Task:
    """Represents a unit of work that needs context and agent planning."""

    query: str
    files_changed: list[str] | None = None
    context_tags: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _validate_non_empty_string(self.query, "query"))
        object.__setattr__(
            self,
            "files_changed",
            _validate_string_list(self.files_changed, "files_changed") or None,
        )
        object.__setattr__(
            self,
            "context_tags",
            _validate_string_list(self.context_tags, "context_tags") or None,
        )


@dataclass(slots=True, frozen=True)
class RetrievedContext:
    """Represents context material retrieved for a task."""

    docs: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "docs", _validate_string_list(self.docs, "docs"))
        object.__setattr__(self, "files", _validate_string_list(self.files, "files"))
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")


@dataclass(slots=True, frozen=True)
class Agent:
    """Describes an execution agent that can handle planned work."""

    name: str
    description: str
    domain: list[str]
    spec_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_non_empty_string(self.name, "name"))
        object.__setattr__(
            self,
            "description",
            _validate_non_empty_string(self.description, "description"),
        )
        object.__setattr__(self, "domain", _validate_string_list(self.domain, "domain"))
        object.__setattr__(
            self,
            "spec_path",
            _validate_non_empty_string(self.spec_path, "spec_path"),
        )


@dataclass(slots=True, frozen=True)
class ExecutionPlan:
    """Captures the selected agent, the retrieved context, and execution steps."""

    agent: Agent
    context: RetrievedContext
    steps: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.agent, Agent):
            raise TypeError("agent must be an Agent instance.")
        if not isinstance(self.context, RetrievedContext):
            raise TypeError("context must be a RetrievedContext instance.")
        object.__setattr__(self, "steps", _validate_string_list(self.steps, "steps"))


@dataclass(slots=True, frozen=True)
class Trace:
    """Captures deterministic execution trace metadata for a single task run."""

    trace_id: str
    task: str
    agent: str
    retrieval_hits: int
    files_used: list[str]
    timestamp: str
    steps: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _validate_non_empty_string(self.trace_id, "trace_id"))
        object.__setattr__(self, "task", _validate_non_empty_string(self.task, "task"))
        object.__setattr__(self, "agent", _validate_non_empty_string(self.agent, "agent"))
        if not isinstance(self.retrieval_hits, int):
            raise TypeError("retrieval_hits must be an integer.")
        if self.retrieval_hits < 0:
            raise ValueError("retrieval_hits must be greater than or equal to zero.")
        object.__setattr__(self, "files_used", _validate_string_list(self.files_used, "files_used"))
        object.__setattr__(self, "timestamp", _validate_non_empty_string(self.timestamp, "timestamp"))
        object.__setattr__(self, "steps", _validate_string_list(self.steps, "steps"))
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")


@dataclass(slots=True, frozen=True)
class Response:
    """Represents the prepared prompt and context metadata for downstream execution."""

    prompt: str
    agent_used: str
    context_used: dict[str, Any]
    trace_id: str
    trace: dict[str, Any]
    latency_ms: float
    output: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _validate_non_empty_string(self.prompt, "prompt"))
        object.__setattr__(
            self,
            "agent_used",
            _validate_non_empty_string(self.agent_used, "agent_used"),
        )
        if not isinstance(self.context_used, dict):
            raise TypeError("context_used must be a dictionary.")
        object.__setattr__(self, "trace_id", _validate_non_empty_string(self.trace_id, "trace_id"))
        if not isinstance(self.trace, dict):
            raise TypeError("trace must be a dictionary.")
        if not isinstance(self.latency_ms, (float, int)):
            raise TypeError("latency_ms must be a float.")
        if float(self.latency_ms) < 0:
            raise ValueError("latency_ms must be greater than or equal to zero.")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        if not isinstance(self.output, str):
            raise TypeError("output must be a string.")
