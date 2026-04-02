"""Abstract interface for context engine implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Agent, ExecutionPlan, RetrievedContext, Task


class ContextEngine(ABC):
    """Defines the core contract for planning context-aware agent execution."""

    @abstractmethod
    def plan(self, task: Task) -> ExecutionPlan:
        """Build an execution plan for a task using agent routing and context retrieval."""

    @abstractmethod
    def retrieve(self, task: Task) -> RetrievedContext:
        """Retrieve relevant context for a task from the configured knowledge sources."""

    @abstractmethod
    def select_agent(self, task: Task) -> Agent:
        """Select the most appropriate agent for a task."""

    @abstractmethod
    def build_prompt(self, plan: ExecutionPlan) -> str:
        """Construct a structured prompt from an execution plan."""
