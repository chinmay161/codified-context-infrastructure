"""Context Compiler engine — deterministic prompt generation for external LLMs.

Pipeline (single-pass, no retry loops):
  1. Coerce the incoming query into a Task.
  2. Select the best-matching agent via keyword-domain scoring.
  3. Load structured agent config (role, rules, workflow, tools).
  4. Enrich the retrieval task with agent domain terms.
  5. Run selective hybrid retrieval — top-k high-signal documents only.
  6. Compile the final structured prompt.
  7. Return a CompiledContext ready to be pasted into Copilot / Codex.

Nothing here executes tool chains, calls an LLM, or retries anything.
agent_trace is kept as an optional internal field and is never written
into the compiled prompt.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import load_agent
from .distillation import compute_compile_confidence, distill_context
from .interfaces import ContextEngine
from .models import Agent, ContextDistillation, ExecutionPlan, RetrievedContext, Task, Trace
from .prompt_builder import build_adaptive_prompt
from .query_analyzer import analyze_query
from .retrieval_memory import RetrievalMemory
from .skill_guidance import select_skill_guidance
from .retrieval_v2 import tokenize_text
from .retrieval_v3 import HybridRetriever
from ..utils.env import load_dotenv
from ..utils.file_loader import load_markdown_documents
from ..utils.logger import get_observability_logger, log_event


@dataclass(slots=True, frozen=True)
class AgentRoute:
    """Maps a set of keywords to a routing agent."""

    agent: Agent
    keywords: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CompiledContext:
    """The final output of the Context Compiler pipeline.

    Fields
    ------
    prompt        : Structured prompt ready for Copilot / Codex consumption.
    agent_used    : Name of the agent that shaped retrieval and prompt structure.
    context_files : Ordered list of knowledge-base files surfaced by retrieval.
    context_docs  : Raw content of every retrieved document (same order).
    trace_id      : Unique run identifier for observability.
    latency_ms    : Wall-clock time for the full compilation pass.
    internal_trace: Step labels recorded during compilation — NEVER injected
                    into *prompt*.  Kept here for offline debugging only.
    """

    prompt: str
    agent_used: str
    context_files: list[str]
    context_docs: list[str]
    trace_id: str
    latency_ms: float
    internal_trace: list[str]
    metadata: dict[str, Any]


class DefaultContextEngine(ContextEngine):
    """Config-driven Context Compiler with reusable per-instance configuration."""

    def __init__(
        self,
        knowledge_path: str | Path | None = None,
        agents_config: str | Path | None = None,
        routing_config: str | Path | None = None,
        enable_memory: bool = True,
        max_results: int = 5,
        storage_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        dotenv_path: str | Path | None = None,
        config_path: str | Path | None = None,
        knowledge_paths: list[str | Path] | None = None,
    ) -> None:
        self._instance_id = uuid4().hex
        self._storage_path = (
            Path(storage_path).expanduser().resolve()
            if storage_path is not None
            else (Path.cwd() / ".ctx").resolve()
        )
        load_dotenv(dotenv_path)
        self._config_path = self._resolve_config_path(
            config_path=config_path,
            routing_config=routing_config,
            agents_config=agents_config,
        )
        self._config = self._load_config(self._config_path)
        self._routes = self._load_routes(self._config)
        self._package_root = Path(__file__).resolve().parents[1]
        self._knowledge_base_root = (
            self._package_root / "corpus" / "knowledge_base"
        ).resolve()
        self._knowledge_paths = self._resolve_knowledge_paths(
            configured_paths=self._config.get("knowledge_paths", []),
            knowledge_path=knowledge_path,
            knowledge_paths=knowledge_paths,
        )
        if max_results < 1:
            raise ValueError("max_results must be greater than zero.")
        self._max_results = max_results
        resolved_log_dir = (
            Path(log_dir).expanduser().resolve() if log_dir else self._storage_path / "logs"
        )
        self._logger = get_observability_logger(
            log_dir=resolved_log_dir,
            logger_name=f"ctx.observability.{self._instance_id}",
        )
        resolved_memory_path = self._resolve_memory_path(
            enable_memory=enable_memory,
            memory_path=memory_path,
        )
        self._retrieval_memory = RetrievalMemory(memory_path=resolved_memory_path)
        self._hybrid_retriever: HybridRetriever | None = None

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def compile(self, task: str | Task) -> CompiledContext:
        """Run the full Context Compiler pipeline and return a CompiledContext.

        This is the primary entry-point for external-LLM workflows.  The
        returned ``CompiledContext.prompt`` is the only artefact that should
        be shared with Copilot / Codex — all internal metadata stays local.

        Steps (single pass, no retries):
          select_agent → load_agent → enrich_task → retrieve → build_prompt
        """
        normalized_task = self._coerce_task(task)
        start_time = time.perf_counter()
        trace_id = str(uuid4())
        internal_trace: list[str] = []

        # 1. Select routing agent.
        selected_agent, selection_meta = self._select_agent_with_metadata(normalized_task)
        internal_trace.append("select_agent")
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="agent_selected",
            message=f"Selected agent '{selected_agent.name}'.",
            metadata=selection_meta,
        )

        # 2. Load structured agent (role, rules, workflow, tools).
        #    Used only for prompt shaping — no tool execution.
        structured_agent: dict[str, object] | None = None
        try:
            structured_agent = load_agent(selected_agent.name)
            internal_trace.append(f"agent_loaded:{selected_agent.name}")
        except KeyError:
            internal_trace.append(f"agent_load_skipped:{selected_agent.name}")

        memory_entry = self._retrieval_memory.retrieve_with_memory(
            normalized_task.query,
            structured_agent or {"name": selected_agent.name, "role": selected_agent.description},
        )
        query_analysis = analyze_query(
            normalized_task.query,
            context={"memory_entry": memory_entry},
        )
        internal_trace.append(f"query_analyzed:{query_analysis.task_mode}")
        # 3. Enrich retrieval task with agent domain signals.
        retrieval_task = self._enrich_task_with_agent(normalized_task, structured_agent)

        # 4. Selective retrieval — top-k high-signal documents only.
        retrieved_context = self._retrieve_hybrid(retrieval_task, structured_agent or selected_agent)
        internal_trace.append("retrieve_context")
        distillation = distill_context(
            task=normalized_task,
            retrieved_context=retrieved_context,
            query_analysis=query_analysis,
        )
        internal_trace.append("distill_context")
        skill_guidance = select_skill_guidance(
            analysis=query_analysis,
            retrieval_metadata=retrieved_context.metadata,
        )
        if skill_guidance:
            internal_trace.append("skill_guidance_selected")
        compile_confidence = compute_compile_confidence(
            retrieval_metadata=retrieved_context.metadata,
            distillation=distillation,
        )
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="retrieval_complete",
            message="Context retrieval finished.",
            metadata={
                "retrieval_hits": len(retrieved_context.docs),
                "files_used": retrieved_context.files,
                "agent": selected_agent.name,
                "query_type": query_analysis.query_type,
                "task_mode": query_analysis.task_mode,
                "compile_confidence": compile_confidence,
            },
        )

        # 5. Compile the structured prompt.
        prompt, budget = self._compile_prompt(
            task=normalized_task,
            selected_agent=selected_agent,
            structured_agent=structured_agent,
            retrieved_context=retrieved_context,
            query_analysis=query_analysis.as_dict(),
            task_mode=query_analysis.task_mode,
            distillation=distillation,
            compile_confidence=compile_confidence,
            skill_guidance=skill_guidance,
        )
        internal_trace.append("prompt_compiled")
        distillation_payload = asdict(distillation)
        distillation_payload["budget_decisions"] = budget
        metadata = {
            **retrieved_context.metadata,
            "query_analysis": query_analysis.as_dict(),
            "task_mode": query_analysis.task_mode,
            "distillation": distillation_payload,
            "budget": budget,
            "compile_confidence": compile_confidence,
            "skill_guidance": skill_guidance,
        }
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="prompt_compiled",
            message="Structured prompt compiled.",
            metadata={
                "agent": selected_agent.name,
                "prompt_length": len(prompt),
                "agent_injected": structured_agent is not None,
                "budget": budget,
            },
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="compile_complete",
            message="Context compilation finished.",
            metadata={
                "agent": selected_agent.name,
                "latency_ms": latency_ms,
                "retrieval_hits": len(retrieved_context.docs),
                "task_mode": query_analysis.task_mode,
            },
        )

        return CompiledContext(
            prompt=prompt,
            agent_used=selected_agent.name,
            context_files=retrieved_context.files,
            context_docs=retrieved_context.docs,
            trace_id=trace_id,
            latency_ms=latency_ms,
            internal_trace=internal_trace,
            metadata=metadata,
        )

    # Backward-compatible shim: execute() → compile() with a legacy Response shape.
    def execute(self, task: str | Task):  # type: ignore[override]
        """Legacy shim — prefer compile() for new code."""
        from .models import Response, Trace

        compiled = self.compile(task)
        trace = Trace(
            trace_id=compiled.trace_id,
            task=self._coerce_task(task).query,
            agent=compiled.agent_used,
            retrieval_hits=len(compiled.context_docs),
            files_used=compiled.context_files,
            timestamp=datetime.now(timezone.utc).isoformat(),
            steps=compiled.internal_trace,
        )
        return Response(
            prompt=compiled.prompt,
            agent_used=compiled.agent_used,
            context_used={
                "files": compiled.context_files,
                "docs": compiled.context_docs,
                "metadata": compiled.metadata,
            },
            trace_id=compiled.trace_id,
            trace=asdict(
                Trace(
                    trace_id=trace.trace_id,
                    task=trace.task,
                    agent=trace.agent,
                    retrieval_hits=trace.retrieval_hits,
                    files_used=trace.files_used,
                    timestamp=trace.timestamp,
                    steps=trace.steps,
                    metadata=compiled.metadata,
                )
            ),
            latency_ms=compiled.latency_ms,
            output=compiled.prompt,
        )

    def plan(self, task: str | Task) -> ExecutionPlan:
        """Build an execution plan from the selected agent and retrieved context."""
        normalized_task = self._coerce_task(task)
        agent = self.select_agent(normalized_task)
        retrieved_context = self._retrieve_hybrid(normalized_task, agent)
        return self._build_plan(
            task=normalized_task, agent=agent, retrieved_context=retrieved_context
        )

    def retrieve(self, task: str | Task) -> RetrievedContext:
        """Retrieve markdown documents using selective hybrid ranking."""
        normalized_task = self._coerce_task(task)
        agent = self.select_agent(normalized_task)
        return self._retrieve_hybrid(normalized_task, agent)

    def select_agent(self, task: str | Task) -> Agent:
        """Select an agent by scoring keyword overlap against routing rules."""
        selected_agent, _ = self._select_agent_with_metadata(self._coerce_task(task))
        return selected_agent

    def build_prompt(self, plan: ExecutionPlan) -> str:
        """Construct a structured prompt from an execution plan."""
        task_payload = plan.context.metadata.get("task", {})
        task_query = task_payload.get("query", "No task query provided.")
        task_files = ", ".join(task_payload.get("files_changed", [])) or "None"
        task_tags = ", ".join(task_payload.get("context_tags", [])) or "None"
        context_files = (
            "\n".join(f"- {path}" for path in plan.context.files) or "- No files matched"
        )
        context_docs = (
            "\n\n".join(plan.context.docs) or "No context documents were retrieved."
        )

        return (
            "SYSTEM\n"
            "You are a context-aware assistant for Copilot/Codex. "
            "Use the agent role and the provided context to answer the task below.\n\n"
            "AGENT\n"
            f"Name: {plan.agent.name}\n"
            f"Description: {plan.agent.description}\n"
            f"Domains: {', '.join(plan.agent.domain)}\n\n"
            "CONTEXT\n"
            f"Files:\n{context_files}\n\n"
            f"Documents:\n{context_docs}\n\n"
            "TASK\n"
            f"Query: {task_query}\n"
            f"Files Changed: {task_files}\n"
            f"Context Tags: {task_tags}"
        )

    def record_retrieval_feedback(
        self,
        task: Task,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Persist retrieval feedback into memory for future signal boosting."""
        self._get_hybrid_retriever().update_feedback(
            task=task,
            selected_docs=selected_docs,
            retrieval_score=retrieval_score,
            observed_keywords=observed_keywords,
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _compile_prompt(
        self,
        task: Task,
        selected_agent: Agent,
        structured_agent: dict[str, object] | None,
        retrieved_context: RetrievedContext,
        query_analysis: dict[str, Any],
        task_mode: str,
        distillation: ContextDistillation,
        compile_confidence: dict[str, Any],
        skill_guidance: list[str],
    ) -> tuple[str, dict[str, Any]]:
        """Choose prompt-builder path and return the final compiled string.

        When a structured agent loaded successfully, ``_build_agent_prompt``
        injects the full agent config (role, rules, workflow, tools).
        Otherwise the plan-based builder provides a clean fallback.
        No agent_trace is written into the output in either path.
        """
        if structured_agent is not None:
            context_text = "\n\n".join(retrieved_context.docs) if retrieved_context.docs else ""
            return build_adaptive_prompt(
                query=task.query,
                context=context_text,
                agent=structured_agent,
                query_analysis=query_analysis,
                task_mode=task_mode,
                distillation=distillation,
                compile_confidence=compile_confidence,
                skill_guidance=skill_guidance,
            )
        plan = self._build_plan(
            task=task,
            agent=selected_agent,
            retrieved_context=retrieved_context,
        )
        prompt = self.build_prompt(plan)
        return (
            prompt,
            {
                "config": {},
                "original_context_chars": len("\n\n".join(retrieved_context.docs)),
                "final_prompt_chars": len(prompt),
                "distillation_chars_used": 0,
                "evidence_chars_used": 0,
                "compression_applied": False,
                "dropped_docs": [],
                "preserved_evidence": [],
            },
        )

    def _retrieve_hybrid(
        self, task: Task, agent: Agent | dict[str, object] | None
    ) -> RetrievedContext:
        """Run selective hybrid retrieval limited to top-k high-signal documents."""
        if agent is None:
            agent_dict = None
        elif isinstance(agent, dict):
            agent_dict = agent
        else:
            agent_dict = {"name": agent.name, "role": agent.description}

        ranked_documents, metadata = self._get_hybrid_retriever().retrieve(
            query=task,
            agent=agent_dict,
            top_k=self._max_results,
        )

        return RetrievedContext(
            docs=[item.content for item in ranked_documents],
            files=[item.path for item in ranked_documents],
            metadata=metadata,
        )

    def _get_hybrid_retriever(self) -> HybridRetriever:
        """Lazily initialise a cached selective hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                documents=load_markdown_documents(self._knowledge_paths),
                knowledge_base_root=self._knowledge_base_root,
                default_top_k=self._max_results,
                memory=self._retrieval_memory,
            )
        return self._hybrid_retriever

    def _build_plan(
        self,
        task: Task,
        agent: Agent,
        retrieved_context: RetrievedContext,
    ) -> ExecutionPlan:
        context = RetrievedContext(
            docs=retrieved_context.docs,
            files=retrieved_context.files,
            metadata={
                **retrieved_context.metadata,
                "task": {
                    "query": task.query,
                    "files_changed": task.files_changed or [],
                    "context_tags": task.context_tags or [],
                },
            },
        )
        steps = [
            f"Route task to agent '{agent.name}' based on keyword-domain match.",
            "Load relevant markdown context from configured knowledge sources.",
            "Compile retrieved context into a structured prompt for Copilot/Codex.",
        ]
        if task.files_changed:
            steps.insert(1, "Prioritize context related to the changed files when available.")
        if task.context_tags:
            steps.insert(1, "Bias retrieval toward the supplied context tags.")
        return ExecutionPlan(agent=agent, context=context, steps=steps)

    def _select_agent_with_metadata(self, task: Task) -> tuple[Agent, dict[str, Any]]:
        task_terms = self._task_terms(task)
        default_route: AgentRoute | None = None
        best_route: AgentRoute | None = None
        best_score = -1
        route_scores: dict[str, int] = {}
        route_matches: dict[str, list[str]] = {}

        for route in self._routes:
            if not route.keywords and default_route is None:
                default_route = route
            matches = sorted(task_terms.intersection(route.keywords))
            score = len(matches)
            route_scores[route.agent.name] = score
            route_matches[route.agent.name] = matches
            if score > best_score:
                best_route = route
                best_score = score

        if best_score <= 0 and default_route is not None:
            selected_route = default_route
            selection_reason = "fallback_default_route"
        else:
            if best_route is None:
                raise ValueError("No agents are configured for routing.")
            selected_route = best_route
            selection_reason = "keyword_overlap"

        return selected_route.agent, {
            "selected_agent": selected_route.agent.name,
            "selection_reason": selection_reason,
            "task_terms": sorted(task_terms),
            "route_scores": route_scores,
            "route_matches": route_matches,
        }

    def _task_terms(self, task: Task) -> set[str]:
        terms = set(tokenize_text(task.query))
        for tag in task.context_tags or []:
            terms.update(tokenize_text(tag))
        for file_path in task.files_changed or []:
            terms.update(tokenize_text(Path(file_path).stem))
        return terms

    @staticmethod
    def _enrich_task_with_agent(
        task: Task,
        structured_agent: dict[str, object] | None,
    ) -> Task:
        """Return a Task enriched with agent domain terms as context_tags.

        Merges the agent's ``role`` keywords and ``name`` into the task's
        context tags so the retriever surfaces agent-relevant documents.
        Returns the original task unchanged when no structured agent is given.
        """
        if structured_agent is None:
            return task

        agent_name: str = str(structured_agent.get("name", ""))
        agent_role: str = str(structured_agent.get("role", ""))

        new_tags: list[str] = []
        for raw in (agent_name, agent_role.split(".")[0]):
            new_tags.extend(tokenize_text(raw))

        seen: set[str] = set(task.context_tags or [])
        merged_tags: list[str] = list(task.context_tags or [])
        for tag in new_tags:
            if tag and tag not in seen and len(tag) > 2:
                merged_tags.append(tag)
                seen.add(tag)

        if not merged_tags:
            return task

        return Task(
            query=task.query,
            files_changed=task.files_changed,
            context_tags=merged_tags,
        )

    @staticmethod
    def _coerce_task(task: str | Task) -> Task:
        if isinstance(task, Task):
            return task
        if isinstance(task, str):
            return Task(query=task)
        raise TypeError("task must be a string query or Task instance.")

    # ---------------------------------------------------------------------- #
    # Config / path resolution (unchanged from original)                      #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Routing config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise TypeError("Routing config root must be a JSON object.")
        return loaded

    def _resolve_knowledge_paths(
        self,
        configured_paths: list[str],
        knowledge_path: str | Path | None,
        knowledge_paths: list[str | Path] | None,
    ) -> list[Path]:
        raw_paths = _normalize_knowledge_inputs(
            knowledge_path=knowledge_path,
            knowledge_paths=knowledge_paths,
            configured_paths=configured_paths,
        )
        resolved_paths: list[Path] = []
        for raw_path in raw_paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = self._resolve_relative_path(candidate)
            else:
                candidate = candidate.resolve()
            if candidate not in resolved_paths:
                resolved_paths.append(candidate)
        if not resolved_paths:
            resolved_paths.append(self._knowledge_base_root)
        return resolved_paths

    def _resolve_relative_path(self, candidate: Path) -> Path:
        search_roots = [
            Path.cwd().resolve(),
            self._config_path.parent.resolve(),
            self._config_path.parent.parent.resolve(),
            self._package_root.resolve(),
        ]
        for root in search_roots:
            possible = (root / candidate).resolve()
            if possible.exists():
                return possible
        return (Path.cwd().resolve() / candidate).resolve()

    @staticmethod
    def _resolve_config_path(
        config_path: str | Path | None,
        routing_config: str | Path | None,
        agents_config: str | Path | None,
    ) -> Path:
        selected = routing_config or agents_config or config_path
        if selected is not None:
            return Path(selected).expanduser().resolve()
        return (Path(__file__).resolve().parents[1] / "config" / "routing.json").resolve()

    def _resolve_memory_path(
        self,
        enable_memory: bool,
        memory_path: str | Path | None,
    ) -> Path | None:
        if not enable_memory:
            return None
        if memory_path is not None:
            return Path(memory_path).expanduser().resolve()
        return (self._storage_path / "memory" / "retrieval_memory.json").resolve()

    @staticmethod
    def _load_routes(config: dict[str, Any]) -> list[AgentRoute]:
        raw_agents = config.get("agents", [])
        if not isinstance(raw_agents, list):
            raise TypeError("agents must be a list in the routing config.")

        routes: list[AgentRoute] = []
        for raw_agent in raw_agents:
            if not isinstance(raw_agent, dict):
                raise TypeError("Each agent config entry must be an object.")
            domain = raw_agent.get("domain", [])
            keywords = raw_agent.get("keywords", [])
            if not isinstance(domain, list):
                raise TypeError("agent domain must be a list of strings.")
            if not isinstance(keywords, list):
                raise TypeError("agent keywords must be a list of strings.")
            agent = Agent(
                name=raw_agent["name"],
                description=raw_agent["description"],
                domain=domain,
                spec_path=raw_agent["spec_path"],
            )
            routes.append(
                AgentRoute(agent=agent, keywords=tuple(str(t).lower() for t in keywords))
            )
        return routes


# --------------------------------------------------------------------------- #
# Module-level helpers                                                          #
# --------------------------------------------------------------------------- #

def _normalize_knowledge_inputs(
    knowledge_path: str | Path | None,
    knowledge_paths: list[str | Path] | None,
    configured_paths: list[str],
) -> list[str | Path]:
    if knowledge_paths is not None:
        return knowledge_paths
    if knowledge_path is not None:
        return [knowledge_path]
    return configured_paths
