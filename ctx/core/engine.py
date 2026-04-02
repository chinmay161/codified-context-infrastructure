"""Default v1 implementation of the context engine."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .interfaces import ContextEngine
from .llm_client import LLMClient
from .models import Agent, ExecutionPlan, Response, RetrievedContext, Task, Trace
from .retrieval_memory import RetrievalMemory
from .retrieval_v2 import tokenize_text
from .retrieval_v3 import HybridRetriever
from ..utils.env import load_dotenv
from ..utils.file_loader import load_markdown_documents
from ..utils.logger import get_observability_logger, log_event


@dataclass(slots=True, frozen=True)
class AgentRoute:
    """Represents a routing rule that maps keywords to an agent."""

    agent: Agent
    keywords: tuple[str, ...]


class DefaultContextEngine(ContextEngine):
    """Config-driven context engine with keyword routing and markdown retrieval."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        knowledge_paths: list[str | Path] | None = None,
        max_results: int = 5,
        log_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the engine with routing configuration and knowledge sources.

        Args:
            config_path: Path to the routing configuration JSON.
            knowledge_paths: Optional override for markdown source directories.
            max_results: Maximum number of markdown documents to return per retrieval.
            log_dir: Optional override for structured observability log output.
        """
        self._config_path = Path(
            config_path or Path(__file__).resolve().parents[1] / "config" / "routing.json"
        ).resolve()
        self._config = self._load_config(self._config_path)
        self._routes = self._load_routes(self._config)
        load_dotenv()
        self._knowledge_paths = self._resolve_knowledge_paths(
            config_path=self._config_path,
            configured_paths=self._config.get("knowledge_paths", []),
            override_paths=knowledge_paths,
        )
        if max_results < 1:
            raise ValueError("max_results must be greater than zero.")
        self._max_results = max_results
        self._logger = get_observability_logger(log_dir=log_dir)
        self._knowledge_base_root = Path(__file__).resolve().parents[1] / "corpus" / "knowledge_base"
        if self._knowledge_base_root.resolve() not in self._knowledge_paths:
            self._knowledge_paths.append(self._knowledge_base_root.resolve())
        self._retrieval_memory = RetrievalMemory()
        self._hybrid_retriever: HybridRetriever | None = None
        self._llm_client = LLMClient()

    def plan(self, task: Task) -> ExecutionPlan:
        """Build an execution plan from the selected agent and retrieved context."""
        agent = self.select_agent(task)
        retrieved_context = self._retrieve_hybrid(task, agent)
        return self._build_plan(task=task, agent=agent, retrieved_context=retrieved_context)

    def retrieve(self, task: Task) -> RetrievedContext:
        """Retrieve markdown documents using hybrid lexical and semantic ranking."""
        agent = self.select_agent(task)
        return self._retrieve_hybrid(task, agent)

    def _retrieve_hybrid(self, task: Task, agent: Agent) -> RetrievedContext:
        """Retrieve markdown documents using cached hybrid retrieval."""
        ranked_documents, metadata = self._get_hybrid_retriever().retrieve(
            task=task,
            agent=agent,
            top_k=self._max_results,
        )

        return RetrievedContext(
            docs=[item.content for item in ranked_documents],
            files=[item.path for item in ranked_documents],
            metadata=metadata,
        )

    def select_agent(self, task: Task) -> Agent:
        """Select an agent by scoring keyword overlap against configured routing rules."""
        selected_agent, _ = self._select_agent_with_metadata(task)
        return selected_agent

    def build_prompt(self, plan: ExecutionPlan) -> str:
        """Construct a structured prompt with system, agent, context, and task sections."""
        task_payload = plan.context.metadata.get("task", {})
        task_query = task_payload.get("query", "No task query provided.")
        task_files = ", ".join(task_payload.get("files_changed", [])) or "None"
        task_tags = ", ".join(task_payload.get("context_tags", [])) or "None"
        context_files = "\n".join(f"- {path}" for path in plan.context.files) or "- No files matched"
        context_docs = "\n\n".join(plan.context.docs) or "No context documents were retrieved."
        plan_steps = "\n".join(f"{index}. {step}" for index, step in enumerate(plan.steps, start=1))

        return (
            "SYSTEM\n"
            "You are operating inside a context-aware execution framework. "
            "Use the assigned agent role, follow the execution plan, and ground reasoning in the provided context.\n\n"
            "AGENT\n"
            f"Name: {plan.agent.name}\n"
            f"Description: {plan.agent.description}\n"
            f"Domains: {', '.join(plan.agent.domain)}\n"
            f"Spec Path: {plan.agent.spec_path}\n"
            f"Plan:\n{plan_steps}\n\n"
            "CONTEXT\n"
            f"Files:\n{context_files}\n\n"
            f"Documents:\n{context_docs}\n\n"
            "TASK\n"
            f"Query: {task_query}\n"
            f"Files Changed: {task_files}\n"
            f"Context Tags: {task_tags}\n"
            "Execute the routed task using the context above."
        )

    def execute(self, task: Task) -> Response:
        """Run a fully traced context-engine execution and return an observable response."""
        start_time = time.perf_counter()
        trace_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        execution_steps: list[str] = []

        selected_agent, selection_metadata = self._select_agent_with_metadata(task)
        execution_steps.append("select_agent")
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="agent_selected",
            message=f"Selected agent '{selected_agent.name}' for task.",
            metadata=selection_metadata,
        )

        retrieved_context = self._retrieve_hybrid(task, selected_agent)
        execution_steps.append("retrieve_context")
        retrieval_metadata = {
            "retrieval_hits": len(retrieved_context.docs),
            "files_used": retrieved_context.files,
            "matched_terms": retrieved_context.metadata.get("matched_terms", []),
            "scores": retrieved_context.metadata.get("scores", {}),
            "ranking": retrieved_context.metadata.get("ranking", []),
            "bm25_ranking": retrieved_context.metadata.get("bm25_ranking", []),
            "semantic_ranking": retrieved_context.metadata.get("semantic_ranking", []),
            "graph_ranking": retrieved_context.metadata.get("graph_ranking", []),
            "fusion_scores": retrieved_context.metadata.get("fusion_scores", []),
            "final_ranking": retrieved_context.metadata.get("final_ranking", []),
            "keyword_hits": retrieved_context.metadata.get("keyword_hits", {}),
            "files_considered": retrieved_context.metadata.get("files_considered", []),
            "top_k": retrieved_context.metadata.get("top_k", []),
            "scoring_breakdown": retrieved_context.metadata.get("scoring_breakdown", {}),
            "embedding_similarity": retrieved_context.metadata.get("embedding_similarity", {}),
            "fusion_method": retrieved_context.metadata.get("fusion_method", ""),
            "boosted_docs": retrieved_context.metadata.get("boosted_docs", []),
            "boost_effect": retrieved_context.metadata.get("boost_effect", []),
            "memory_hit": retrieved_context.metadata.get("memory_hit", False),
            "memory_confidence": retrieved_context.metadata.get("memory_confidence", 0.0),
            "expanded_query_terms": retrieved_context.metadata.get("expanded_query_terms", []),
            "graph_nodes": retrieved_context.metadata.get("graph_nodes", []),
            "graph_paths": retrieved_context.metadata.get("graph_paths", []),
            "graph_scores": retrieved_context.metadata.get("graph_scores", []),
            "traversal_depth": retrieved_context.metadata.get("traversal_depth", 0),
            "graph_score_weight": retrieved_context.metadata.get("graph_score_weight", 0.0),
            "graph_size": retrieved_context.metadata.get("graph_size", {}),
            "graph_example": retrieved_context.metadata.get("graph_example", {}),
            "query_type": retrieved_context.metadata.get("query_type", "keyword"),
            "query_analysis": retrieved_context.metadata.get("query_analysis", {}),
            "intent": retrieved_context.metadata.get("intent", ""),
            "weight_distribution": retrieved_context.metadata.get("weight_distribution", {}),
            "rerank_effect": retrieved_context.metadata.get("rerank_effect", []),
            "rerank_top_changes": retrieved_context.metadata.get("rerank_top_changes", []),
            "candidate_count": retrieved_context.metadata.get("candidate_count", 0),
            "graph_expansion_terms": retrieved_context.metadata.get("graph_expansion_terms", []),
            "strategy": retrieved_context.metadata.get("strategy", "hybrid_retrieval_v3"),
        }
        for boosted_doc in retrieval_metadata["boosted_docs"]:
            log_event(
                logger=self._logger,
                trace_id=trace_id,
                event_type="boost_applied",
                message="Applied retrieval boost to ranked document.",
                metadata=boosted_doc,
            )
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="memory_hit" if retrieval_metadata["memory_hit"] else "memory_miss",
            message="Resolved retrieval memory state for query.",
            metadata={
                "memory_hit": retrieval_metadata["memory_hit"],
                "expanded_query_terms": retrieval_metadata["expanded_query_terms"],
            },
        )
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="retrieval_complete",
            message="Completed context retrieval.",
            metadata=retrieval_metadata,
        )
        if retrieval_metadata["weight_distribution"]:
            log_event(
                logger=self._logger,
                trace_id=trace_id,
                event_type="adaptive_weights_applied",
                message="Applied adaptive retrieval weights for the query.",
                metadata={
                    "query_type": retrieval_metadata["query_type"],
                    "intent": retrieval_metadata["intent"],
                    "weight_distribution": retrieval_metadata["weight_distribution"],
                },
            )
        for effect in retrieval_metadata["rerank_top_changes"]:
            log_event(
                logger=self._logger,
                trace_id=trace_id,
                event_type="rerank_adjustment",
                message="Reranking updated a candidate document position.",
                metadata=effect,
            )
        for effect in retrieval_metadata["boost_effect"]:
            log_event(
                logger=self._logger,
                trace_id=trace_id,
                event_type="boosted_doc_selected",
                message="Boosting changed a document's final ranking.",
                metadata=effect,
            )

        plan = self._build_plan(task=task, agent=selected_agent, retrieved_context=retrieved_context)
        prompt = self.build_prompt(plan)
        execution_steps.append("build_prompt")
        output, llm_metadata = self._execute_with_llm(prompt=prompt, agent=selected_agent)
        execution_steps.append("generate_response")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        trace = Trace(
            trace_id=trace_id,
            task=task.query,
            agent=selected_agent.name,
            retrieval_hits=len(retrieved_context.docs),
            files_used=retrieved_context.files,
            timestamp=timestamp,
            steps=execution_steps,
            metadata={
                "agent_selection": selection_metadata,
                "retrieval": retrieval_metadata,
                "llm": llm_metadata,
                "context_tags": task.context_tags or [],
                "files_changed": task.files_changed or [],
            },
        )

        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="prompt_built",
            message="Built execution prompt.",
            metadata={
                "agent": selected_agent.name,
                "prompt_length": len(prompt),
                "latency_ms": latency_ms,
                "retrieval_hits": len(retrieved_context.docs),
                "llm_provider": llm_metadata.get("provider", "none"),
                "llm_model": llm_metadata.get("model", ""),
            },
        )
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="llm_response_generated",
            message="Generated downstream LLM response.",
            metadata={
                "provider": llm_metadata.get("provider", "none"),
                "model": llm_metadata.get("model", ""),
                "fallback": llm_metadata.get("fallback", False),
                "output_length": len(output),
            },
        )
        log_event(
            logger=self._logger,
            trace_id=trace_id,
            event_type="execution_complete",
            message="Completed traced context-engine execution.",
            metadata={
                "agent": selected_agent.name,
                "latency_ms": latency_ms,
                "retrieval_hits": len(retrieved_context.docs),
                "files_used": retrieved_context.files,
            },
        )

        return Response(
            prompt=prompt,
            agent_used=selected_agent.name,
            context_used={
                "files": retrieved_context.files,
                "docs": retrieved_context.docs,
                "metadata": {
                    **retrieved_context.metadata,
                    "llm": llm_metadata,
                },
            },
            trace_id=trace_id,
            trace=asdict(trace),
            latency_ms=latency_ms,
            output=output,
        )

    def _execute_with_llm(self, prompt: str, agent: Agent) -> tuple[str, dict[str, Any]]:
        """Execute the final prompt with Grok when configured, otherwise return a deterministic fallback."""

        if self._should_use_mock_execution():
            return prompt, {
                "provider": "mock",
                "model": "",
                "fallback": True,
                "reason": "mock_execution_enabled",
            }

        system_prompt = (
            f"You are {agent.name}. {agent.description} "
            "Answer using the supplied context. Be precise and concise."
        )
        result = self._llm_client.generate(prompt=prompt, system_prompt=system_prompt)
        return result.output, {
            **result.metadata,
            "fallback": False,
        }

    @staticmethod
    def _should_use_mock_execution() -> bool:
        mode = os.environ.get("CTX_EXECUTION_MODE", "").strip().lower()
        if mode == "mock":
            return True
        xai_api_key = os.environ.get("XAI_API_KEY", "").strip()
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        has_xai = bool(xai_api_key) and xai_api_key.lower() != "your_xai_api_key_here"
        has_gemini = bool(gemini_api_key) and gemini_api_key.lower() != "your_gemini_api_key_here"
        return not (has_xai or has_gemini)

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Routing config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise TypeError("Routing config root must be a JSON object.")
        return loaded

    @staticmethod
    def _resolve_knowledge_paths(
        config_path: Path,
        configured_paths: list[str],
        override_paths: list[str | Path] | None,
    ) -> list[Path]:
        base_dir = config_path.parent.parent.parent.resolve()
        raw_paths = override_paths if override_paths is not None else configured_paths
        resolved_paths: list[Path] = []
        for raw_path in raw_paths:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            resolved_paths.append(candidate)
        return resolved_paths

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
                AgentRoute(agent=agent, keywords=tuple(str(term).lower() for term in keywords))
            )
        return routes

    def _get_hybrid_retriever(self) -> HybridRetriever:
        """Lazily initialize a cached hybrid retriever for the configured knowledge paths."""

        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                documents=load_markdown_documents(self._knowledge_paths),
                knowledge_base_root=self._knowledge_base_root,
                default_top_k=self._max_results,
                memory=self._retrieval_memory,
            )
        return self._hybrid_retriever

    def record_retrieval_feedback(
        self,
        task: Task,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Persist retrieval feedback into memory for future boosting."""

        self._get_hybrid_retriever().update_feedback(
            task=task,
            selected_docs=selected_docs,
            retrieval_score=retrieval_score,
            observed_keywords=observed_keywords,
        )

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
            "Synthesize the retrieved context into a structured prompt.",
            "Execute downstream agent workflow using the prepared prompt.",
        ]

        if task.files_changed:
            steps.insert(1, "Prioritize context related to the changed files when available.")
        if task.context_tags:
            steps.insert(1, "Bias retrieval and planning toward the supplied context tags.")

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
