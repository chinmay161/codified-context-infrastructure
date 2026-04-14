"""FastAPI service for the reusable context engine."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..core.engine import DefaultContextEngine
from ..core.models import Task


class QueryRequest(BaseModel):
    task: str = Field(..., min_length=1)
    files_changed: list[str] | None = None
    context_tags: list[str] | None = None


class QueryResponse(BaseModel):
    output: str
    trace_id: str
    agent_used: str
    latency_ms: float
    files_used: list[str]


class FeedbackRequest(BaseModel):
    task: str = Field(..., min_length=1)
    selected_docs: list[str] = Field(default_factory=list)
    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    observed_keywords: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    status: str
    task: str
    selected_docs: list[str]
    retrieval_score: float
    observed_keywords: list[str]


@lru_cache(maxsize=1)
def get_engine() -> DefaultContextEngine:
    return DefaultContextEngine(
        knowledge_path=os.environ.get("CTX_KNOWLEDGE_PATH"),
        agents_config=os.environ.get("CTX_AGENTS_CONFIG"),
        routing_config=os.environ.get("CTX_ROUTING_CONFIG"),
        enable_memory=os.environ.get("CTX_ENABLE_MEMORY", "true").strip().lower() not in {"0", "false", "no"},
        storage_path=os.environ.get("CTX_STORAGE_PATH"),
        memory_path=os.environ.get("CTX_MEMORY_PATH"),
        log_dir=os.environ.get("CTX_LOG_DIR"),
        dotenv_path=os.environ.get("CTX_DOTENV_PATH"),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Codified Context API", version="0.2.0")

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        response = get_engine().execute(
            Task(
                query=request.task,
                files_changed=request.files_changed,
                context_tags=request.context_tags,
            )
        )
        return QueryResponse(
            output=response.output,
            trace_id=response.trace_id,
            agent_used=response.agent_used,
            latency_ms=response.latency_ms,
            files_used=response.context_used.get("files", []),
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        get_engine().record_retrieval_feedback(
            task=Task(query=request.task),
            selected_docs=list(request.selected_docs),
            retrieval_score=float(request.retrieval_score),
            observed_keywords=list(request.observed_keywords),
        )
        return FeedbackResponse(
            status="ok",
            task=request.task,
            selected_docs=list(request.selected_docs),
            retrieval_score=float(request.retrieval_score),
            observed_keywords=list(request.observed_keywords),
        )

    return app


app = create_app()
