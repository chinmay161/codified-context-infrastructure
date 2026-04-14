from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx import DefaultContextEngine, Task
from ctx.core.agent import load_agent
from ctx.core.distillation import compute_compile_confidence, distill_context
from ctx.core.models import RetrievedContext
from ctx.core.prompt_builder import build_adaptive_prompt
from ctx.core.query_analyzer import analyze_query
from ctx.core.skill_guidance import select_skill_guidance


def test_distillation_extracts_rules_api_and_relationships() -> None:
    retrieved = RetrievedContext(
        docs=[
            "Always validate the payload before enqueueing work.\nPOST /jobs creates the worker request.\nThe queue depends on the broker.",
        ],
        files=["C:/tmp/worker.md"],
        metadata={"ranking": [{"path": "C:/tmp/worker.md", "score": 0.8}]},
    )
    query_analysis = analyze_query("build a worker job endpoint")

    distilled = distill_context(
        task=Task(query="build a worker job endpoint"),
        retrieved_context=retrieved,
        query_analysis=query_analysis,
    )

    assert any("Always validate" in item for item in distilled.rule_extractions)
    assert any("POST /jobs" in item for item in distilled.api_summary)
    assert any("depends on" in item for item in distilled.relationship_map)
    assert distilled.priority_order == ["worker.md"]


def test_distillation_filters_table_rows_and_preserves_clean_snippets() -> None:
    retrieved = RetrievedContext(
        docs=[
            "# Auth\n\n"
            "| Symptom | Cause | Fix |\n"
            "| --- | --- | --- |\n"
            "| 403 | Role mismatch | Check JWT |\n"
            "Always verify JWT roles before permission checks.\n"
            "The auth service depends on the session store.\n"
            "This sentence is complete. This second sentence should be clipped cleanly before any abrupt truncation because it is intentionally long and descriptive for testing purposes.\n",
        ],
        files=["C:/tmp/authentication.md"],
        metadata={"ranking": [{"path": "C:/tmp/authentication.md", "score": 0.9}]},
    )

    distilled = distill_context(
        task=Task(query="debug auth role mismatch"),
        retrieved_context=retrieved,
        query_analysis=analyze_query("debug auth role mismatch"),
    )

    assert all("|" not in item for item in distilled.rule_extractions)
    assert all("|" not in item for item in distilled.relationship_map)
    assert distilled.priority_order == ["authentication.md"]
    assert distilled.evidence_snippets
    assert "|" not in distilled.evidence_snippets[0]
    assert not distilled.evidence_snippets[0].endswith(" role-ba")


def test_adaptive_prompt_uses_task_mode_and_confidence_strategy() -> None:
    agent = load_agent("draft_agent")
    query_analysis = analyze_query("debug queue retry failure")
    retrieved = RetrievedContext(
        docs=["Always inspect retries before changing worker logic."],
        files=["C:/tmp/retry.md"],
        metadata={"ranking": [{"path": "C:/tmp/retry.md", "score": 0.92}], "memory_hit": True},
    )
    distilled = distill_context(
        task=Task(query="debug queue retry failure"),
        retrieved_context=retrieved,
        query_analysis=query_analysis,
    )
    compile_confidence = compute_compile_confidence(retrieval_metadata=retrieved.metadata, distillation=distilled)

    prompt, budget = build_adaptive_prompt(
        query="debug queue retry failure",
        context="\n\n".join(retrieved.docs),
        agent=agent,
        query_analysis=query_analysis.as_dict(),
        task_mode=query_analysis.task_mode,
        distillation=distilled,
        compile_confidence=compile_confidence,
        skill_guidance=["Always find root cause before proposing fixes."],
    )

    assert "Prompt Strategy: debug" in prompt
    assert f"Confidence Mode: {compile_confidence['band']}" in prompt
    assert "Reasoning Policy:" in prompt
    assert "Fallback Policy:" in prompt
    assert "Guiding Heuristics:" in prompt
    assert "AGENT PROFILE" not in prompt
    assert "Name:" not in prompt
    assert "Available Tools:" not in prompt
    assert "Workflow (execute in order):" not in prompt
    assert budget["final_prompt_chars"] == len(prompt)


def test_engine_compile_adds_v3_metadata_and_template_sections(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "refactor-guide.md").write_text(
        "# Refactor Guide\n\n"
        "Always compare the current and target flows before changing architecture.\n"
        "handler(input: Request) -> Response\n"
        "The handler depends on the auth service.\n",
        encoding="utf-8",
    )

    engine = DefaultContextEngine(
        knowledge_path=knowledge_dir,
        storage_path=tmp_path,
    )
    compiled = engine.compile("refactor the authentication handler flow")

    assert compiled.metadata["task_mode"] == "refactor"
    assert "DISTILLED CONTEXT" in compiled.prompt
    assert "EVIDENCE" in compiled.prompt
    assert compiled.metadata["distillation"]["api_summary"]
    assert compiled.metadata["budget"]["final_prompt_chars"] == len(compiled.prompt)


def test_skill_guidance_selects_small_curated_subset_for_auth_debug_queries() -> None:
    analysis = analyze_query("debug authentication jwt role mismatch")
    guidance = select_skill_guidance(
        analysis=analysis,
        retrieval_metadata={
            "top_k": ["C:/tmp/authentication.md"],
            "graph_scores": [
                {"matched_concepts": ["authentication", "jwt", "role"]}
            ],
        },
    )

    assert 1 <= len(guidance) <= 5
    assert any("root cause" in item.lower() or "debug" in item.lower() for item in guidance)
    assert all(len(item) <= 120 for item in guidance)


def test_skill_guidance_uses_graph_before_query_fallback() -> None:
    analysis = analyze_query("please help")
    guidance = select_skill_guidance(
        analysis=analysis,
        retrieval_metadata={
            "top_k": ["C:/tmp/authentication.md"],
            "graph_scores": [{"matched_concepts": ["jwt", "session", "role"]}],
        },
    )

    assert guidance
    assert len(guidance) <= 5
