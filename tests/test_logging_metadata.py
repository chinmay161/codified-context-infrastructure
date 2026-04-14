from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx import DefaultContextEngine


def test_execute_exposes_v3_metadata_and_prompt_output(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "debug-guide.md").write_text(
        "# Debug Guide\n\n"
        "Always verify the failing path before changing the implementation.\n"
        "POST /sessions opens the authentication flow.\n"
        "The retry worker depends on the queue service.\n",
        encoding="utf-8",
    )

    engine = DefaultContextEngine(
        knowledge_path=knowledge_dir,
        storage_path=tmp_path,
    )
    response = engine.execute("debug authentication retry failure")

    metadata = response.context_used["metadata"]
    assert response.output == response.prompt
    assert metadata["task_mode"] == "debug"
    assert metadata["query_analysis"]["task_mode"] == "debug"
    assert metadata["compile_confidence"]["band"] in {"low", "medium", "high"}
    assert "distillation" in metadata
    assert "budget" in metadata
    assert metadata["distillation"]["rule_extractions"]


def test_compile_metadata_records_budget_decisions(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    for index in range(5):
        (knowledge_dir / f"guide-{index}.md").write_text(
            "# Guide\n\n"
            "Always keep constraints explicit in the implementation plan.\n"
            "field_name: string\n"
            "Service A depends on Service B.\n",
            encoding="utf-8",
        )

    engine = DefaultContextEngine(
        knowledge_path=knowledge_dir,
        storage_path=tmp_path,
    )
    compiled = engine.compile("build a safer implementation plan for service changes")

    assert compiled.metadata["budget"]["final_prompt_chars"] > 0
    assert "preserved_evidence" in compiled.metadata["budget"]
    assert compiled.metadata["distillation"]["priority_order"]
