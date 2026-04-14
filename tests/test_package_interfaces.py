from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

import ctx
from ctx.api.server import create_app
from ctx.cli import build_parser
from ctx.core.models import Task


def test_package_exports_version_and_engine() -> None:
    assert ctx.__version__ == "0.3.0"
    assert ctx.DefaultContextEngine is not None


def test_cli_supports_feedback_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "feedback",
            "debug auth flow",
            "--selected-doc",
            "knowledge/auth.md",
            "--retrieval-score",
            "0.9",
            "--observed-keyword",
            "auth",
        ]
    )

    assert args.command == "feedback"
    assert args.selected_docs == ["knowledge/auth.md"]
    assert args.retrieval_score == 0.9
    assert args.observed_keywords == ["auth"]


def test_api_supports_feedback_endpoint() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/feedback",
        json={
            "task": "debug auth flow",
            "selected_docs": ["knowledge/auth.md"],
            "retrieval_score": 0.87,
            "observed_keywords": ["auth"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["retrieval_score"] == 0.87


def test_default_memory_path_persists_feedback_across_engine_instances(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    guide_path = knowledge_dir / "sync-debug-guide.md"
    guide_path.write_text(
        "# Sync Debug Guide\n\n"
        "Investigate sync failures, ranking issues, and debug retrieval behavior.\n",
        encoding="utf-8",
    )
    overview_path = knowledge_dir / "overview.md"
    overview_path.write_text(
        "# Overview\n\n"
        "General project notes.\n",
        encoding="utf-8",
    )

    storage_path = tmp_path / ".ctx" / "smartproctor"
    first_engine = ctx.DefaultContextEngine(
        knowledge_path=knowledge_dir,
        storage_path=storage_path,
        enable_memory=True,
    )
    first_engine.record_retrieval_feedback(
        task=Task(query="debug sync failures"),
        selected_docs=[str(guide_path)],
        retrieval_score=0.95,
        observed_keywords=["sync", "debug", "ranking"],
    )

    second_engine = ctx.DefaultContextEngine(
        knowledge_path=knowledge_dir,
        storage_path=storage_path,
        enable_memory=True,
    )
    retrieved = second_engine.retrieve(Task(query="optimize sync failure ranking"))

    assert retrieved.metadata["memory_hit"] is True
    assert retrieved.metadata["memory_match_type"] == "approximate"
    assert retrieved.files[0] == str(guide_path)
