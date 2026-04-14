from __future__ import annotations

import json
from pathlib import Path

from ctx.factory.cli import build_parser
from ctx.factory.smart_factory import SmartFactory


def test_smart_factory_creates_expected_project_structure(tmp_path: Path) -> None:
    factory = SmartFactory("my_project", "Flask authentication system", base_path=tmp_path)
    result = factory.run()

    project_root = tmp_path / "my_project"
    assert project_root.exists()
    assert result["project_root"] == str(project_root.resolve())

    expected_paths = [
        "ai-context/core_rules.md",
        "ai-context/patterns.md",
        "ai-context/examples.md",
        ".ctx/memory",
        ".ctx/logs",
        ".ctx/cache",
        "schemas/code_generation.json",
        "schemas/debugging.json",
        "config/factory_config.yaml",
        "src/factory/smart_factory.py",
        "src/factory/generators.py",
        "src/factory/templates.py",
        "cli.py",
        ".gitignore",
        "run_context_engine.py",
        "README.md",
    ]

    for relative_path in expected_paths:
        assert (project_root / relative_path).exists(), relative_path

    core_rules = (project_root / "ai-context/core_rules.md").read_text(encoding="utf-8")
    assert "Flask authentication system" in core_rules
    assert "## Constraints" in core_rules

    config_text = (project_root / "config/factory_config.yaml").read_text(encoding="utf-8")
    assert 'domain: "Flask authentication system"' in config_text
    assert 'default: "debugging"' in config_text
    assert "top_k: 15" in config_text

    gitignore_text = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert ".ctx/" in gitignore_text
    assert "__pycache__/" in gitignore_text
    assert "*.pyc" in gitignore_text
    assert ".env" in gitignore_text

    runner_text = (project_root / "run_context_engine.py").read_text(encoding="utf-8")
    assert 'knowledge_path="ai-context"' in runner_text
    assert 'storage_path=".ctx"' in runner_text
    assert 'engine.compile("your query here")' in runner_text

    code_generation_schema = json.loads(
        (project_root / "schemas/code_generation.json").read_text(encoding="utf-8")
    )
    assert code_generation_schema == {
        "files_to_create": [],
        "files_to_modify": [],
        "code": {},
        "summary": "",
    }

    debugging_schema = json.loads(
        (project_root / "schemas/debugging.json").read_text(encoding="utf-8")
    )
    assert debugging_schema == {
        "answer": "",
        "confidence": 0.0,
        "sources": [],
    }


def test_factory_cli_parses_init_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["init", "my_project", "--domain", "Flask authentication system", "--output-dir", "workspace"]
    )

    assert args.command == "init"
    assert args.project_name == "my_project"
    assert args.domain == "Flask authentication system"
    assert args.output_dir == "workspace"


def test_smart_factory_refuses_to_overwrite_existing_project(tmp_path: Path) -> None:
    factory = SmartFactory("my_project", "Flask authentication system", base_path=tmp_path)
    factory.run()

    duplicate = SmartFactory("my_project", "Flask authentication system", base_path=tmp_path)

    try:
        duplicate.run()
    except ValueError as exc:
        assert "Project already exists" in str(exc)
    else:
        raise AssertionError("Expected SmartFactory to reject an existing project.")
