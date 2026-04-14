"""Deterministic scaffolding tool for Context Engine projects."""

from __future__ import annotations

from pathlib import Path

from .generators import GeneratedFile, ensure_directory, write_text
from .templates import (
    CODE_GENERATION_SCHEMA,
    DEBUGGING_SCHEMA,
    render_factory_config,
    render_core_rules,
    render_examples,
    render_generated_templates,
    render_patterns,
    render_schema_json,
)


class SmartFactory:
    """Create a minimal Context Engine project scaffold."""

    def __init__(self, project_name: str, domain: str, base_path: str | Path | None = None):
        cleaned_name = project_name.strip()
        cleaned_domain = " ".join(domain.strip().split())
        if not cleaned_name:
            raise ValueError("project_name must not be empty.")
        if not cleaned_domain:
            raise ValueError("domain must not be empty.")

        self.project_name = cleaned_name
        self.domain = cleaned_domain
        root_parent = Path(base_path).expanduser().resolve() if base_path is not None else Path.cwd()
        self.project_root = (root_parent / self.project_name).resolve()

    def create_project(self) -> list[GeneratedFile]:
        if self.project_root.exists():
            raise ValueError(f"Project already exists: {self.project_root}")
        directories = (
            "ai-context",
            ".ctx/memory",
            ".ctx/logs",
            ".ctx/cache",
            "schemas",
            "config",
            "src/factory",
        )
        for relative_dir in directories:
            ensure_directory(self.project_root / relative_dir)
        return [
            GeneratedFile(path=f"{self.project_name}/{relative_dir}", description="Created directory.")
            for relative_dir in directories
        ]

    def generate_knowledge_base(self) -> list[GeneratedFile]:
        files = {
            "ai-context/core_rules.md": (render_core_rules(self.domain), "System rules, constraints, and invariants."),
            "ai-context/patterns.md": (render_patterns(self.domain), "Common patterns and best practices."),
            "ai-context/examples.md": (render_examples(self.domain), "Small domain-specific examples."),
        }
        written: list[GeneratedFile] = []
        for relative_path, (content, description) in files.items():
            write_text(self.project_root / relative_path, content)
            written.append(GeneratedFile(path=f"{self.project_name}/{relative_path}", description=description))
        return written

    def generate_config(self) -> list[GeneratedFile]:
        relative_path = "config/factory_config.yaml"
        write_text(self.project_root / relative_path, render_factory_config(self.domain))
        return [
            GeneratedFile(
                path=f"{self.project_name}/{relative_path}",
                description="Factory configuration with retrieval, prompt, and schema defaults.",
            )
        ]

    def generate_schemas(self) -> list[GeneratedFile]:
        files = {
            "schemas/code_generation.json": (
                render_schema_json(CODE_GENERATION_SCHEMA),
                "Strict schema template for code generation responses.",
            ),
            "schemas/debugging.json": (
                render_schema_json(DEBUGGING_SCHEMA),
                "Strict schema template for debugging responses.",
            ),
        }
        written: list[GeneratedFile] = []
        for relative_path, (content, description) in files.items():
            write_text(self.project_root / relative_path, content)
            written.append(GeneratedFile(path=f"{self.project_name}/{relative_path}", description=description))
        return written

    def generate_support_files(self) -> list[GeneratedFile]:
        generated_templates = render_generated_templates(self.project_name, self.domain)
        written: list[GeneratedFile] = []
        descriptions = {
            "src/factory/templates.py": "Template helpers embedded into the generated project.",
            "src/factory/generators.py": "Filesystem helpers for the generated project.",
            "src/factory/smart_factory.py": "SmartFactory implementation for the generated project.",
            "cli.py": "CLI entrypoint for initializing the generated project.",
            ".gitignore": "Ignore rules for generated local state and Python artifacts.",
            "run_context_engine.py": "Helper script for compiling a prompt with the generated project.",
            "README.md": "Usage guide for the generated project.",
        }
        for relative_path, content in generated_templates.items():
            write_text(self.project_root / relative_path, content)
            written.append(
                GeneratedFile(
                    path=f"{self.project_name}/{relative_path}",
                    description=descriptions[relative_path],
                )
            )
        return written

    def run(self) -> dict[str, object]:
        created = []
        created.extend(self.create_project())
        created.extend(self.generate_knowledge_base())
        created.extend(self.generate_config())
        created.extend(self.generate_schemas())
        created.extend(self.generate_support_files())
        return {
            "summary": f"Initialized '{self.project_name}' with a deterministic Context Engine scaffold for {self.domain}.",
            "project_root": str(self.project_root),
            "files": created,
        }
