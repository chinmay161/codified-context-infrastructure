"""Static templates used by the Smart Factory."""

from __future__ import annotations

import json
import re
import textwrap

def render_factory_config(domain: str) -> str:
    title = domain_title(domain)
    return textwrap.dedent(
        f"""\
        domain: "{title}"

        retrieval:
          top_k: 15
          use_graph: true

        prompt:
          max_context_chars: 4000

        schema:
          default: "debugging"
        """
    )

CODE_GENERATION_SCHEMA = {
    "files_to_create": [],
    "files_to_modify": [],
    "code": {},
    "summary": "",
}

DEBUGGING_SCHEMA = {
    "answer": "",
    "confidence": 0.0,
    "sources": [],
}


def project_slug(project_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", project_name.strip()).strip("_").lower()
    return normalized or "context_project"


def domain_title(domain: str) -> str:
    compact = " ".join(domain.strip().split())
    return compact or "General software system"


def render_core_rules(domain: str) -> str:
    title = domain_title(domain)
    return textwrap.dedent(
        f"""\
        # Core Rules

        ## System Rules
        - Treat the knowledge base as the primary source of truth for {title}.
        - Prefer deterministic behavior, explicit contracts, and auditable state.
        - Keep responses grounded in project files, schemas, and configuration.

        ## Constraints
        - Do not assume undocumented workflows or hidden dependencies.
        - Minimize moving parts before adding new abstractions.
        - Preserve backward-compatible interfaces unless requirements say otherwise.

        ## Invariants
        - Inputs, outputs, and file names must stay predictable.
        - Configuration changes should be traceable and easy to review.
        - Examples should reinforce production-safe patterns for {title}.
        """
    )


def render_patterns(domain: str) -> str:
    title = domain_title(domain)
    return textwrap.dedent(
        f"""\
        # Patterns

        ## Common Patterns
        - Separate configuration, schema definitions, and runtime code paths.
        - Favor small modules with one clear responsibility.
        - Use explicit validation and stable file layouts for {title}.

        ## Best Practices
        - Keep domain language consistent across prompts, docs, and schemas.
        - Add concise examples that show the happy path and one failure mode.
        - Optimize for readability before cleverness.
        """
    )


def render_examples(domain: str) -> str:
    title = domain_title(domain)
    return textwrap.dedent(
        f"""\
        # Examples

        ## Example 1
        Task: Explain the main constraints in {title}.
        Expected context: core rules, config defaults, and schema contracts.

        ## Example 2
        Task: Debug a failing workflow in {title}.
        Expected context: debugging schema, known patterns, and relevant examples.

        ## Example 3
        Task: Propose a new implementation step for {title}.
        Expected context: code generation schema, invariants, and best practices.
        """
    )


def render_generated_templates(project_name: str, domain: str) -> dict[str, str]:
    slug = project_slug(project_name)
    title = domain_title(domain)
    return {
        "src/factory/templates.py": textwrap.dedent(
            f"""\
            \"\"\"Templates for the generated Smart Factory project.\"\"\"

            from __future__ import annotations

            import json
            import textwrap

            CODE_GENERATION_SCHEMA = {{
                "files_to_create": [],
                "files_to_modify": [],
                "code": {{}},
                "summary": "",
            }}

            DEBUGGING_SCHEMA = {{
                "answer": "",
                "confidence": 0.0,
                "sources": [],
            }}


            def build_factory_config(domain: str) -> str:
                domain = " ".join(domain.strip().split()) or "{title}"
                return textwrap.dedent(
                    f\"\"\"\\
                    domain: "{{domain}}"

                    retrieval:
                      top_k: 15
                      use_graph: true

                    prompt:
                      max_context_chars: 4000

                    schema:
                      default: "debugging"
                    \"\"\"
                )


            def build_knowledge_base(domain: str) -> dict[str, str]:
                domain = " ".join(domain.strip().split()) or "{title}"
                return {{
                    "ai-context/core_rules.md": textwrap.dedent(
                        f\"\"\"\\
                        # Core Rules

                        ## System Rules
                        - Treat the knowledge base as the primary source of truth for {{domain}}.
                        - Prefer deterministic behavior, explicit contracts, and auditable state.
                        - Keep responses grounded in project files, schemas, and configuration.

                        ## Constraints
                        - Do not assume undocumented workflows or hidden dependencies.
                        - Minimize moving parts before adding new abstractions.
                        - Preserve backward-compatible interfaces unless requirements say otherwise.

                        ## Invariants
                        - Inputs, outputs, and file names must stay predictable.
                        - Configuration changes should be traceable and easy to review.
                        - Examples should reinforce production-safe patterns for {{domain}}.
                        \"\"\"
                    ),
                    "ai-context/patterns.md": textwrap.dedent(
                        f\"\"\"\\
                        # Patterns

                        ## Common Patterns
                        - Separate configuration, schema definitions, and runtime code paths.
                        - Favor small modules with one clear responsibility.
                        - Use explicit validation and stable file layouts for {{domain}}.

                        ## Best Practices
                        - Keep domain language consistent across prompts, docs, and schemas.
                        - Add concise examples that show the happy path and one failure mode.
                        - Optimize for readability before cleverness.
                        \"\"\"
                    ),
                    "ai-context/examples.md": textwrap.dedent(
                        f\"\"\"\\
                        # Examples

                        ## Example 1
                        Task: Explain the main constraints in {{domain}}.
                        Expected context: core rules, config defaults, and schema contracts.

                        ## Example 2
                        Task: Debug a failing workflow in {{domain}}.
                        Expected context: debugging schema, known patterns, and relevant examples.

                        ## Example 3
                        Task: Propose a new implementation step for {{domain}}.
                        Expected context: code generation schema, invariants, and best practices.
                        \"\"\"
                    ),
                }}


            def build_schema_templates() -> dict[str, str]:
                return {{
                    "schemas/code_generation.json": json.dumps(CODE_GENERATION_SCHEMA, indent=2) + "\\n",
                    "schemas/debugging.json": json.dumps(DEBUGGING_SCHEMA, indent=2) + "\\n",
                }}
            """
        ),
        "src/factory/generators.py": textwrap.dedent(
            '''\
            """Helpers for deterministic file generation."""

            from __future__ import annotations

            from pathlib import Path


            def ensure_directory(path: Path) -> None:
                path.mkdir(parents=True, exist_ok=True)


            def write_text(path: Path, content: str) -> None:
                ensure_directory(path.parent)
                path.write_text(content, encoding="utf-8")
            '''
        ),
        "src/factory/smart_factory.py": textwrap.dedent(
            f"""\
            \"\"\"Deterministic Smart Factory for provisioning a Context Engine project.\"\"\"

            from __future__ import annotations

            from pathlib import Path

            from .generators import ensure_directory, write_text
            from .templates import build_factory_config, build_knowledge_base, build_schema_templates


            class SmartFactory:
                def __init__(self, project_name: str, domain: str):
                    self.project_name = project_name
                    self.domain = domain
                    self.project_root = Path(project_name).resolve()

                def create_project(self):
                    if self.project_root.exists():
                        raise ValueError(f"Project already exists: {{self.project_root}}")
                    for relative_dir in (
                        "ai-context",
                        ".ctx/memory",
                        ".ctx/logs",
                        ".ctx/cache",
                        "schemas",
                        "config",
                        "src/factory",
                    ):
                        ensure_directory(self.project_root / relative_dir)

                def generate_knowledge_base(self):
                    for relative_path, content in build_knowledge_base(self.domain).items():
                        write_text(self.project_root / relative_path, content)

                def generate_config(self):
                    write_text(
                        self.project_root / "config/factory_config.yaml",
                        build_factory_config(self.domain),
                    )

                def generate_schemas(self):
                    for relative_path, content in build_schema_templates().items():
                        write_text(self.project_root / relative_path, content)

                def generate_support_files(self):
                    write_text(
                        self.project_root / ".gitignore",
                        ".ctx/\\n__pycache__/\\n*.pyc\\n.env\\n",
                    )
                    write_text(
                        self.project_root / "run_context_engine.py",
                        \"\"\"from ctx.core.engine import DefaultContextEngine\\n\\nengine = DefaultContextEngine(\\n    knowledge_path=\\\"ai-context\\\",\\n    storage_path=\\\".ctx\\\"\\n)\\n\\nresult = engine.compile(\\\"your query here\\\")\\nprint(result.prompt)\\n\"\"\",
                    )
                    write_text(
                        self.project_root / "cli.py",
                        \"\"\"from factory.smart_factory import SmartFactory\\n\\n\\nif __name__ == '__main__':\\n    SmartFactory('{slug}', '{title}').run()\\n\"\"\",
                    )
                    write_text(
                        self.project_root / "README.md",
                        \"\"\"# {project_name}\\n\\nGenerated Smart Factory project for {title}.\\n\"\"\",
                    )

                def run(self):
                    self.create_project()
                    self.generate_knowledge_base()
                    self.generate_config()
                    self.generate_schemas()
                    self.generate_support_files()
            """
        ),
        "cli.py": textwrap.dedent(
            f"""\
            \"\"\"CLI entrypoint for the generated Smart Factory project.\"\"\"

            from __future__ import annotations

            import argparse

            from src.factory.smart_factory import SmartFactory


            def build_parser() -> argparse.ArgumentParser:
                parser = argparse.ArgumentParser(prog="ctx-factory")
                parser.add_argument("command", choices=["init"])
                parser.add_argument("project_name", nargs="?", default="{slug}")
                parser.add_argument("--domain", dest="domain", default="{title}")
                return parser


            def main() -> None:
                args = build_parser().parse_args()
                SmartFactory(args.project_name, args.domain).run()
                print(f"Initialized {{args.project_name}} for {{args.domain}}.")


            if __name__ == "__main__":
                main()
            """
        ),
        ".gitignore": ".ctx/\\n__pycache__/\\n*.pyc\\n.env\\n",
        "run_context_engine.py": textwrap.dedent(
            """\
            from ctx.core.engine import DefaultContextEngine

            engine = DefaultContextEngine(
                knowledge_path="ai-context",
                storage_path=".ctx"
            )

            result = engine.compile("your query here")
            print(result.prompt)
            """
        ),
        "README.md": textwrap.dedent(
            f"""\
            # {project_name}

            Deterministic Smart Factory scaffold for a Context Engine workspace.

            ## Usage
            - CLI: `python cli.py init {slug} --domain "{title}"`
            - Python:
              `from src.factory.smart_factory import SmartFactory`
              `SmartFactory("{slug}", "{title}").run()`
            - Engine helper:
              `python run_context_engine.py`
            """
        ),
    }


def render_schema_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2) + "\n"
