"""Gap detection and repository analysis for corpus expansion."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def detect_gaps(eval_report: dict[str, Any], threshold: float = 0.45) -> list[dict[str, Any]]:
    """Detect knowledge gaps from evaluation results."""

    gaps: list[dict[str, Any]] = []
    for result in eval_report.get("results", []):
        failure_mode = str(result.get("failure_mode", ""))
        retrieval_score = float(result.get("retrieval_score", 0.0))
        if failure_mode != "retrieval_failure" and retrieval_score >= threshold:
            continue

        expected_keywords = [str(keyword) for keyword in result.get("expected_keywords", []) if str(keyword).strip()]
        observed_keywords = {str(keyword).lower() for keyword in result.get("observed_keywords", [])}
        missing_concepts = [keyword for keyword in expected_keywords if keyword.lower() not in observed_keywords]
        if not missing_concepts:
            missing_concepts = expected_keywords or [str(result.get("task", "missing-context")).strip()]

        gaps.append(
            {
                "task": str(result.get("task", "")),
                "failure_mode": failure_mode,
                "missing_concepts": missing_concepts,
                "expected_domain": _infer_domain(result),
                "trace_id": str(result.get("trace_id", "")),
                "retrieval_score": retrieval_score,
            }
        )
    return gaps


def analyze_codebase(root_path: str) -> dict[str, Any]:
    """Analyze the repository structure, Python symbols, and file relationships."""

    root = Path(root_path).resolve()
    python_files = sorted(root.rglob("*.py"))
    markdown_files = sorted(root.rglob("*.md"))

    modules: list[dict[str, Any]] = []
    routes: list[dict[str, str]] = []
    relationships: list[dict[str, str]] = []
    patterns: set[str] = set()

    for file_path in python_files:
        relative_path = file_path.relative_to(root).as_posix()
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports.extend(
            module_name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for module_name in ([node.module] if node.module else [])
        )

        for import_name in imports:
            relationships.append({"source": relative_path, "target": import_name})

        module_entry = {
            "path": relative_path,
            "functions": functions,
            "classes": classes,
            "imports": imports,
        }
        modules.append(module_entry)

        if any(name.startswith(("get_", "post_", "put_", "delete_")) for name in functions):
            patterns.add("handler_functions")
        if classes:
            patterns.add("object_oriented_modules")
        if any("dataclass" in import_name for import_name in imports):
            patterns.add("dataclass_models")
        if any("ABC" in import_name for import_name in imports):
            patterns.add("interface_abstractions")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                route_path = _extract_route_decorator(decorator)
                if route_path:
                    routes.append({"file": relative_path, "function": node.name, "route": route_path})

    return {
        "root": str(root),
        "modules": modules,
        "markdown_files": [path.relative_to(root).as_posix() for path in markdown_files],
        "routes": routes,
        "patterns": sorted(patterns),
        "file_relationships": relationships,
    }


def _infer_domain(result: dict[str, Any]) -> str:
    task = str(result.get("task", "")).lower()
    keywords = " ".join(str(keyword).lower() for keyword in result.get("expected_keywords", []))
    combined = f"{task} {keywords}"
    if any(token in combined for token in ("auth", "login", "jwt", "token")):
        return "auth"
    if any(token in combined for token in ("database", "sql", "query", "model")):
        return "database"
    if any(token in combined for token in ("workflow", "repository", "process", "pipeline")):
        return "workflow"
    if any(token in combined for token in ("architecture", "system", "context", "engine")):
        return "architecture"
    return "general"


def _extract_route_decorator(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
        if decorator.func.attr.lower() in {"get", "post", "put", "delete", "patch"} and decorator.args:
            first_arg = decorator.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                return first_arg.value
    return None
