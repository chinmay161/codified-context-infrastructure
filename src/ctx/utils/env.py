"""Minimal environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path | None = None) -> Path:
    """Load key=value pairs from a nearby .env file into process env."""

    resolved_path = _resolve_dotenv_path(dotenv_path)
    if not resolved_path.exists():
        return resolved_path

    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
    return resolved_path


def _resolve_dotenv_path(dotenv_path: str | Path | None) -> Path:
    if dotenv_path is not None:
        return Path(dotenv_path).expanduser().resolve()

    search_roots = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    visited: set[Path] = set()
    for root in search_roots:
        for candidate_dir in [root, *root.parents]:
            if candidate_dir in visited:
                continue
            visited.add(candidate_dir)
            candidate = candidate_dir / ".env"
            if candidate.exists():
                return candidate.resolve()

    return (Path.cwd().resolve() / ".env").resolve()
