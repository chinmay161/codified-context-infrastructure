"""Utilities for loading markdown knowledge sources from disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True, frozen=True)
class MarkdownDocument:
    """A markdown document loaded from a configured source path."""

    path: str
    content: str


def load_markdown_documents(paths: Iterable[str | Path]) -> list[MarkdownDocument]:
    """Load markdown files from directories or individual file paths."""

    documents: list[MarkdownDocument] = []
    seen_paths: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue

        if path.is_file() and path.suffix.lower() == ".md":
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(candidate.resolve() for candidate in path.rglob("*.md"))
        else:
            continue

        for candidate in candidates:
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            documents.append(
                MarkdownDocument(
                    path=str(candidate),
                    content=candidate.read_text(encoding="utf-8"),
                )
            )

    return documents
