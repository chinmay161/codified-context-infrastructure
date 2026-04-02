"""Minimal environment loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path | None = None) -> Path:
    """Load key=value pairs from a local .env file into process env."""

    resolved_path = Path(dotenv_path or Path(__file__).resolve().parents[2] / ".env").resolve()
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
