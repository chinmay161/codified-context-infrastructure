"""Structured JSON logging utilities for the context engine."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Formats log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "trace_id": getattr(record, "trace_id", ""),
            "event_type": getattr(record, "event_type", record.levelname.lower()),
            "message": record.getMessage(),
            "metadata": getattr(record, "metadata", {}),
        }
        return json.dumps(payload, ensure_ascii=True)


def get_observability_logger(
    log_dir: str | Path | None = None,
    logger_name: str = "ctx.observability",
) -> logging.Logger:
    """Create or reuse a logger that writes JSON logs to a rotating file."""

    resolved_log_dir = Path(log_dir or Path(__file__).resolve().parents[1] / "logs").resolve()
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / "context_engine.jsonl"

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=True,
    )
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    trace_id: str,
    event_type: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a structured observability event to the configured logger."""

    logger.info(
        message,
        extra={
            "trace_id": trace_id,
            "event_type": event_type,
            "metadata": metadata or {},
        },
    )
