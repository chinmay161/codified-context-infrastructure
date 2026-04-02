"""Persistent retrieval memory for feedback-aware ranking."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .adaptive_weights import update_weights_from_feedback
from .retrieval_v2 import tokenize_text

DEFAULT_MEMORY_PATH = Path(__file__).resolve().with_name("retrieval_memory_store.json")


def normalize_query(query: str) -> str:
    """Normalize a query into a deterministic signature."""

    return "|".join(sorted(set(tokenize_text(query))))


class RetrievalMemory:
    """Bounded persistent memory of successful and failed retrieval outcomes."""

    def __init__(
        self,
        memory_path: str | Path | None = None,
        max_entries: int = 200,
        max_docs_per_entry: int = 20,
    ) -> None:
        self._memory_path = Path(memory_path or DEFAULT_MEMORY_PATH).resolve()
        self._max_entries = max_entries
        self._max_docs_per_entry = max_docs_per_entry
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._load()

    def get(self, query: str) -> dict[str, Any]:
        """Return memory for a query signature."""

        signature = normalize_query(query)
        entry = self._store.get(signature)
        if entry is None:
            return _with_confidence(
                {
                    "successful_docs": [],
                    "failed_docs": [],
                    "boosted_docs": [],
                    "successful_keywords": [],
                    "success_count": 0,
                    "failure_count": 0,
                    "total_uses": 0,
                    "weight_adjustments": {},
                }
            )
        self._store.move_to_end(signature)
        return _with_confidence(
            {
                "successful_docs": list(entry.get("successful_docs", [])),
                "failed_docs": list(entry.get("failed_docs", [])),
                "boosted_docs": list(entry.get("boosted_docs", [])),
                "successful_keywords": list(entry.get("successful_keywords", [])),
                "success_count": int(entry.get("success_count", 0)),
                "failure_count": int(entry.get("failure_count", 0)),
                "total_uses": int(entry.get("total_uses", 0)),
                "weight_adjustments": dict(entry.get("weight_adjustments", {})),
            }
        )

    def update(
        self,
        query: str,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Update memory using evaluated retrieval outcomes."""

        signature = normalize_query(query)
        entry = self._store.get(
            signature,
            {
                "successful_docs": [],
                "failed_docs": [],
                "boosted_docs": [],
                "successful_keywords": [],
                "success_count": 0,
                "failure_count": 0,
                "total_uses": 0,
                "weight_adjustments": {},
            },
        )
        previous_score = _estimate_entry_score(entry)
        entry["total_uses"] = int(entry.get("total_uses", 0)) + 1

        if retrieval_score < 0.3:
            entry["failed_docs"] = _merge_limited([], selected_docs, self._max_docs_per_entry)
            entry["successful_docs"] = []
            entry["boosted_docs"] = []
            entry["successful_keywords"] = []
            entry["failure_count"] = int(entry.get("failure_count", 0)) + 1
        elif retrieval_score >= 0.6:
            entry["successful_docs"] = _merge_limited([], selected_docs, self._max_docs_per_entry)
            entry["boosted_docs"] = _merge_limited([], selected_docs, self._max_docs_per_entry)
            entry["successful_keywords"] = _merge_limited(
                [],
                [keyword.lower() for keyword in (observed_keywords or [])],
                self._max_docs_per_entry,
            )
            entry["failed_docs"] = []
            entry["success_count"] = int(entry.get("success_count", 0)) + 1
        else:
            entry["successful_docs"] = []
            entry["boosted_docs"] = []
            entry["successful_keywords"] = []
            entry["failed_docs"] = []

        entry["weight_adjustments"] = update_weights_from_feedback(
            query=query,
            performance={
                "retrieval_score": retrieval_score,
                "previous_score": previous_score,
                "memory_entry": entry,
            },
        )

        self._store[signature] = entry
        self._store.move_to_end(signature)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)
        self._save()

    def expand_query(self, query: str) -> list[str]:
        """Return successful keywords to use for query expansion."""

        entry = self.get(query)
        return list(entry.get("successful_keywords", []))

    def weight_adjustments(self, query: str) -> dict[str, float]:
        """Return learned adaptive weight adjustments for a query signature."""

        entry = self.get(query)
        adjustments = entry.get("weight_adjustments", {})
        if not isinstance(adjustments, dict):
            return {}
        return {str(key): float(value) for key, value in adjustments.items()}

    def snapshot(self) -> dict[str, Any]:
        """Return the current memory state."""

        return {key: value for key, value in self._store.items()}

    def _load(self) -> None:
        if not self._memory_path.exists():
            return
        try:
            payload = json.loads(self._memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        for key, value in payload.items():
            if isinstance(value, dict):
                self._store[key] = value

    def _save(self) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")


def _merge_limited(existing: list[str], new_values: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*new_values, *existing]:
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
        if len(merged) >= limit:
            break
    return merged


def _with_confidence(entry: dict[str, Any]) -> dict[str, Any]:
    total_uses = int(entry.get("total_uses", 0))
    success_count = int(entry.get("success_count", 0))
    confidence = round(success_count / total_uses, 4) if total_uses > 0 else 0.0
    return {
        **entry,
        "memory_confidence": confidence,
    }


def _estimate_entry_score(entry: dict[str, Any]) -> float:
    total_uses = int(entry.get("total_uses", 0))
    if total_uses <= 0:
        return 0.0
    success_count = int(entry.get("success_count", 0))
    return round(success_count / total_uses, 4)
