"""Corpus update and self-improvement loop utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..utils.logger import get_observability_logger, log_event
from .analyzer import analyze_codebase, detect_gaps
from .enrichment import (
    enrich_doc,
    estimate_enrichment_improvement,
    merge_docs,
    should_enrich,
    valid_structure,
)
from .generator import generate_context_doc

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"


def update_corpus(doc: str, path: str, knowledge_base_dir: str | Path | None = None) -> dict[str, str]:
    """Write a generated context document into the knowledge base without overwriting blindly."""

    base_dir = Path(knowledge_base_dir or KNOWLEDGE_BASE_DIR).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    target_path = (base_dir / path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        existing_content = target_path.read_text(encoding="utf-8")
        if existing_content == doc:
            return {"path": str(target_path), "status": "unchanged"}

        version = 2
        while True:
            versioned_path = target_path.with_stem(f"{target_path.stem}.v{version}")
            if not versioned_path.exists():
                versioned_path.write_text(doc, encoding="utf-8")
                return {"path": str(versioned_path), "status": "versioned"}
            version += 1

    target_path.write_text(doc, encoding="utf-8")
    return {"path": str(target_path), "status": "created"}


def improve_corpus(
    eval_report: dict[str, Any],
    root_path: str,
    knowledge_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Detect gaps from evaluation output and generate new corpus documents."""

    logger = get_observability_logger()
    gaps = detect_gaps(eval_report)
    log_event(
        logger=logger,
        trace_id="corpus-improvement",
        event_type="gaps_detected",
        message="Detected corpus gaps from evaluation report.",
        metadata={"gap_count": len(gaps)},
    )

    if not gaps:
        return {
            "gaps_detected": 0,
            "docs_created": [],
            "enrichment": [],
            "status": "no_changes",
        }

    code_context = analyze_codebase(root_path)
    created_docs: list[dict[str, str]] = []
    enrichment_events: list[dict[str, Any]] = []
    for gap in gaps:
        document = generate_context_doc(gap, code_context)
        enrichment_record: dict[str, Any] = {
            "task": gap.get("task", ""),
            "triggered": False,
        }
        if should_enrich(gap, gap):
            enrichment_record["triggered"] = True
            log_event(
                logger=logger,
                trace_id=gap.get("trace_id", "corpus-gap"),
                event_type="enrichment_triggered",
                message="Triggered LLM enrichment for a detected corpus gap.",
                metadata={
                    "task": gap.get("task", ""),
                    "missing_concepts": gap.get("missing_concepts", []),
                    "retrieval_score": gap.get("retrieval_score", 0.0),
                },
            )
            enriched = enrich_doc(gap, document)
            if valid_structure(enriched):
                merged = merge_docs(document, enriched)
                improvement = estimate_enrichment_improvement(
                    gap=gap,
                    before_doc=document,
                    after_doc=merged,
                    before_score=float(gap.get("retrieval_score", 0.0)),
                )
                document = merged
                enrichment_record.update(
                    {
                        "status": "success",
                        "enrichment_improvement": improvement,
                    }
                )
                log_event(
                    logger=logger,
                    trace_id=gap.get("trace_id", "corpus-gap"),
                    event_type="enrichment_success",
                    message="Completed LLM enrichment for generated context document.",
                    metadata={
                        "task": gap.get("task", ""),
                        "missing_concepts": gap.get("missing_concepts", []),
                        "enrichment_improvement": improvement,
                        "tokens_used": len(enriched.split()),
                    },
                )
            else:
                enrichment_record.update({"status": "discarded"})
                log_event(
                    logger=logger,
                    trace_id=gap.get("trace_id", "corpus-gap"),
                    event_type="enrichment_skipped",
                    message="Discarded LLM enrichment output due to invalid structure.",
                    metadata={
                        "task": gap.get("task", ""),
                        "missing_concepts": gap.get("missing_concepts", []),
                    },
                )
        else:
            enrichment_record.update({"status": "skipped"})
            log_event(
                logger=logger,
                trace_id=gap.get("trace_id", "corpus-gap"),
                event_type="enrichment_skipped",
                message="Skipped LLM enrichment for detected corpus gap.",
                metadata={
                    "task": gap.get("task", ""),
                    "missing_concepts": gap.get("missing_concepts", []),
                    "retrieval_score": gap.get("retrieval_score", 0.0),
                },
            )

        filename = _build_doc_filename(gap)
        update_result = update_corpus(
            doc=document,
            path=filename,
            knowledge_base_dir=knowledge_base_dir,
        )
        created_docs.append(update_result)
        enrichment_record["output_path"] = update_result["path"]
        enrichment_events.append(enrichment_record)
        log_event(
            logger=logger,
            trace_id=gap.get("trace_id", "corpus-gap"),
            event_type="context_doc_generated",
            message="Generated context document for detected retrieval gap.",
            metadata={
                "task": gap.get("task", ""),
                "missing_concepts": gap.get("missing_concepts", []),
                "output_path": update_result["path"],
                "status": update_result["status"],
            },
        )

    return {
        "gaps_detected": len(gaps),
        "docs_created": created_docs,
        "enrichment": enrichment_events,
        "status": "updated",
    }


def _build_doc_filename(gap: dict[str, Any]) -> str:
    concept = "-".join(
        "".join(character.lower() if character.isalnum() else "-" for character in str(item)).strip("-")
        for item in gap.get("missing_concepts", [])[:2]
    ).strip("-")
    if not concept:
        concept = hashlib.sha1(str(gap.get("task", "context-gap")).encode("utf-8")).hexdigest()[:12]
    return f"{concept or 'context-gap'}.md"
