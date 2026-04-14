"""Command line entrypoint for the context engine package."""

from __future__ import annotations

import argparse
import json

from .core.engine import DefaultContextEngine
from .core.models import Task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ctx", description="Run the codified context engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute a context query.")
    run_parser.add_argument("task", help="Task or query to execute.")
    run_parser.add_argument("--knowledge-path", dest="knowledge_path")
    run_parser.add_argument("--agents-config", dest="agents_config")
    run_parser.add_argument("--routing-config", dest="routing_config")
    run_parser.add_argument("--storage-path", dest="storage_path")
    run_parser.add_argument("--memory-path", dest="memory_path")
    run_parser.add_argument("--log-dir", dest="log_dir")
    run_parser.add_argument("--disable-memory", action="store_true")
    run_parser.add_argument("--json", action="store_true", dest="emit_json")

    feedback_parser = subparsers.add_parser("feedback", help="Record retrieval feedback for future ranking.")
    feedback_parser.add_argument("task", help="Original task or query.")
    feedback_parser.add_argument(
        "--selected-doc",
        dest="selected_docs",
        action="append",
        default=[],
        help="A document path that was relevant for the task. Repeat this flag to pass multiple docs.",
    )
    feedback_parser.add_argument(
        "--retrieval-score",
        dest="retrieval_score",
        type=float,
        required=True,
        help="Observed retrieval score in the range 0.0-1.0.",
    )
    feedback_parser.add_argument(
        "--observed-keyword",
        dest="observed_keywords",
        action="append",
        default=[],
        help="A keyword observed in the retrieved context. Repeat this flag to pass multiple keywords.",
    )
    feedback_parser.add_argument("--knowledge-path", dest="knowledge_path")
    feedback_parser.add_argument("--agents-config", dest="agents_config")
    feedback_parser.add_argument("--routing-config", dest="routing_config")
    feedback_parser.add_argument("--storage-path", dest="storage_path")
    feedback_parser.add_argument("--memory-path", dest="memory_path")
    feedback_parser.add_argument("--log-dir", dest="log_dir")
    feedback_parser.add_argument("--disable-memory", action="store_true")
    feedback_parser.add_argument("--json", action="store_true", dest="emit_json")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command not in {"run", "feedback"}:
        raise SystemExit(2)

    engine = DefaultContextEngine(
        knowledge_path=args.knowledge_path,
        agents_config=args.agents_config,
        routing_config=args.routing_config,
        enable_memory=not args.disable_memory,
        storage_path=args.storage_path,
        memory_path=args.memory_path,
        log_dir=args.log_dir,
    )

    if args.command == "feedback":
        engine.record_retrieval_feedback(
            task=Task(query=args.task),
            selected_docs=list(args.selected_docs),
            retrieval_score=max(0.0, min(1.0, float(args.retrieval_score))),
            observed_keywords=list(args.observed_keywords),
        )
        payload = {
            "status": "ok",
            "task": args.task,
            "selected_docs": list(args.selected_docs),
            "retrieval_score": max(0.0, min(1.0, float(args.retrieval_score))),
            "observed_keywords": list(args.observed_keywords),
        }
        if args.emit_json:
            print(json.dumps(payload, indent=2))
            return
        print("feedback recorded")
        return

    response = engine.execute(args.task)

    if args.emit_json:
        print(
            json.dumps(
                {
                    "output": response.output,
                    "trace_id": response.trace_id,
                    "agent_used": response.agent_used,
                    "latency_ms": response.latency_ms,
                    "files_used": response.context_used.get("files", []),
                },
                indent=2,
            )
        )
        return

    print(response.output)
    print(f"\ntrace_id={response.trace_id}")


if __name__ == "__main__":
    main()
