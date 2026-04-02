"""Batch runner for context engine evaluation test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..corpus.updater import improve_corpus
from ..core.engine import DefaultContextEngine
from ..core.models import Agent, RetrievedContext
from ..core.models import Response, Task
from ..core.retrieval_v2 import tokenize_text
from ..utils.file_loader import load_markdown_documents
from .evaluator import Evaluator


REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_THRESHOLD = 0.75


class BaselineKeywordContextEngine(DefaultContextEngine):
    """Baseline engine for A/B evaluation using simple keyword-overlap retrieval."""

    def _retrieve_hybrid(self, task: Task, agent: Agent) -> RetrievedContext:
        documents = load_markdown_documents(self._knowledge_paths)
        query_terms = self._task_terms(task)
        document_paths = [document.path for document in documents]

        if not documents or not query_terms:
            return RetrievedContext(
                docs=[],
                files=[],
                metadata={
                    "strategy": "keyword_markdown_search_v1",
                    "matched_terms": sorted(query_terms),
                    "scores": {},
                    "ranking": [],
                    "top_k": [],
                    "keyword_hits": {},
                    "files_considered": document_paths,
                    "scoring_breakdown": {},
                },
            )

        scored_documents: list[tuple[int, str, str, list[str]]] = []
        for document in documents:
            document_terms = set(self._task_terms(Task(query=document.content)))
            filename_terms = set(self._task_terms(Task(query=Path(document.path).stem)))
            overlap = sorted(query_terms.intersection(document_terms.union(filename_terms)))
            if not overlap:
                continue
            scored_documents.append((len(overlap), document.path, document.content, overlap))

        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        selected = scored_documents[: self._max_results]

        return RetrievedContext(
            docs=[item[2] for item in selected],
            files=[item[1] for item in selected],
            metadata={
                "strategy": "keyword_markdown_search_v1",
                "matched_terms": sorted(query_terms),
                "scores": {item[1]: item[0] for item in selected},
                "ranking": [
                    {"rank": index, "path": item[1], "score": item[0]}
                    for index, item in enumerate(selected, start=1)
                ],
                "top_k": [item[1] for item in selected],
                "keyword_hits": {item[1]: item[3] for item in selected},
                "files_considered": document_paths,
                "scoring_breakdown": {
                    item[1]: {"keyword": item[0], "total": item[0]}
                    for item in selected
                },
            },
        )

    def record_retrieval_feedback(
        self,
        task: Task,
        selected_docs: list[str],
        retrieval_score: float,
        observed_keywords: list[str] | None = None,
    ) -> None:
        """Keep baseline evaluation deterministic without adaptive feedback."""
        return None


def load_test_cases(test_case_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load evaluation test cases from JSON."""

    resolved_path = Path(test_case_path or Path(__file__).resolve().parent / "test_cases.json").resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise TypeError("Evaluation test cases must be a JSON list.")
    return cases


def build_engine(engine_name: str) -> DefaultContextEngine:
    """Build a named engine variant for evaluation or comparison."""

    normalized = engine_name.lower()
    if normalized == "v1":
        return BaselineKeywordContextEngine()
    if normalized in {"v2", "v3"}:
        return DefaultContextEngine()
    raise ValueError(f"Unsupported engine variant: {engine_name}")


def execute_case(engine: DefaultContextEngine, case: dict[str, Any]) -> Response:
    """Execute a single evaluation case through the context engine."""

    task = Task(
        query=str(case["task"]),
        files_changed=list(case.get("files_changed", []) or []) or None,
        context_tags=list(case.get("context_tags", []) or []) or None,
    )
    return engine.execute(task)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate evaluation metrics into summary scores."""

    if not results:
        return {
            "avg_retrieval_score": 0.0,
            "routing_score_pct": 0.0,
            "trajectory_score": 0.0,
            "avg_context_utilization": 0.0,
            "avg_latency_ms": 0.0,
            "avg_latency_score": 0.0,
            "avg_overall_score": 0.0,
            "failure_breakdown": {},
        }

    total = len(results)
    failure_breakdown: dict[str, int] = {}
    for item in results:
        failure_mode = str(item["failure_mode"])
        failure_breakdown[failure_mode] = failure_breakdown.get(failure_mode, 0) + 1

    return {
        "avg_retrieval_score": round(sum(item["retrieval_score"] for item in results) / total, 4),
        "routing_score_pct": round(
            100.0 * sum(item["routing_score"] for item in results) / total,
            2,
        ),
        "trajectory_score": round(sum(item["trajectory_score"] for item in results) / total, 4),
        "avg_context_utilization": round(
            sum(item["context_utilization"] for item in results) / total,
            4,
        ),
        "avg_latency_ms": round(sum(item["latency_ms"] for item in results) / total, 3),
        "avg_latency_score": round(sum(item["latency_score"] for item in results) / total, 4),
        "avg_overall_score": round(sum(item["overall_score"] for item in results) / total, 4),
        "failure_breakdown": failure_breakdown,
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Format aggregated evaluation results for terminal output."""

    failure_breakdown = summary.get("failure_breakdown", {})
    failure_text = ", ".join(f"{key}={value}" for key, value in sorted(failure_breakdown.items())) or "none"
    return "\n".join(
        [
            "Evaluation Summary:",
            f"* Avg Retrieval Score: {summary['avg_retrieval_score']:.2f}",
            f"* Routing Score: {summary['routing_score_pct']:.2f}%",
            f"* Avg Trajectory Score: {summary['trajectory_score']:.2f}",
            f"* Avg Context Utilization: {summary['avg_context_utilization']:.2f}",
            f"* Avg Latency: {summary['avg_latency_ms']:.3f}ms",
            f"* Avg Latency Score: {summary['avg_latency_score']:.2f}",
            f"* Avg Overall Score: {summary['avg_overall_score']:.2f}",
            f"* Failure Breakdown: {failure_text}",
        ]
    )


def run_evaluations(
    engine: DefaultContextEngine | None = None,
    test_case_path: str | Path | None = None,
    engine_name: str = "v2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the configured evaluation suite and return detailed and summary results."""

    active_engine = engine or build_engine(engine_name)
    evaluator = Evaluator()
    test_cases = load_test_cases(test_case_path)

    results: list[dict[str, Any]] = []
    for case in test_cases:
        task = Task(
            query=str(case["task"]),
            files_changed=list(case.get("files_changed", []) or []) or None,
            context_tags=list(case.get("context_tags", []) or []) or None,
        )
        response = active_engine.execute(task)
        result = evaluator.evaluate(response, case)
        results.append(result)
        relevant_docs = _select_feedback_docs(response=response, case=case)
        active_engine.record_retrieval_feedback(
            task=task,
            selected_docs=relevant_docs,
            retrieval_score=float(result["retrieval_score"]),
            observed_keywords=list(result.get("observed_keywords", [])),
        )

    return results, summarize_results(results)


def _select_feedback_docs(response: Response, case: dict[str, Any]) -> list[str]:
    """Choose memory feedback docs that are strongly aligned with the task intent."""

    expected_keywords = {str(keyword).lower() for keyword in case.get("expected_keywords", [])}
    if not expected_keywords:
        return list(response.context_used.get("files", []))

    task_terms = set(tokenize_text(str(case.get("task", ""))))
    docs = list(response.context_used.get("docs", []))
    files = list(response.context_used.get("files", []))
    keyword_hits = dict(response.context_used.get("metadata", {}).get("keyword_hits", {}))

    selected_docs: list[str] = []
    for path, doc in zip(files, docs):
        hit_terms = {str(hit).lower() for hit in keyword_hits.get(path, [])}
        hit_overlap = expected_keywords.intersection(hit_terms)
        if not hit_overlap:
            continue

        path_terms = set(tokenize_text(Path(path).stem))
        origin_task = _extract_origin_task(doc)
        origin_terms = set(tokenize_text(origin_task))
        path_overlap = expected_keywords.intersection(path_terms)
        origin_overlap = task_terms.intersection(origin_terms)
        expected_hit_ratio = len(hit_overlap) / max(len(expected_keywords), 1)
        origin_overlap_ratio = len(origin_overlap) / max(len(task_terms), 1)

        if path_overlap or expected_hit_ratio >= 0.5 or origin_overlap_ratio >= 0.5:
            selected_docs.append(path)

    return selected_docs


def _extract_origin_task(document: str) -> str:
    """Extract the originating task string from generated context docs when present."""

    marker = "task '"
    lowered = document.lower()
    start = lowered.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = document.find("'", start)
    if end == -1:
        return ""
    return document[start:end].strip()


def compare_summaries(
    baseline_name: str,
    baseline_summary: dict[str, Any],
    candidate_name: str,
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compute a delta report between two evaluation summaries."""

    return {
        "baseline_engine": baseline_name,
        "candidate_engine": candidate_name,
        "deltas": {
            "retrieval_score": round(
                candidate_summary["avg_retrieval_score"] - baseline_summary["avg_retrieval_score"],
                4,
            ),
            "routing_score_pct": round(
                candidate_summary["routing_score_pct"] - baseline_summary["routing_score_pct"],
                2,
            ),
            "trajectory_score": round(
                candidate_summary["trajectory_score"] - baseline_summary["trajectory_score"],
                4,
            ),
            "latency_ms": round(
                candidate_summary["avg_latency_ms"] - baseline_summary["avg_latency_ms"],
                3,
            ),
            "overall_score": round(
                candidate_summary["avg_overall_score"] - baseline_summary["avg_overall_score"],
                4,
            ),
        },
    }


def format_comparison(report: dict[str, Any]) -> str:
    """Format an A/B comparison report for console output."""

    deltas = report["deltas"]
    return "\n".join(
        [
            "Evaluation Delta:",
            (
                f"* Retrieval Score: {report['baseline_engine']} -> {report['candidate_engine']} "
                f"({deltas['retrieval_score']:+.2f})"
            ),
            (
                f"* Routing Score: {report['baseline_engine']} -> {report['candidate_engine']} "
                f"({deltas['routing_score_pct']:+.2f}%)"
            ),
            (
                f"* Trajectory Score: {report['baseline_engine']} -> {report['candidate_engine']} "
                f"({deltas['trajectory_score']:+.2f})"
            ),
            (
                f"* Avg Latency: {report['baseline_engine']} -> {report['candidate_engine']} "
                f"({deltas['latency_ms']:+.3f}ms)"
            ),
            (
                f"* Overall Score: {report['baseline_engine']} -> {report['candidate_engine']} "
                f"({deltas['overall_score']:+.2f})"
            ),
        ]
    )


def write_report(report: dict[str, Any], output_path: str | Path | None = None) -> Path:
    """Write a JSON evaluation report to disk."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(output_path or REPORTS_DIR / "latest.json").resolve()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def build_report(
    engine_name: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    corpus_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured evaluation report."""

    return {
        "engine": engine_name,
        "summary": summary,
        "results": results,
        "comparison": comparison,
        "corpus_update": corpus_update,
    }


def evaluate_threshold(summary: dict[str, Any], threshold: float) -> bool:
    """Return True when the evaluation passes the minimum quality threshold."""

    return float(summary["avg_overall_score"]) >= threshold


def evaluate_regression(comparison: dict[str, Any] | None) -> bool:
    """Return True when the candidate does not regress on overall score."""

    if comparison is None:
        return True
    return float(comparison["deltas"]["overall_score"]) >= 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context engine evaluation runner.")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "compare"])
    parser.add_argument("--engine", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--baseline", default="v1", choices=["v1", "v2", "v3"])
    parser.add_argument("--candidate", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--test-cases", default=None)
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.command == "compare":
        baseline_results, baseline_summary = run_evaluations(
            test_case_path=args.test_cases,
            engine_name=args.baseline,
        )
        candidate_results, candidate_summary = run_evaluations(
            test_case_path=args.test_cases,
            engine_name=args.candidate,
        )
        comparison = compare_summaries(
            baseline_name=args.baseline,
            baseline_summary=baseline_summary,
            candidate_name=args.candidate,
            candidate_summary=candidate_summary,
        )
        corpus_update = improve_corpus(
            eval_report={
                "engine": args.candidate,
                "summary": candidate_summary,
                "results": candidate_results,
                "comparison": comparison,
            },
            root_path=str(Path(__file__).resolve().parents[2]),
        )
        report = build_report(
            engine_name=args.candidate,
            results=candidate_results,
            summary=candidate_summary,
            comparison=comparison,
            corpus_update=corpus_update,
        )
        write_report(report, args.output)
        print(format_summary(candidate_summary))
        print()
        print(format_comparison(comparison))
        raise SystemExit(
            0 if evaluate_threshold(candidate_summary, args.threshold) and evaluate_regression(comparison) else 1
        )

    detailed_results, summary = run_evaluations(
        test_case_path=args.test_cases,
        engine_name=args.engine,
    )
    corpus_update = improve_corpus(
        eval_report={
            "engine": args.engine,
            "summary": summary,
            "results": detailed_results,
        },
        root_path=str(Path(__file__).resolve().parents[2]),
    )
    report = build_report(
        engine_name=args.engine,
        results=detailed_results,
        summary=summary,
        corpus_update=corpus_update,
    )
    write_report(report, args.output)
    print(format_summary(summary))
    if corpus_update.get("docs_created"):
        print("\nCorpus Update:")
        for item in corpus_update["docs_created"]:
            print(f"- {item['status']}: {item['path']}")
    print("\nPer-Case Results:")
    for result in detailed_results:
        print(
            "- "
            f"task={result['task']!r}, "
            f"selected_agent={result['selected_agent']!r}, "
            f"retrieval_score={result['retrieval_score']:.2f}, "
            f"routing_score={result['routing_score']:.2f}, "
            f"trajectory_score={result['trajectory_score']:.2f}, "
            f"failure_mode={result['failure_mode']!r}, "
            f"overall_score={result['overall_score']:.2f}, "
            f"latency_ms={result['latency_ms']:.3f}"
        )
    raise SystemExit(0 if evaluate_threshold(summary, args.threshold) else 1)
