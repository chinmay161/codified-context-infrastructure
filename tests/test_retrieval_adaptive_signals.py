from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx.core.models import Agent, Task
from ctx.core.retrieval_memory import RetrievalMemory
from ctx.core.retrieval_v3 import HybridRetriever
from ctx.utils.file_loader import MarkdownDocument


def test_retrieval_memory_matches_similar_queries(tmp_path: Path) -> None:
    memory = RetrievalMemory(memory_path=tmp_path / "memory.json")
    memory.update(
        query="debug combat sync failure",
        selected_docs=["/docs/host-authoritative-damage-spec.md"],
        retrieval_score=0.9,
        observed_keywords=["combat", "sync", "damage"],
    )

    entry = memory.get("debug sync combat issue")

    assert entry["memory_match_type"] == "approximate"
    assert entry["memory_confidence"] > 0.0
    assert "/docs/host-authoritative-damage-spec.md" in entry["successful_docs"]
    assert "damage" in entry["successful_keywords"]


def test_hybrid_retriever_exposes_expansion_and_memory_boost(tmp_path: Path) -> None:
    knowledge_root = tmp_path / "knowledge_base"
    knowledge_root.mkdir()
    generated_path = knowledge_root / "sync-debug-guide.md"
    generated_path.write_text(
        "# Sync Debug Guide\n\n"
        "Generated for task 'optimize retrieval boost for sync failures'.\n"
        "Investigate client sync failures, debug network state, and improve retrieval relevance.\n",
        encoding="utf-8",
    )
    static_path = tmp_path / "general-overview.md"
    static_path.write_text(
        "# General Overview\n\n"
        "Repository conventions and broad workflow notes.\n",
        encoding="utf-8",
    )

    docs = [
        MarkdownDocument(path=str(generated_path), content=generated_path.read_text(encoding="utf-8")),
        MarkdownDocument(path=str(static_path), content=static_path.read_text(encoding="utf-8")),
    ]
    memory = RetrievalMemory(memory_path=tmp_path / "retrieval_memory.json")
    memory.update(
        query="debug sync failures",
        selected_docs=[str(generated_path)],
        retrieval_score=0.95,
        observed_keywords=["sync", "debug", "network"],
    )
    retriever = HybridRetriever(
        documents=docs,
        knowledge_base_root=knowledge_root,
        memory=memory,
    )
    agent = Agent(
        name="retrieval-agent",
        description="Retrieval optimizer",
        domain=["retrieval", "ranking", "debug"],
        spec_path="agents/retrieval.md",
    )

    results, metadata = retriever.retrieve(
        query=Task(query="optimize sync failure ranking"),
        agent={
            "name": "retrieval-agent",
            "role": "Retrieval optimizer",
            "domain": ["retrieval", "ranking", "debug"],
            "workflow": ["expand_query"]
        },
        top_k=2,
    )

    assert results[0].path == str(generated_path)
    assert metadata["memory_hit"] is True
    assert metadata["memory_match_type"] == "approximate"
    assert metadata["boost_signal"] > 0.0
    assert metadata["expanded_query_terms"]
    assert "improve" in metadata["expanded_query_terms"] or "debug" in metadata["expanded_query_terms"]
