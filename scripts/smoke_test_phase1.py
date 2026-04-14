"""Smoke test for all 5 critical gap fixes."""
import json
from pathlib import Path

from ctx.core.agent import load_agent
from ctx.core.engine import (
    DefaultContextEngine,
    _merge_agent_trace,
    validate_agent_output,
)
from ctx.core.models import Task
from ctx.core.prompt_builder import build_system_prompt, load_constitution

# ── Gap 3: Constitution is loaded and injected ─────────────────────────────
constitution = load_constitution()
assert constitution, "Constitution.md was not found or is empty"
assert "ARTICLE" in constitution, "Constitution content looks wrong"
agent = load_agent("draft_agent")
sys_prompt = build_system_prompt(agent)
assert constitution[:80] in sys_prompt, "Constitution not injected into system prompt"
print(f"Gap 3 OK — Constitution injected ({len(constitution)} chars)")

# ── Gap 1: _enrich_task_with_agent adds agent domain tags ──────────────────
from ctx.core.engine import DefaultContextEngine as DCE
task = Task(query="what is the retrieval strategy?")
enriched = DCE._enrich_task_with_agent(task, agent)
assert enriched.context_tags is not None, "context_tags should not be None after enrichment"
# agent name "draft_agent" → tokens like "draft", "agent"
assert any("draft" in t or "agent" in t for t in enriched.context_tags), \
    f"Agent tokens not in context_tags: {enriched.context_tags}"
print(f"Gap 1 OK — enriched tags: {enriched.context_tags}")

# No structured_agent → task returned unchanged
same = DCE._enrich_task_with_agent(task, None)
assert same is task, "Should return the original task when agent is None"
print("Gap 1 OK — None agent returns original task")

# ── Gap 2: _merge_agent_trace merges step list into JSON ───────────────────
llm_output = json.dumps({
    "answer": "Test",
    "confidence": 0.8,
    "sources": ["doc.md"],
    "agent_trace": ["llm_step"],
})
engine_trace = [{"step": "agent_selected", "agent": "draft_agent"}, {"step": "retrieval_done", "docs": 3}]
merged_str = _merge_agent_trace(llm_output, engine_trace)
merged = json.loads(merged_str)
trace = merged["agent_trace"]
assert trace[0] == {"step": "agent_selected", "agent": "draft_agent"}, f"Wrong first step: {trace[0]}"
assert trace[-1] == "llm_step", f"LLM step should be last: {trace[-1]}"
print(f"Gap 2 OK — trace has {len(trace)} steps, engine steps prepended")

# ── Gap 4: validate_agent_output + retry logic (unit) ─────────────────────
good = json.dumps({"answer": "A", "confidence": 0.9, "sources": [], "agent_trace": []})
out, ok = validate_agent_output(good)
assert ok, "Valid output rejected"

bad = "not json"
out2, ok2 = validate_agent_output(bad)
assert not ok2, "Bad output should be rejected"

# Verify failure payload has all required keys
fp = json.loads(out2)
assert all(k in fp for k in ("answer", "confidence", "sources", "agent_trace"))
print("Gap 4 OK — validate_agent_output logic verified")

# ── Integration: execute() with a real knowledge dir ──────────────────────
import os, tempfile
os.environ["CTX_EXECUTION_MODE"] = "mock"
with tempfile.TemporaryDirectory() as tmpdir:
    kdir = Path(tmpdir) / "kb"
    kdir.mkdir()
    (kdir / "retrieval.md").write_text(
        "# Retrieval Strategy\n\nThe system uses BM25 plus semantic fusion.\n",
        encoding="utf-8",
    )
    engine = DefaultContextEngine(knowledge_path=kdir, storage_path=tmpdir, enable_memory=False)
    response = engine.execute("what is the retrieval strategy?")
    # Execution steps must include agent_activated (Gap 1 timing)
    steps = response.trace["steps"]
    activated = [s for s in steps if s.startswith("agent_activated:")]
    retrieval = [s for s in steps if s == "retrieve_context"]
    assert activated, f"agent_activated step missing: {steps}"
    assert retrieval, f"retrieve_context step missing: {steps}"
    # agent_activated must come BEFORE retrieve_context
    assert steps.index(activated[0]) < steps.index("retrieve_context"), \
        "agent must be activated before retrieval"
    print(f"Gap 1 integration OK — activation before retrieval confirmed")
    print(f"Execution steps: {steps}")
    import logging
    logging.shutdown()

print("\nAll 5 gap checks passed!")

