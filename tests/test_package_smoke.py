import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx import DefaultContextEngine


def test_execute_string_query_from_library_mode(tmp_path: Path) -> None:
    os.environ["CTX_EXECUTION_MODE"] = "mock"
    engine = DefaultContextEngine(
        knowledge_path="case-study/context-docs",
        enable_memory=False,
        storage_path=tmp_path,
    )

    response = engine.execute("debug login issue")

    assert response.output
    assert response.trace_id
    assert response.agent_used
