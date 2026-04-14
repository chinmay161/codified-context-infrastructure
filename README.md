# Codified Context

Codified Context is packaged as a reusable Python library plus an optional FastAPI service. It now also includes a deterministic Smart Factory that scaffolds the folders and files a Context Engine project needs before you start compiling prompts.

Version `0.3.0` upgrades the compiler from `retrieval -> prompt` to `retrieval -> distillation -> prompt`, adding:

- deterministic context distillation
- stronger agent persona injection
- query-aware prompt templates
- prompt budgeting
- compile-time confidence metadata
- deterministic project scaffolding via `ctx-factory`

## Structure

```text
codified-context/
|-- src/
|   `-- ctx/
|       |-- api/
|       |   |-- __init__.py
|       |   `-- server.py
|       |-- core/
|       |-- corpus/
|       |-- evals/
|       |-- config/
|       |-- cli.py
|       `-- __init__.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Install

```bash
pip install -e .
```

Console entrypoints:

- `ctx` for compiling prompts and recording retrieval feedback
- `ctx-factory` for scaffolding a Context Engine project

## Quickstart

Create a new project scaffold:

```bash
ctx-factory init my_project --domain "Flask authentication system"
```

This generates:

```text
my_project/
|-- ai-context/
|   |-- core_rules.md
|   |-- patterns.md
|   `-- examples.md
|-- .ctx/
|   |-- memory/
|   |-- logs/
|   `-- cache/
|-- schemas/
|   |-- code_generation.json
|   `-- debugging.json
|-- config/
|   `-- factory_config.yaml
|-- src/
|   `-- factory/
|       |-- smart_factory.py
|       |-- generators.py
|       `-- templates.py
|-- .gitignore
|-- cli.py
|-- run_context_engine.py
`-- README.md
```

Then run the helper inside the generated project:

```bash
cd my_project
python run_context_engine.py
```

## What You Need To Run It

Minimum required inputs:

- Python with the package installed from this repo
- a knowledge source containing markdown files

Optional but commonly useful inputs:

- a routing config JSON file if you want project-specific agent routing
- a writable storage directory for logs and retrieval memory
- a `.env` file if you want configuration loaded automatically

Minimum project shape if you are wiring a project manually:

```text
project_x/
|-- knowledge/
|   |-- auth.md
|   |-- db.md
|   `-- architecture.md
`-- routing.json   # optional
```

Example required and commonly used files:

`project_x/knowledge/auth.md`

```md
# Authentication Flow

## Purpose
Explain login, session validation, and retry handling.

## Rules
- Always validate the session token before loading user state.
- Never create a new session until credentials are verified.

## API Notes
- `POST /login` creates a session after successful authentication.
- `POST /refresh-session` renews an existing session.

## Relationships
- The auth handler depends on the session store.
- The retry worker reads from the auth event queue.
```

`project_x/knowledge/db.md`

```md
# Database Notes

## Purpose
Describe how authentication state is stored.

## Rules
- Always write audit records for failed login attempts.
- Do not delete active sessions during a refresh operation.

## Schema Hints
- `users(id, email, password_hash)`
- `sessions(id, user_id, expires_at, refresh_token)`
```

`project_x/knowledge/architecture.md`

```md
# System Architecture

## Overview
The API layer calls the auth service, which reads from the database and writes to the session store.

## Relationships
- API gateway forwards authentication requests to the auth service.
- Auth service depends on the user repository and session store.
- Session refresh flows into the audit logger.
```

Ready-to-use routing example:

`project_x/routing.json`

```json
{
  "knowledge_paths": ["project_x/knowledge"],
  "agents": [
    {
      "name": "ContextArchitect",
      "description": "Handles architecture, planning, and context-engineering tasks.",
      "domain": ["architecture", "planning", "design", "context"],
      "keywords": [
        "architecture",
        "system",
        "design",
        "flow",
        "dependency",
        "context",
        "agent",
        "planning"
      ],
      "spec_path": "agents/contextarchitect.md"
    },
    {
      "name": "GameplayKnowledgeAgent",
      "description": "Handles gameplay, combat, sync, save, and UI system questions.",
      "domain": ["gameplay", "combat", "sync", "save", "ui"],
      "keywords": [
        "combat",
        "damage",
        "enemy",
        "drop",
        "loot",
        "save",
        "sync",
        "network",
        "ui",
        "gameplay"
      ],
      "spec_path": "agents/gameplayknowledgeagent.md"
    },
    {
      "name": "GeneralistAgent",
      "description": "Fallback for broad repository and implementation questions.",
      "domain": ["general", "repository", "workflow"],
      "keywords": [],
      "spec_path": "agents/generalistagent.md"
    }
  ]
}
```

Notes:

- The final agent with an empty `keywords` list acts as the default fallback route.
- The `name` values should match the routed agents supported by the engine: `ContextArchitect`, `GameplayKnowledgeAgent`, and `GeneralistAgent`.
- `spec_path` is metadata for the route entry; the built-in structured persona is loaded from the package agent registry.

Optional environment example:

`.env`

```env
CTX_KNOWLEDGE_PATH=project_x/knowledge
CTX_ROUTING_CONFIG=project_x/routing.json
CTX_STORAGE_PATH=.ctx/project_x
CTX_ENABLE_MEMORY=true
```

Pre-existing files the engine can use:

- Markdown knowledge files under your `knowledge_path`
- Routing config JSON passed as `routing_config` or `CTX_ROUTING_CONFIG`
- Optional `.env` file in the repo or passed via `CTX_DOTENV_PATH`
- Optional writable storage path such as `.ctx/project_x` for:
  - retrieval memory
  - logs

If you do not provide a custom knowledge path, the package falls back to bundled sample knowledge in `src/ctx/corpus/knowledge_base/`.

## Environment Variables

Supported runtime configuration:

- `CTX_KNOWLEDGE_PATH`
- `CTX_AGENTS_CONFIG`
- `CTX_ROUTING_CONFIG`
- `CTX_ENABLE_MEMORY`
- `CTX_STORAGE_PATH`
- `CTX_MEMORY_PATH`
- `CTX_LOG_DIR`
- `CTX_DOTENV_PATH`

Optional LLM provider variables:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_MAX_TOKENS`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `GEMINI_MODEL`
- `CTX_LLM_PROVIDER_ORDER`

Current default behavior:

- If no LLM credentials are configured, the engine still runs and returns the compiled prompt as output.
- Retrieval, distillation, budgeting, and prompt compilation are deterministic and do not require network access.

## Smart Factory

Use the Smart Factory when you want a deterministic starting point for a new Context Engine workspace.

CLI:

```bash
ctx-factory init my_project --domain "Flask authentication system"
```

Python:

```python
from ctx.factory import SmartFactory

factory = SmartFactory("my_project", "Flask authentication system")
factory.run()
```

Generated defaults include:

- `ai-context/` with minimal high-signal docs
- `.ctx/` storage folders
- `schemas/` JSON templates
- `config/factory_config.yaml` with retrieval, prompt, schema, and `domain` defaults
- `.gitignore` for local state and Python artifacts
- `run_context_engine.py` as a minimal compile helper

Safety behavior:

- The factory refuses to overwrite an existing target directory

Example generated helper:

```python
from ctx.core.engine import DefaultContextEngine

engine = DefaultContextEngine(
    knowledge_path="ai-context",
    storage_path=".ctx"
)

result = engine.compile("your query here")
print(result.prompt)
```

## Library Mode

```python
from ctx import DefaultContextEngine, Task

engine = DefaultContextEngine(
    knowledge_path="project_x/knowledge",
    routing_config="project_x/routing.json",  # optional
    enable_memory=True,
    storage_path=".ctx/project_x",
)

compiled = engine.compile(
    Task(
        query="debug login issue",
        files_changed=["backend/auth.py"],
        context_tags=["login", "sessions"],
    )
)

print(compiled.prompt)
print(compiled.agent_used)
print(compiled.metadata["task_mode"])
print(compiled.metadata["compile_confidence"])
print(compiled.metadata["distillation"])
print(compiled.metadata["budget"])
```

If you want the legacy response shape:

```python
from ctx import DefaultContextEngine, Task

engine = DefaultContextEngine(
    knowledge_path="project_x/knowledge",
    routing_config="project_x/routing.json",
)

response = engine.execute(
    Task(
        query="analyze authentication flow",
        files_changed=["backend/auth.py"],
        context_tags=["login", "sessions"],
    )
)

print(response.output)  # compiled prompt
print(response.context_used["files"])
print(response.context_used["metadata"]["query_analysis"])
print(response.context_used["metadata"]["task_mode"])
```

Important metadata keys exposed in v0.3:

- `query_analysis`
- `task_mode`
- `distillation`
- `budget`
- `compile_confidence`

## API Mode

`src/ctx/api/server.py` exposes the engine through FastAPI.

```bash
uvicorn ctx.api.server:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"debug authentication retry failure\",\"context_tags\":[\"auth\"]}"
```

Example feedback request:

```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"debug authentication retry failure\",\"selected_docs\":[\"project_x/knowledge/auth.md\"],\"retrieval_score\":0.92,\"observed_keywords\":[\"auth\",\"retry\"]}"
```

Note: the API currently keeps a compact response model. The full v0.3 metadata is available from the library surface today.

## CLI

```bash
ctx run "debug authentication retry failure" --knowledge-path project_x/knowledge
```

Project scaffolding:

```bash
ctx-factory init my_project --domain "Flask authentication system"
```

JSON output is available with:

```bash
ctx run "debug authentication retry failure" --knowledge-path project_x/knowledge --json
```

To record retrieval feedback for future memory/boosting:

```bash
ctx feedback "debug authentication retry failure" \
  --selected-doc project_x/knowledge/auth.md \
  --retrieval-score 0.92 \
  --observed-keyword auth \
  --observed-keyword retry
```

Current CLI behavior:

- `ctx run ...` prints the compiled prompt
- `ctx run ... --json` returns a compact wrapper with prompt output, trace id, agent, latency, and files
- full v0.3 metadata is available through the Python package interface

## Per-Project Plug-In

Each project can bring its own knowledge base and routing file:

```text
project_x/
|-- knowledge/
|   |-- auth.md
|   |-- db.md
|   `-- architecture.md
`-- routing.json
```

Point the engine at that project:

```python
engine = DefaultContextEngine(
    knowledge_path="project_x/knowledge",
    routing_config="project_x/routing.json",
    storage_path=".ctx/project_x",
)
```

If you do not want to assemble the project by hand, use `ctx-factory init ...` to generate the baseline structure first.

## Why This Enables Reuse

- The package uses `src/` layout, so imports are clean and installation-ready.
- `DefaultContextEngine` no longer hardcodes repo-local knowledge or shared retrieval memory.
- Each engine instance can isolate knowledge, routing, logs, and retrieval memory for a single project.
- The same core engine powers direct imports, CLI execution, and the FastAPI service.
- Adaptive retrieval feedback can be recorded through the library, CLI, or API so memory and boosting improve over time.
- Distillation and budgeting metadata are attached without breaking the prompt-oriented interface.

## Notes

- If no Groq, OpenRouter, or Gemini credentials are configured, the engine still compiles context and returns the generated prompt as output.
- Bundled sample knowledge in `src/ctx/corpus/knowledge_base/` remains available as a default fallback when no project knowledge path is supplied.
