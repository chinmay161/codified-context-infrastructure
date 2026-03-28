# Codified Context: Infrastructure for AI Agents in a Complex Codebase

Companion repository to: *"Codified Context: Infrastructure for AI Agents in a Complex Codebase"* by Aris Vasilopoulos ([arXiv:2602.20478](https://arxiv.org/abs/2602.20478)).

This trimmed distribution focuses on the paper, representative context documents, and the analysis pipeline. MCP server code and Claude-specific orchestration artifacts have been removed.

## About

This project is based on:
Codified Context Infrastructure by Aristidis Vasilopoulos

## Changes

- Replaced Claude/MCP execution layer
- Adapting for GitHub Copilot
- Planned extensions:
  - Code Knowledge Graph
  - Lifecycle Management Layer

## What Remains

- `case-study/context-docs/` contains representative system documents from the paper's case study.
- `data/` contains the extraction and analysis scripts, methodology notes, sample aggregates, and annotated case-study excerpts used in the paper.
- `paper/` contains the abstract, citation, and links for the publication.

## Repository Structure

```text
case-study/
  context-docs/               Representative knowledge-base documents

data/
  case-study-excerpts/        Annotated excerpts referenced in the paper
  samples/                    Sample aggregate datasets
  *.py                        Analysis and extraction scripts
  *.md                        Methodology documentation

paper/
  README.md                   Abstract, citation, and links
```

## Using This Repository

Use the files here as reference material for:

- Studying the documentation patterns described in the paper
- Reusing the structure of the context documents in your own tooling
- Reproducing or extending the analysis pipeline with your own conversation data

The remaining materials are intentionally tool-agnostic at the repository level, even where the underlying research data comes from a specific assistant workflow.

## Paper-to-Repo Mapping

| Paper Section | Repo Directory |
|---------------|----------------|
| Representative context documents | `case-study/context-docs/` |
| Evaluation metrics and methodology | `data/` |
| Case-study excerpts | `data/case-study-excerpts/` |
| Paper reference and citation | `paper/` |

## Links

- **Paper:** [arXiv:2602.20478](https://arxiv.org/abs/2602.20478)
- **Author:** [Aris Vasilopoulos](https://github.com/arisvas4)

## License

MIT
