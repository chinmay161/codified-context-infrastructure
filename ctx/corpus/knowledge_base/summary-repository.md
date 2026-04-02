# summary / repository / workflow Context Guide

## Purpose
Explain operational flow, handoffs, and repository-level working conventions. This document closes the retrieval gap for the task 'summarize a generic repository workflow'.

## Core Mechanism
1. Identify the triggering task or user intent that requires this knowledge.
2. Map the concept to the relevant modules, files, and documented system boundaries.
3. Explain the expected flow, key rules, and recovery guidance for future retrieval.

## Rules
- Document the sequence of steps from trigger to completion.
- Do not assume implicit operator knowledge for cross-file workflows.
- Capture entry points, outputs, and coordination rules clearly.

## Patterns
- Missing concepts: summary, repository, workflow
- Known repository patterns: dataclass_models, handler_functions, object_oriented_modules
- Expected domain: workflow

## Failure Modes
| Symptom | Cause | Fix |
| --- | --- | --- |
| Retrieval failure | Missing or weak context coverage | Add or refresh this document with concept-specific examples |
| Routing uncertainty | Domain context is underspecified | Clarify ownership, boundaries, and related modules |
| Outdated implementation guidance | Code and docs drifted apart | Version this document and update related file references |

## Related Files
- ctx/evals/runner.py
- data/analyze_impact.py
- data/extract_prompts.py
