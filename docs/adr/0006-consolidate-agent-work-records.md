# 0006 — Consolidate agent work records

- **Status:** Accepted — 2026-08-09
- **Date:** 2026-08-09
- **Supersedes:** ADR-0004 P4/P5 document-location decisions

## Context

Agent plans and handoffs were split across tracked `tasks/`, tracked
`docs/plans/`, tracked `.remember/`, `docs/temp/`, and ignored `agentdocs/`.
Excluding some paths from Sphinx did not make ownership clear, and agents kept
placing operational records in the durable documentation tree.

## Decision

- `agentdocs/` is the sole tracked home for agent-operational records.
- `agentdocs/tasks/<full-branch-name>/` owns branch task records.
- `agentdocs/plans/` is reserved for cross-task sequencing that cannot belong
  to one branch. `agentdocs/archive/` may retain superseded handoffs.
- `docs/` owns durable project design, reference, research, and validation
  material only. Promote durable results there; do not author task plans there.
- Disposable notes use ignored `scratch/` or `/tmp`.

## Consequences

The former root `tasks/`, `docs/plans/`, `.remember/`, and `docs/temp/` task
surfaces are retired. `TODO.md`, agent workflow skills, code references, and
Serena project memory point to `agentdocs/`. The tree is tracked but remains
outside the public Sphinx documentation site by construction.
