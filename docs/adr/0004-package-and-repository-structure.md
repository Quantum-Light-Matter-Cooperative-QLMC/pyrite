# 0004 — Package & repository structure

- **Status:** Accepted — 2026-08-01 (P4/P5 superseded by ADR-0006)
- **Date:** 2026-08-01
- **Rationale:** [`docs/package-structure-rfc.md`](../package-structure-rfc.md)

## Context

Command implementations live in two homes; one concept (energy-grid) has three
module names; recompute/prune modules overlap; docs mixed durable reference with
ephemeral plans; multiple TODO surfaces bypassed the `TODO.md merge=ours` driver.

## Decision

- **P4 (accepted, landed 2026-08-01):** split docs by lifetime — `docs/plans/`
  for tracked-ephemeral plans/handoffs, `docs/adr/` for this log; durable
  reference stays in `docs/` root.
- **P5 (accepted, landed 2026-08-01):** fold `TODO_CLI.md` / `TODO_UI.md` into
  `TODO.md` `## CLI backlog` / `## UI backlog`, so the single
  merge-driver-protected file is the only backlog on `main`.
- **P1–P3 (accepted, landed 2026-08-09):** command wiring is under
  `src/cxr_mc/cli/commands/`, the energy-grid package matches its surface term,
  and recompute/cleanup implementations are grouped under their domain owners.

## Consequences

Document-location consequences below are historical and superseded by
[ADR-0006](0006-consolidate-agent-work-records.md), which consolidates tracked
agent work under `agentdocs/`.

- Untracked agent scratch renamed `claudedocs/` → `agentdocs/` (agent-neutral;
  the tree already standardizes on `AGENTS.md`).
- `docs/plans/` and `docs/adr/` are excluded from the Sphinx site (dev-facing).
- P1–P3 landed with compatibility re-exports for former module paths.
