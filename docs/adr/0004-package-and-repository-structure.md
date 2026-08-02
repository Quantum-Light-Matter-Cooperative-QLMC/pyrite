# 0004 — Package & repository structure

- **Status:** Accepted — 2026-08-01 (P4/P5 landed; P1–P3 scheduled)
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
- **P1–P3 (accepted, scheduled):** one command home under `src/cxr_mc/cli/`
  (redesign phase 0), rename the energy-grid module to match its surface term,
  and fold the recompute/prune module sprawl with RFC D4's verb collapse.

## Consequences

- Untracked agent scratch renamed `claudedocs/` → `agentdocs/` (agent-neutral;
  the tree already standardizes on `AGENTS.md`).
- `docs/plans/` and `docs/adr/` are excluded from the Sphinx site (dev-facing).
- P1–P3 remain to be scheduled; P1 gates the CLI-surface redesign (ADR-0002).
