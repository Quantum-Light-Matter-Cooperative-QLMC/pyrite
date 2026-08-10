# 0005 — Energy-grid schema decisions

- **Status:** Accepted
- **Date:** 2026-07 (recorded 2026-08-01)
- **Historical rationale:** retired `docs/cli-energy-grid-sweep-rework-plan.md` (recoverable from Git history)

## Context

The energy-grid / line-grid rework made several schema decisions (schema
inversion, per-material `E_grid_brem`, line-grid bounds ownership) that shipped
code and tests cite by number ("decision 2/3").

## Decision

The original decisions were recorded in the now-retired
`docs/cli-energy-grid-sweep-rework-plan.md`. That
document is a live decision source — cited across `materials/catalog.py`,
`energy_grid/apply.py`, `cli/json.py`, and tests — so it stays in `docs/` root as
reference rather than moving with ephemeral agent plans (now `agentdocs/`; see
ADR-0006).

## Consequences

- A greppable home for the decisions the code references.
- The plan doc is reference material, not an ephemeral plan.
