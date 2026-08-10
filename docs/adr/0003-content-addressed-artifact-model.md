# 0003 — Content-addressed artifact model

- **Status:** Accepted
- **Date:** 2026-08-01
- **Historical rationale:** retired `docs/cli-artifact-model-rfc.md` (recoverable from Git history)

## Context

Derived grids/materials are mutable and stored in two places at once;
`energy-grid apply` mutates the `standard` profile; there is no dedup or cheap
staleness check.

## Decision

Adopt DVC's split: derived data is immutable and content-addressed; the profile
is the sole mutable, git-ref-like pointer. Uniform lifecycle verbs (`add`,
`verify`, `gc`) with a git-reflog-style grace window; a completed run emits a
campaign lockfile from the first cut. Land behind the stabilized CLI surface
(RFC §4 phase 5).

## Consequences

- Free dedup and hash-diff staleness / remote-sync checks.
- The hash-input tuple becomes a frozen compatibility contract (freeze test).
- Reproducible, citable campaigns via the lockfile.
