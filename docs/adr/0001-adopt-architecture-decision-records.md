# 0001 — Adopt architecture decision records

- **Status:** Accepted
- **Date:** 2026-08-01
- **Deciders:** Alex Amador

## Context

"Why did we decide X" was scattered across `docs/*-rfc.md`, one-off plan files,
and commit messages, with no durable, greppable decision log. See
`docs/adr/0004-package-and-repository-structure.md` §2.5.

## Decision

Keep a numbered ADR log under `docs/adr/` (MADR-lite). When an RFC or design is
accepted, add a short ADR that records the decision and links the rationale. RFCs
remain the long-form rationale; ADRs are the index of what was decided.

## Consequences

- One place to grep for decisions; RFCs no longer double as the decision record.
- Small per-decision overhead (one stub file) at accept time.
- Backfilled the CLI/structure RFCs as ADRs (0002–0004), accepted 2026-08-01.
