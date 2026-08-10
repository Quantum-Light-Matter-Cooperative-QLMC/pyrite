# 0004 — Package & repository structure

- **Status:** Accepted — 2026-08-01 (document-location details partly superseded by ADR-0006)
- **Date:** 2026-08-01

## Context

Repository structure had drifted outside the well-factored physics packages: command implementations lived in multiple locations, the energy-grid concept had inconsistent module names, recompute and cleanup behavior was spread across overlapping modules, and durable documentation was mixed with temporary planning material. Project decisions and backlog ownership were similarly fragmented.

This made subsequent CLI and repository work harder because there was no predictable ownership rule for command code, project records, or several cross-cutting implementation concerns.

## Decision

Command-line wiring belongs under `src/cxr_mc/cli/`, with reusable behavior remaining under its domain owner. Implementation names should follow canonical project terminology, including consolidation around `energy-grid`, and related recompute/cleanup behavior should be grouped under the packages that own it.

Top-level implementation modules may be grouped into responsibility-oriented packages such as `checkpoints/`, `campaign/`, `runs/`, `apps/`, `validation/`, `perf/`, and `remote/`, while compatibility re-exports preserve supported former paths.

Durable project documentation belongs under `docs/`, and architectural decisions are recorded as numbered ADRs. `TODO.md` is the single backlog authority on `main`. The original location policy for temporary agent work was subsequently superseded by [ADR-0006](0006-consolidate-agent-work-records.md).

## Consequences

The repository has clearer ownership boundaries and a predictable home for command wiring, implementation domains, documentation, decisions, and backlog state. Structural refactors can be made without implying changes to the public CLI, Python API, or numerical/physics behavior.

Some former paths must remain as compatibility re-exports or follow explicit deprecation processes. This decision does not authorize changes to the established physics-package layout or removal of public interfaces.
