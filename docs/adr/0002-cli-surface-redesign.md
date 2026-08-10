# 0002 — CLI surface redesign (noun→verb)

- **Status:** Accepted
- **Date:** 2026-08-01
- **Historical rationale:** retired “CLI redesign RFC” (recoverable from Git history)

## Context

The command surface has inconsistent noun/verb ordering, remote handled as a
top-level verb, an ad-hoc job lifecycle, and uncontrolled flag vocabularies.

## Decision

Adopt the redesign RFC: noun→verb ordering, remote-as-modifier, unified job
lifecycle, controlled verb/flag vocabularies, an `-o/--output` contract (`json`
the sole automation-bound format), and a git/kubectl-style deprecation policy.
Sequenced in the RFC §4; phase 0 is the package-structure command-home
consolidation (ADR-0004 P1).

## Consequences

- Breaking command changes gated behind the deprecation policy.
- `docs/repo-design/cli/cli-reference.md` freeze test guards the migration.
