# 0010 — Reduce the CLI noun surface

- **Status:** Accepted
- **Date:** 2026-08-14
- **Supersedes:** [ADR-0002](0002-cli-surface-redesign.md)
- **Context source:** [core architecture RFC, Change 7](../repo-design/core-architecture-rfc.md#change-7--reduce-the-cli-noun-surface)

## Context

ADR-0002 established noun→verb ordering, remote execution as a modifier, one
job lifecycle, controlled vocabulary, stable machine output, and D7 warning
windows. Those decisions remain sound, but the resulting user CLI exposed
thirteen top-level nouns. Several named derived-artifact maintenance rather
than a user workflow, while setup and shell completion were separate nouns
despite both configuring the local installation.

The object model now has public `Beam`, `Target`, `Detector`, `Scene`, `Sweep`,
`Numerics`, and `Analysis` owners plus filesystem-free `simulate`. There is no
longer an API gap that requires maintenance machinery to remain prominent in
the user CLI.

## Decision

Retain ADR-0002's noun→verb, remote-modifier, job, vocabulary, output, and D7
decisions. Amend its command organization to exactly nine visible top-level
nouns:

`run`, `app`, `checkpoint`, `config`, `remote`, `job`, `profile`, `material`,
and `beam`.

- `setup` and `completion` move below `config`.
- Material energy-grid derivation and inspection move below `material`; shared
  derivation defaults move below `profile`.
- Energy-grid artifact mutation/verification and local performance-artifact
  management move to `pyrite-dev`.
- Every former spelling remains a hidden D7 alias for one support window and
  invokes the same callback as its replacement.
- Interactive profile mutation remains. Removing leaf verbs would not reduce
  the noun surface and would conflict with named-beam attachment and approved
  profile lifecycle follow-ups.
- `beam` remains a noun. It is a named physical object and public API primitive.
- `checkpoint` remains a noun because checkpoints are expensive results and
  validation evidence, not disposable cache entries. No `cache` noun is
  created.

## Consequences

- Root help exposes nine concepts without changing command parameters,
  precedence, streams, exits, JSON schemas, destructive previews, confirmation,
  revalidation, or side effects.
- User workflows remain in `pyrite`; source-maintenance and regenerable-artifact
  operations require `pyrite-dev` and therefore a project/developer install.
- Old scripts keep working with one stderr warning through the D7 window. Help
  and completion advertise only canonical paths.
- A future first-class detector lifecycle may justify a tenth noun, but this
  decision neither introduces nor precludes it.
