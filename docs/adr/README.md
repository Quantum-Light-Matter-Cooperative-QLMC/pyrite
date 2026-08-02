# Architecture Decision Records

Numbered, append-only log of architectural decisions (MADR-lite; see
<https://adr.github.io/madr/>). One decision per file, `NNNN-kebab-title.md`. A
record is immutable once **Accepted**: to change a decision, add a new ADR that
supersedes it and update the old record's Status.

An ADR is short — the *decision* and its consequences. Long-form rationale stays
in the matching `docs/*-rfc.md`; the ADR links it.

Status values: `Proposed` · `Accepted` · `Superseded by ADR-NNNN` · `Deprecated`.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-adopt-architecture-decision-records.md) | Adopt architecture decision records | Accepted |
| [0002](0002-cli-surface-redesign.md) | CLI surface redesign (noun→verb) | Accepted |
| [0003](0003-content-addressed-artifact-model.md) | Content-addressed artifact model | Accepted |
| [0004](0004-package-and-repository-structure.md) | Package & repository structure | Accepted (P4/P5 landed) |
| [0005](0005-energy-grid-schema-decisions.md) | Energy-grid schema decisions | Accepted |
