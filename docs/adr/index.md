# Architecture Decision Records

Architecture Decision Records (ADRs) are the numbered, append-only log of architectural decisions for PyRITE.

The format is MADR-lite. One decision lives in each file, named:

```text
NNNN-kebab-title.md
```

A record is immutable once **Accepted**. To change an accepted decision, add a new ADR that supersedes it and update the old record's status.

Allowed status values are:

* `Proposed`
* `Accepted`
* `Superseded by ADR-NNNN`
* `Deprecated`

## ADRs and RFCs

An ADR is intentionally short: it records the **decision**, the relevant context, and its consequences.

Long-form rationale, alternatives, interface sketches, and implementation analysis may live in repository-design documentation while a decision is being developed. Accepted planning documents may be retired once the ADR is concise and self-contained; Git history preserves their point-in-time detail.

In other words:

* use an **RFC/design document** to explore what should be done when extended analysis remains useful;
* use an **ADR** to record what was ultimately decided.

## Decision records

```{toctree}
:maxdepth: 1
:hidden:

0001-adopt-architecture-decision-records
0002-cli-surface-redesign
0003-content-addressed-artifact-model
0004-package-and-repository-structure
0005-energy-grid-schema-decisions
0006-consolidate-agent-work-records
0007-project-identity
0008-no-arbitrary-target-geometry
0009-result-persistence-format
0010-reduce-cli-noun-surface
0012-backlog-in-github-issues
0013-automatic-line-grids-by-default
0014-packaged-data-layout
0015-xraydb-atomic-scattering-data
0016-photon-transport-source-and-scoring
```

| ADR                                                 | Title                               | Status                                 |
| --------------------------------------------------- | ----------------------------------- | -------------------------------------- |
| [0001](0001-adopt-architecture-decision-records.md) | Adopt architecture decision records | Accepted                               |
| [0002](0002-cli-surface-redesign.md)                | CLI surface redesign (noun→verb)    | Superseded by ADR-0010                 |
| [0003](0003-content-addressed-artifact-model.md)    | Content-addressed artifact model    | Accepted                               |
| [0004](0004-package-and-repository-structure.md)    | Package & repository structure      | Accepted; P4/P5 superseded by ADR-0006; backlog clause superseded by ADR-0012 |
| [0005](0005-energy-grid-schema-decisions.md)        | Energy-grid schema decisions        | Accepted                               |
| [0006](0006-consolidate-agent-work-records.md)      | Consolidate agent work records      | Accepted                               |
| [0007](0007-project-identity.md)                    | PyRITE project identity             | Accepted                               |
| [0008](0008-no-arbitrary-target-geometry.md)        | No arbitrary target geometry        | Accepted                               |
| [0009](0009-result-persistence-format.md)           | Result persistence format           | Accepted                               |
| [0010](0010-reduce-cli-noun-surface.md)             | Reduce the CLI noun surface          | Accepted                               |
| [0012](0012-backlog-in-github-issues.md)            | Backlog in GitHub Issues, not TODO.md | Accepted                               |
| [0013](0013-automatic-line-grids-by-default.md)     | Automatic line grids by default      | Accepted                               |
| [0014](0014-packaged-data-layout.md)                | Packaged data layout                 | Accepted                               |
| [0015](0015-xraydb-atomic-scattering-data.md)       | xraydb as the atomic scattering data source | Accepted                         |
| [0016](0016-photon-transport-source-and-scoring.md) | Photon transport source and scoring | Accepted |
