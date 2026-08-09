# Documentation map

Guides, references, and validation records. They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../docs/repo_map.md`](repo_map.md) — canonical package ownership and dependency map.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.
- [`../agentdocs/`](../agentdocs/) — tracked agent task plans and handoffs;
  never published project documentation.

| Document | Topic | Status |
|---|---|---|
| [cli-reference.md](cli-reference.md) | Generated reference for every current `cxr` command | authoritative |
| [cli-deprecations.md](cli-deprecations.md) | Generated compatibility and removal-window registry | authoritative |
| [api.md](api.md) | Generated Python API reference | authoritative |
| [development-workspace.md](development-workspace.md) | Single-project contributor environment and focused verification | guide |
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `cxr run` under SLURM (`sbatch` + job-array templates) | guide |
| [performance-profile-analysis.md](performance-profile-analysis.md) | Analyze performance-profile NDJSON, classify bottlenecks, and design controlled tuning runs | guide |
| [sweep-profiles.md](sweep-profiles.md) | Named full/survey fidelity policies, resolved provenance, and variant checkpoint identity | guide |
| [checkpoint-case-store.md](checkpoint-case-store.md) | Cross-profile per-case CAS, manifests, cache modes, and compatibility | implemented |
| [physics-validation-ledger.md](physics-validation-ledger.md) | Physics claim status and evidence | living ledger |
| [coherent-emission.md](coherent-emission.md) | Optional phased segment/electron sum and validation boundary | experimental, unverified |
| [coherent-streaming-rawkernel.md](coherent-streaming-rawkernel.md) | Streaming coherent GPU reduction design and evidence | implemented |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic broadening and exact orientation averaging | implemented |
| [detector-solid-angle.md](detector-solid-angle.md) | Default single-direction treatment and opt-in face integral | opt-in integral implemented |
| [external-bremsstrahlung-validation.md](external-bremsstrahlung-validation.md) | Versioned external-background fixtures, comparison, fitting, and subtraction | implemented |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: absorption, radiation, transport | implemented |
| [atomic-data-sources.md](atomic-data-sources.md) | Atomic scattering data supplied by xraydb | adopted |
| [crystal-db-comparison.md](crystal-db-comparison.md) | Offline external-database lattice cross-check | implemented |
| [debye-waller-audit.md](debye-waller-audit.md) | Thermal-displacement provenance and scalar/tensor model scope | in progress |

## Decisions

These are dev-facing and excluded from the published site.

- [`adr/`](adr/) — numbered architecture decision records (MADR-lite); the
  durable, greppable "why we decided X" log.
- `*-rfc.md` — long-form design rationale (surface, artifact model, structure);
  each accepted RFC gets an ADR stub.

Agent-operational plans and handoffs belong only in
[`../agentdocs/`](../agentdocs/), not under `docs/`.
