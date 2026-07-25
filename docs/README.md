# Documentation map

Guides, references, validation records, and historical design notes. They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../docs/repo_map.md`](repo_map.md) — canonical package ownership and dependency map.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.

| Document | Topic | Status |
|---|---|---|
| [cli-reference.md](cli-reference.md) | Generated reference for every current `cxr` command | authoritative |
| [api.md](api.md) | Generated Python API reference | authoritative |
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `cxr scan` under SLURM (`sbatch` + job-array templates) | guide |
| [physics-validation-ledger.md](physics-validation-ledger.md) | Physics claim status and evidence | living ledger |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic broadening and exact orientation averaging | implemented |
| [detector-solid-angle.md](detector-solid-angle.md) | Default single-direction treatment and opt-in face integral | opt-in integral implemented |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: absorption, radiation, transport | implemented |
| [atomic-data-sources.md](atomic-data-sources.md) | Atomic scattering data supplied by xraydb | adopted |
| [crystal-db-comparison.md](crystal-db-comparison.md) | Offline external-database lattice cross-check | implemented |
| [superpowers/README.md](superpowers/README.md) | Dated implementation plans and design specifications | historical |

`superpowers/` records describe decisions at their authoring dates; use the CLI and API
references for current behavior.
