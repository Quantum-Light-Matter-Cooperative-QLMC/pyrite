# Documentation map

Guides, references, and validation records. They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../docs/repo_map.md`](repo_map.md) — canonical package ownership and dependency map.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.

| Document | Topic | Status |
|---|---|---|
| [cli-reference.md](cli-reference.md) | Generated reference for every current `cxr` command | authoritative |
| [api.md](api.md) | Generated Python API reference | authoritative |
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `cxr scan` under SLURM (`sbatch` + job-array templates) | guide |
| [performance-profile-analysis.md](performance-profile-analysis.md) | Analyze performance-profile NDJSON, classify bottlenecks, and design controlled tuning runs | guide |
| [sweep-profiles.md](sweep-profiles.md) | Named full/survey fidelity policies, resolved provenance, and variant checkpoint identity | guide |
| [physics-validation-ledger.md](physics-validation-ledger.md) | Physics claim status and evidence | living ledger |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic broadening and exact orientation averaging | implemented |
| [detector-solid-angle.md](detector-solid-angle.md) | Default single-direction treatment and opt-in face integral | opt-in integral implemented |
| [external-bremsstrahlung-validation.md](external-bremsstrahlung-validation.md) | Versioned external-background fixtures, comparison, fitting, and subtraction | implemented |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: absorption, radiation, transport | implemented |
| [atomic-data-sources.md](atomic-data-sources.md) | Atomic scattering data supplied by xraydb | adopted |
| [crystal-db-comparison.md](crystal-db-comparison.md) | Offline external-database lattice cross-check | implemented |
| [debye-waller-audit.md](debye-waller-audit.md) | Thermal-displacement provenance and scalar/tensor model scope | in progress |
