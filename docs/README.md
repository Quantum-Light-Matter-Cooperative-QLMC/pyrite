# Design notes & decision records

In-depth "why / how / trade-offs" write-ups that would bloat the top-level docs.
They complement:

- [`../README.md`](../README.md) — user-facing overview, physics, install, validation.
- [`../docs/repo_map.md`](repo_map.md) — canonical package ownership and dependency map.
- [`../TODO.md`](../TODO.md) — the feature / patch backlog.

| Note | Topic | Status |
|---|---|---|
| [running-on-a-cluster.md](running-on-a-cluster.md) | Headless `cxr scan` under SLURM (`sbatch` + job-array templates) | guide |
| [crystal-mosaicity.md](crystal-mosaicity.md) | Analytic mosaic line-broadening vs. exact Monte-Carlo orientation averaging | analytic ✅ · MC ✅ |
| [detector-solid-angle.md](detector-solid-angle.md) | Single-direction approximation (shipped) vs. a first-principles solid-angle integral | ⏳ |
| [multilayer-materials.md](multilayer-materials.md) | Film-on-substrate stacks: cross-stack self-absorption, per-layer radiation, multilayer transport | slices 1–3 ✅ |
| [atomic-data-sources.md](atomic-data-sources.md) | Hard-coded Henke/Cromer–Mann tables vs. an external library (xraydb) | adopted ✅ |

Legend: ✅ implemented · ⏳ designed, not implemented.
