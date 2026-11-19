# Repository map

Navigation aid for `src/pyrite/` — importable package. Read this front page before exploring source, then open the [detail page](#detail-pages) for the area you touch. For *why* (physics, validation, provenance) see `README.md` and the design notes under `docs/`; backlog in [GitHub Issues](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues). Cross-task agent plans and archived handoffs live under `agentdocs/`, never under the public documentation tree. Print current top-level directory inventory: `pyrite-dev repo-map`. Regenerate or check the static dependency region with `pyrite-dev repo-map --write|--check`; `pyrite-dev verify` runs the check and the import-linter contracts from `pyproject.toml`.

## Workspace ownership

- Root `pyrite-mc`: sole publishable distribution and owner of `src/pyrite/`, packaged data, `pyrite`, `pyrite-dev`, and the test suite. No `uv` workspace split; `pyrite-dev ...` needs no `--package` flag.
- `pyrite-dev test-suite` partitions tests into transport/spectrum leaves, quantum channeling solvers, channeling integration, energy-grid, materials, xsgen, CLI, apps, packaging, and residual core. Family names are leaves, not aggregates; `integration` and `identity-pins` overlap deliberately. Select affected nodes/modules first; `verify` runs repository checks and validates committed tutorial captures without executing tutorials. See [suite names and focused verification](repo-design/development-workspace.md).
- Runtime paths resolve through `pyrite.paths` and `console.config`: packaged read-only data stays package-relative; a complete selected catalog uses per-call `--catalog` > `PYRITE_CATALOG` > `catalog.path` > bundled; generated outputs use `console.outputs` under `<workspace>/pyrite-output/<kind>/`; workspace precedence is explicit path > `PYRITE_HOME` > `workspace.root` config > nearest ancestor containing `pyrite-output/` > cwd; mutable user state uses Click's platform app directory. An explicit workspace also owns generated `xsgen/` tables and fetched reference data. Marimo apps ship under `pyrite.apps`; the browser-only docs demo `docs/demo/results_app.py` imports no `pyrite` and reads data written by `scripts/export_docs_demo.py`; validation figure builders and their reference data ship under `pyrite.validation`; standalone `checks/` scripts remain developer-only.
- `vendor/xsgen/{elsepa,sbethe}/`: generator inputs (ELSEPA 2020 Fortran and `database/`, `sbethe.f`), not in the wheel. `xsgen.sources` resolves them as the vendored tier from a checkout, and `pyrite remote sync` ships `vendor/`. Their data classes are set by [ADR-0014](adr/0014-packaged-data-layout.md). ELSEPA's published test-run output is a test fixture under `tests/data/xsgen/elsepa/`.
- Root domain implementations are grouped under `campaign/`, `checkpoints/`, `runs/`, `apps/`, `validation/`, `perf/`, and `remote/`. Former flat root module paths were removed in 0.5.0; import the owning subpackage.

## Detail pages

This page is the front page: read it first, then open only the detail page for
the area you touch. Detail pages hold per-module ownership notes (public and
internal symbols, dependencies, tests and design links).

```{toctree}
:maxdepth: 1

Entry points <repo-map/entry-points>
Core physics <repo-map/core-physics>
Campaign, checkpoints, run drivers & optimization <repo-map/campaign-and-runs>
Results, plotting & downstream models <repo-map/results-and-instruments>
CLI & packaging <repo-map/cli-and-packaging>
```

## Where to start

| Task | Open |
|---|---|
| Add or change a CLI command, option or app launcher | [CLI & packaging](repo-map/cli-and-packaging.md), then [Entry points](repo-map/entry-points.md) |
| Trace a user workflow from `pyrite`/`import pyrite` to its implementation | [Entry points](repo-map/entry-points.md) |
| Transport, collision, radiation or materials physics | [Core physics](repo-map/core-physics.md) |
| Profiles, sweeps, case lowering, identity, checkpoints, run drivers, optimization studies | [Campaign, checkpoints, run drivers & optimization](repo-map/campaign-and-runs.md) |
| Result containers, observations, plotting, instrument or detector response | [Results, plotting & downstream models](repo-map/results-and-instruments.md) |
| Remote jobs, devtools, packaging, validation apps | [CLI & packaging](repo-map/cli-and-packaging.md) |
| Import-boundary changes | The dependency DAG below and [import architecture](repo-design/import-architecture.md) |

## Package index

One line per top-level package under `src/pyrite/`. Newer facade packages
(`beams`, `config`, `execution`, `experimental`, `geometry`, `storage`) have no
dedicated section yet; the column names the page covering their closest owner.

| Package | Owns | Nearest detail page |
|---|---|---|
| `_entry` | Thin `python -m` shims invoked on the remote box | [Entry points](repo-map/entry-points.md) |
| `_util` | Domain-free private helpers: atomic writes, file/JSON digests, env, extras, deprecation schedules | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `apps` | Packaged marimo apps and validation resources | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `beams` | Public beam and source-distribution descriptions | [Campaign](repo-map/campaign-and-runs.md) |
| `campaign` | Campaign configuration, profiles, sweep construction, case lowering | [Campaign](repo-map/campaign-and-runs.md) |
| `checkpoints` | Checkpoint storage, lifecycle, recompute and repair | [Campaign](repo-map/campaign-and-runs.md), [CLI](repo-map/cli-and-packaging.md) |
| `cli` | Click command-line entry point and commands | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `config` | Public physics, numerical and sampling configuration | [Campaign](repo-map/campaign-and-runs.md) |
| `console` | Terminal presentation primitives, output selector, dashboard | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `detectors` | Detector models, acceptance, geometry, sensor response | [Results & downstream](repo-map/results-and-instruments.md) |
| `devtools` | Repository-only `pyrite-dev` tooling | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `energy_grid` | Photon-energy-grid derivation, encoding, artifacts | [Campaign](repo-map/campaign-and-runs.md) |
| `execution` | Backend-independent campaign execution primitives | [Campaign](repo-map/campaign-and-runs.md) |
| `experimental` | Discoverable experimental interfaces by owning domain | [Entry points](repo-map/entry-points.md) |
| `geometry` | Public sample geometry, targets and physical bodies | [Core physics](repo-map/core-physics.md) |
| `instrument` | Physical downstream photon geometry and filters | [Results & downstream](repo-map/results-and-instruments.md) |
| `materials` | Material identity, crystal configuration, X-ray material physics | [Core physics](repo-map/core-physics.md) |
| `montecarlo` | Transport, collision, emission and spectrum engines; runner | [Core physics](repo-map/core-physics.md) |
| `observations` | Measurement, acquisition, scoring plans, persisted observations | [Results & downstream](repo-map/results-and-instruments.md) |
| `optimization` | Optional scientific optimization studies and kernel tuning | [Campaign](repo-map/campaign-and-runs.md) |
| `perf` | Performance profiling and analysis | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `plots` | Plotting backends | [Results & downstream](repo-map/results-and-instruments.md) |
| `radiation` | Source-level radiation calculations facade | [Entry points](repo-map/entry-points.md) |
| `remote` | SSH/SLURM remote host scheduling | [CLI & packaging](repo-map/cli-and-packaging.md) |
| `results` | Result containers, reductions and views | [Results & downstream](repo-map/results-and-instruments.md) |
| `runs` | Local run, scan and blazed-sweep drivers | [Campaign](repo-map/campaign-and-runs.md) |
| `storage` | Persistence facade for checkpoints, observations, execution artifacts | [Campaign](repo-map/campaign-and-runs.md) |
| `tools` | Deterministic scientific calculator facade | [Entry points](repo-map/entry-points.md) |
| `transport` | Advanced transport and saved-trajectory analysis facade | [Entry points](repo-map/entry-points.md) |
| `validation` | Validation adapters, oracles and catalog checks | [Core physics](repo-map/core-physics.md), [CLI](repo-map/cli-and-packaging.md) |
| `xsgen` | Generation and resolution of external Fortran tables | [Core physics](repo-map/core-physics.md) |

Root modules (`api.py`, `scoring.py`, `phases.py`, `_backend.py`, `_precision.py`, …) are
described on the detail page of the area that uses them.

## Static package dependency DAG

<!-- BEGIN GENERATED PACKAGE DEPENDENCIES -->
<!-- Generated by pyrite.devtools.repo_map; update with `pyrite-dev repo-map --write`. -->

Arrows point from importer to imported package. Strongly connected packages are
collapsed into one node so this static condensation graph is a DAG. Imports hidden
behind lazy dispatch, ``__getattr__``, or ``import_module`` are outside this scan.

```text
Components (a + b denotes a static import cycle):
  p0: _entry
  p1: _util
  p2: apps
  p3: beams
  p4: campaign
  p5: checkpoints
  p6: cli
  p7: config
  p8: console
  p9: detectors
  p10: devtools
  p11: energy_grid
  p12: execution
  p13: experimental
  p14: geometry
  p15: instrument
  p16: materials
  p17: montecarlo
  p18: observations
  p19: optimization
  p20: perf
  p21: plots
  p22: quantum
  p23: radiation
  p24: remote
  p25: results
  p26: runs
  p27: storage
  p28: tools
  p29: transport
  p30: validation
  p31: xsgen
Edges (importer -> imported):
  p0 -> p16
  p0 -> p30
  p0 -> p6
  p0 -> p8
  p10 -> p1
  p10 -> p11
  p10 -> p21
  p10 -> p25
  p10 -> p26
  p10 -> p30
  p10 -> p4
  p10 -> p6
  p10 -> p8
  p11 -> p1
  p11 -> p16
  p11 -> p17
  p11 -> p24
  p11 -> p25
  p11 -> p4
  p11 -> p5
  p11 -> p8
  p11 -> p9
  p12 -> p1
  p12 -> p17
  p12 -> p4
  p12 -> p5
  p15 -> p1
  p15 -> p16
  p15 -> p9
  p16 -> p1
  p17 -> p1
  p17 -> p16
  p17 -> p22
  p17 -> p31
  p17 -> p8
  p18 -> p1
  p18 -> p15
  p18 -> p17
  p18 -> p25
  p18 -> p8
  p18 -> p9
  p19 -> p25
  p19 -> p26
  p19 -> p4
  p2 -> p1
  p2 -> p14
  p2 -> p16
  p2 -> p17
  p2 -> p18
  p2 -> p21
  p2 -> p24
  p2 -> p25
  p2 -> p26
  p2 -> p27
  p2 -> p3
  p2 -> p30
  p2 -> p4
  p2 -> p5
  p2 -> p7
  p2 -> p8
  p2 -> p9
  p20 -> p16
  p20 -> p17
  p20 -> p4
  p20 -> p8
  p21 -> p1
  p21 -> p16
  p21 -> p17
  p21 -> p25
  p21 -> p8
  p21 -> p9
  p22 -> p1
  p22 -> p16
  p24 -> p1
  p24 -> p16
  p24 -> p17
  p24 -> p20
  p24 -> p25
  p24 -> p26
  p24 -> p30
  p24 -> p31
  p24 -> p4
  p24 -> p5
  p24 -> p8
  p25 -> p1
  p25 -> p15
  p25 -> p16
  p25 -> p17
  p25 -> p22
  p25 -> p9
  p26 -> p1
  p26 -> p12
  p26 -> p15
  p26 -> p16
  p26 -> p17
  p26 -> p18
  p26 -> p20
  p26 -> p22
  p26 -> p25
  p26 -> p31
  p26 -> p4
  p26 -> p5
  p26 -> p8
  p30 -> p16
  p30 -> p17
  p30 -> p4
  p30 -> p5
  p30 -> p8
  p30 -> p9
  p31 -> p1
  p31 -> p16
  p31 -> p8
  p4 -> p1
  p4 -> p14
  p4 -> p15
  p4 -> p16
  p4 -> p17
  p4 -> p18
  p4 -> p22
  p4 -> p25
  p4 -> p31
  p4 -> p7
  p4 -> p9
  p5 -> p1
  p5 -> p16
  p5 -> p17
  p5 -> p25
  p5 -> p4
  p5 -> p8
  p6 -> p1
  p6 -> p11
  p6 -> p14
  p6 -> p15
  p6 -> p16
  p6 -> p17
  p6 -> p20
  p6 -> p24
  p6 -> p26
  p6 -> p31
  p6 -> p4
  p6 -> p5
  p6 -> p8
  p6 -> p9
  p8 -> p1
  p8 -> p16
  p9 -> p1
  p9 -> p16
  p9 -> p7
```
<!-- END GENERATED PACKAGE DEPENDENCIES -->

`detectors.timepix_response` / `detectors.eaglexo_response` depend only on `materials.crystal` plus shared `_si_sensor` plumbing (Si constants, response caching, Poisson core). Packaged data resolve via `pyrite.DATA_DIR` — imports work from any cwd.
