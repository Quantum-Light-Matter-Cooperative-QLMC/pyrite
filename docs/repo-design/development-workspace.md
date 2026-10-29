# Development environment and focused verification

The repository has one uv project and one publishable distribution. The root project owns the `pyrite-mc` distribution, `src/pyrite/`, packaged data, `pyrite`, `pyrite-dev`, and the test suite. Contributor tools are dependency groups in the root `pyproject.toml`; there is no uv workspace split or separate test-tools package.

## Install from source

For installed-package use, follow [Getting started](../guides/getting-started.md).
Contributor setup requires Git and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
uv sync --locked
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

A plain sync includes the contributor tools and interactive apps. Run commands
through `uv run` without activation, or activate the environment in each shell.

## Extras versus dependency groups

Dependencies imported by shipped code for a user-facing feature are package extras, published in wheel metadata; contributor, CI, and validation-only tools are dependency groups, which wheels never carry. `plots` holds the runtime of `pyrite.plots` (and the lazily imported matplotlib figures in `perf/` and `validation/`); `apps` is `pyrite-mc[plots]` plus the marimo stack for `src/pyrite/apps/`. The `notebooks` group is `pyrite-mc[apps]` plus notebook conversion tools (`nbconvert`, `pandoc`) that nothing in `src/` imports, so a plain `uv sync` stays complete. Entry points that need an extra check it with `pyrite._extras.require_extra` (CLI: `pyrite.cli._extras.require_cli_extra`), which names `pyrite-mc[<extra>]` and `uv sync --extra <extra>` without importing the dependency. `pyrite-dev package-smoke` installs the built wheel without extras to assert those hints, then with `[apps]` to smoke-launch each app.

## Why source was not split across distributions

The import and data inventory has one natural wheel owner. Core material and Monte Carlo modules feed `sweep`, `results`, profiles, configuration, and run drivers; plotting consumes results and Monte Carlo APIs; the lazy CLI dispatches drivers, remote orchestration, plotting, and apps. All public imports live under `pyrite`, and `pyrite.DATA_DIR` owns the bundled catalogs, CIFs, detector data, and validation fixtures.

Splitting that namespace between wheels would make installation order decide which files and data survive. Renaming implementation namespaces would add a large compatibility layer without producing isolation in uv's shared workspace environment. Therefore source, apps, and both entry points remain in the root wheel. A third analysis/app member is deferred.

## Commands

Commands below assume `pyrite` and `pyrite-dev` are on your `PATH`: activate the project environment (`source .venv/bin/activate`) after `uv sync`, or install them as a tool (`uv tool install .`).

```bash
# Normal contributor setup: root package and every dependency group
# (`tool.uv.default-groups = "all"`; `uv run` syncs the same set).
uv sync --locked

# Runtime-only environment, used by remote installations.
uv sync --no-default-groups --locked

# Start with the changed behavior's test node or owning module.
pyrite-dev test path/to/test.py -k test_name

# Domain partitions for changes spanning a domain. Together these contain
# every tests/**/test_*.py once; core is shared/residual, not the full suite.
pyrite-dev test-suite transport
pyrite-dev test-suite transport-collisions
pyrite-dev test-suite transport-cascades
pyrite-dev test-suite transport-runner
pyrite-dev test-suite spectrum
pyrite-dev test-suite spectrum-coherent
pyrite-dev test-suite spectrum-brem
pyrite-dev test-suite quantum
pyrite-dev test-suite channeling
pyrite-dev test-suite energy-grid
pyrite-dev test-suite materials
pyrite-dev test-suite xsgen
pyrite-dev test-suite core
pyrite-dev test-suite cli
pyrite-dev test-suite apps
pyrite-dev test-suite packaging

# Include the opt-in slow tier (CI always does).
pyrite-dev test --slow path/to/test.py
pyrite-dev test-suite transport --slow

# Fresh-process CPU imports and README simulation; see the startup reference.
pyrite-dev startup --cache warm --repeats 3

# Static and formatting checks.
pyrite-dev lint
pyrite-dev format
pyrite-dev typecheck
pyrite-dev check-lines
pyrite-dev complexity          # report; --check is the high-CC ratchet
pyrite-dev precommit

# Cross-boundary sample only when those public boundaries are affected.
pyrite-dev test-suite integration

# Validate and render committed tutorial captures alongside repository checks.
pyrite-dev verify

# Full slow-inclusive CPU suite: explicit audit/release operation.
pyrite-dev test --slow

# Clean wheel and editable-install compatibility check.
pyrite-dev package-smoke
```

Agent and git hooks:

- `uv run pre-commit install` installs both configured stages: commit-time Ruff,
  notebook stripping and skill-mirror sync, and pre-push import-linter,
  `check-skills` and `repo-map --check`. `pyrite-dev verify` stays the full
  repository gate, including tutorial capture validation and rendering.
- `.agents/hooks/guard_bash.sh` (Claude Code and Codex `PreToolUse`) blocks
  pattern kills, bulk `git add`, non-lease force pushes, and pushes to `main`.
- `.agents/hooks/ruff_after_edit.sh` (Claude Code `PostToolUse`) formats each
  edited Python file and applies safe Ruff fixes, leaving unused-import
  findings for the agent.
- `.claude/settings.json` pins Claude Code's tool shell to bash
  (`CLAUDE_CODE_SHELL`); zsh does not word-split unquoted variables.

Use the project runner rather than bare `pytest` or an environment-specific Python path. If the project environment is not writable, add `UV_PROJECT_ENVIRONMENT=/tmp/pyrite-venv` instead of switching interpreters.

See [CPU startup latency](compute/startup-latency.md) for benchmark cache policies,
profiling, and the recorded baseline.

Suite ownership uses ordered filename rules in `pyrite.devtools.dev_cli`. First match owns a module; `core` owns the remainder. Regression tests require all primary suites to partition every test module exactly once and require CI's matrix to match that inventory. Empty selections fail rather than accidentally invoking the full suite. The `integration` suite intentionally overlaps primary suites; it samples public imports/data, exports, CLI contract, remote, sweep/run, and a headless app path.

| Suite | Ownership |
|---|---|
| `transport` | Flight integration, beam/RNG entry, geometry, grooves, event/segment contracts and transport-core diagnostics. |
| `transport-collisions` | Atomic/shell/elastic scattering, stopping, energy-loss straggling and their configuration/table contracts. |
| `transport-cascades` | Coupled hard-radiative flights, secondary transport, pair conversion, positrons and annihilation, including process-specific runner tests. |
| `transport-runner` | Shared case/adaptive/block scheduling, backend/resource policy, trajectory artifacts, telemetry and cross-stage baseline helpers. |
| `spectrum` | Shared/incoherent CXR, absorption/escape, bin integration, mosaicity, characteristic and transition radiation, spectrum staging and mixed emission tests. |
| `spectrum-coherent` | Coherent CXR fields, physical-population/transverse factors, formation absorption, omission and temporal intensity. |
| `spectrum-brem` | Analytic/EEDL/BremsLib bremsstrahlung scoring, hard-photon event scoring, source/angular comparisons and positron scaling. |
| `quantum` | Channeling bound states, density-matrix and MCWF solvers, scattering operators and dipole-emission physics. |
| `channeling` | Channeling selection/configuration, source production, native-transport handoff, replay and physical observations. |
| `energy-grid` | Grid policy, derivation, artifacts and convergence; CLI module stays in `cli`. |
| `materials` | Material/catalog/crystal/attenuation tests, after packaging, apps, CLI and channeling exceptions. |
| `xsgen` | Cross-section generation, table stores, fetch/release tooling and external-code adapters. |
| `core` | Shared/residual tests: API/campaign, checkpoints/results/observations, detector/instrument, plotting, remaining remote and developer tooling. |
| `cli`, `apps`, `packaging` | Existing public-boundary selections take precedence over domain rules. |

Transport and spectrum family selectors are disjoint leaves: `transport` and
`spectrum` do not rerun their sibling suites. `quantum` covers channeling solver
physics; classical coherent CXR remains in `spectrum-coherent`. Mixed cascade
runner tests stay with their physical process, while shared scheduling and
trajectory artifacts live in `transport-runner`. Grid-specific coherent tests
remain in `energy-grid`.

For ordinary edits, select the changed behavior's node/module and trace its affected callers and consumers. Widen to a domain only for changes spanning that domain or its fixtures; do not run all of `core` as a generic safety check. See `AGENTS.md` "Choose tests by impact" for examples. Documentation/guidance-only work needs relevant static checks, not simulation tests. Repeat checks only after new edits, failures or unresolved concerns.

`verify` never invokes pytest. It checks types, Ruff lint and formatting
(without formatting files), the 1,250-line Python source-module budget, the
high-CC ratchet ([source complexity baseline](complexity.md)), the API inventory,
a clean warnings-as-errors Sphinx build, and import-linter.

The HTML documentation build validates committed tutorial captures and renders
their stored outputs into untracked HTML under `docs/_build/`, without executing
notebooks or fetching reference tables. Missing or stale captures fail the build.
`verify` calls that same build; it does not update notebook sources, release pins
or committed captures. Regenerate affected captures explicitly on a suitably
provisioned worker with the required dependencies and reference tables, then
commit them with their sources. See
[tutorial authoring](documentation.md#interactive-notebook-builds). The separate
browser results demo is exported to WASM during the build and executes only
when opened in a browser.

`--skip-tests` remains an undocumented compatibility no-op; pytest arguments
and `--slow` are rejected. Run tests explicitly. CI runs each primary domain
with the `slow` tier once per supported Python version (3.13 and 3.14), with
adaptive workers and duration reporting; skills and generated repository-map
checks remain separate CI steps.

The suites read hash-pinned fetched data rather than packaged data (ADR-0014): the EPICS2025 EEDL and EADL files, the derived EPDL photon table, SBETHE's `sdbase/` and catalogue stopping tables, and the ELSEPA and BremsLib tables, with the positron SBETHE and ELSEPA pins alongside the electron ones. Install all of it once per machine with `pyrite tables fetch` (about 140 MB; one code with `pyrite tables fetch CODE`, a positron pin with `pyrite tables fetch {sbethe-tables,elsepa} --projectile positron`, or `--archive PATH` from a local copy); a test that needs one fails naming the command. Positron tests check every table their pinned release lists (`tests/helpers/positron_tables.py`): none installed skips, naming the fetch command (fails instead when `CI` is set, so CI never skips a positron anchor silently); some installed fails, naming the command. Tests never see a developer's `PYRITE_HOME` or config store, so they read the default user-data location, `~/.local/share/pyrite/datasets/` on Linux; fetch without `PYRITE_HOME` set. PyRITE's own archives (EPDL, SBETHE tables, ELSEPA, BremsLib, and the EEDL/EADL mirrors) are public GitHub Release assets, so fetching them needs no GitHub token. CI caches the datasets, `xsgen/tables` and `xsgen/reference-data` directories under a key derived from `src/pyrite/datasets.py`, `src/pyrite/xsgen/fetch.py` (the `sdbase/` pin) and `src/pyrite/data/xsgen/*-tables.json`, and runs `pyrite tables fetch`; they verify a cached copy and need no network on a hit. The `elastic_model="mott"` tests read the synthetic SRD 64-format tables in `tests/data/mott_srd64_synthetic/` (regenerate with `uv run python -m tests.helpers.mott_synthetic`).

`pyrite-dev test` and `test-suite` use `pytest-xdist` workers by default: `min(CPUs, available memory / 2 GiB, 6)`. This is a sizing heuristic, not a hard memory bound; individual heavy tests can exceed it. Set `PYRITE_TEST_WORKERS=N` to choose the count (`1` runs serially). `test` runs that name test paths remain serial by default; both commands honor explicit `-n`/`--numprocesses` or `-p no:xdist`. Pytest keeps `tmp_path` directories only for failed tests. Run full slow audits remotely, starting with `OPENBLAS_NUM_THREADS=1` for parallel CPU work; see the [measured memory and axial-tail audit](compute/slow-test-audit.md) before choosing more threads or running a high-memory node locally.

Tests marked `slow` are skipped by default. The marker covers tests that take more than 5 s wall time or have a large peak RSS, plus rarely-needed exhaustive cases. The skip reason and a closing summary line say how to enable them. Pass `--slow` to `test` or `test-suite` (before forwarded pytest arguments), or set `PYRITE_SLOW_TESTS=1`. A slow test named by node id (`path::test_name`), or selected by any `-m` expression that names `slow`, runs without the flag. CI sets `PYRITE_SLOW_TESTS=1` on every PR, so physics anchors may carry the marker. Tag a whole heavy module with `pytestmark = pytest.mark.slow`.

## Coverage

```bash
pyrite-dev test --cov
pyrite-dev test --numba --cov
pyrite-dev test --slow --numba --cov
```

`--cov` is forwarded straight to `pytest-cov`, which reads `[tool.coverage.*]` in `pyproject.toml` for source/branch/report settings; any other coverage flag (`--cov-report=html`, `-k`, ...) composes the same way. `--numba` must precede forwarded arguments; it sets `NUMBA_DISABLE_JIT=1` so CPU `@njit` bodies run under the Python tracer. These audit timings are not comparable to compiled CI runs. Include `--slow` for a complete CPU audit; default coverage omits those tests.

Read totals against the compiled-code caveats beside `[tool.coverage.report]` in `pyproject.toml`. For a CPU-only assessment, separately report exclusions for the CUDA device helpers/kernels (`transport/_jit_*device.py`, `_jit_*kernel.py`, `_jit_annihilation.py`, and `spectrum/*_jit_kernel.py`); retain host dispatch/fallback and error paths. In particular, `transport/_jit_radiative.py` is **CPU Numba**, despite its name: do not exclude it with a blanket `_jit_*` glob. Hardware, online, external-Fortran and optional-oracle skips still limit what a CPU run establishes; report them alongside line and branch coverage.

## Measurements (2026-08-01)

Measured on this WSL worktree with uv 0.11.28, Python 3.14, warm cache, CPU backend, and the repository `.venv`:

| Workload | Baseline pytest / wall | Final pytest / wall | Final peak RSS | Result |
|---|---:|---:|---:|---|
| collect full suite | 9.25 / 11.49 s | 2.61 / 3.83 s | 343 MB | 2,276 tests after 3 new tests |
| representative core | 1.20 / 2.00 s | 1.13 / 1.85 s | 236 MB | 11 passed |
| representative CLI/remote | 1.71 / 2.60 s | 1.25 / 2.04 s | 239 MB | 178 passed |
| representative apps | 0.21 / 0.91 s | 0.11 / 0.50 s | 88 MB | 27 passed |
| full suite | 97.57 / 100.22 s | 85.10 / 87.50 s | 1.34 GB | final: 2,237 passed, 39 skipped, 1 sandbox-only deselection |

Final domain partitions measured 945 passed + 39 skipped + one sandbox-only deselection for core (52.27 / 53.97 s), 839 passed for CLI (27.11 / 28.05 s), 279 passed for apps (19.42 / 21.10 s), and 173 passed for packaging (4.66 / 5.54 s). The additive integration sample ran 634 tests in 30.80 / 31.71 s.

The baseline full-suite failure was the known restricted-sandbox forkserver bind error: `PermissionError: [Errno 1] Operation not permitted` in `test_run_cases_engine_cpu_end_to_end_returns_finite_spectrum`. Collection and focused final runs benefited from a hotter filesystem/import cache than the baseline, so the observed deltas are not attributed to workspace metadata. No speed or isolation claim follows from this report.

Outside the restricted sandbox, `marimo check` passed for all four apps and `pyrite app analysis launch --smoke` completed successfully during the recorded measurement run.
