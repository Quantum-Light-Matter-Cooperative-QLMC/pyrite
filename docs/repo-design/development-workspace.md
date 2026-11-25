# Development environment and focused verification

The repository has one pixi workspace and one publishable distribution. The root project owns the `pyrite-mc` distribution, `src/pyrite/`, packaged data, `pyrite`, `pyrite-dev`, and the test suite. Contributor tools are dependency groups in the root `pyproject.toml`; there is no workspace split or separate test-tools package. pixi manages development, CI and lab environments only (#1462). Users install the published package with pip or any other installer, and the package metadata is unchanged.

## Install from source

For installed-package use, follow [Getting started](../guides/getting-started.md).
Contributor setup requires Git and [pixi](https://pixi.sh):

```bash
git clone https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite.git
cd pyrite
pixi install --locked
pixi run pyrite-dev --help
```

`pixi run` uses the `default` environment: Python 3.13, every dependency group,
the `dask` and `optimization` extras and the interactive apps. Run commands
through `pixi run`, or open an activated shell with `pixi shell`.

## Pixi workspace

`[tool.pixi]` in `pyproject.toml` defines the environments; `pixi.lock`
pins every one of them, for every platform it supports. CI and lab jobs
install with `--locked`, so a manifest edit without `pixi lock` fails fast.

| Environment | Platforms | Contents |
|---|---|---|
| `default` | linux-64, osx-arm64, win-64 | Python 3.13; groups `test`, `lint`, `docs`, `notebooks`, `oracle`, `browser`; extras `dask`, `optimization` |
| `py314` | linux-64, osx-arm64, win-64 | `default` on Python 3.14 (CI's second interpreter) |
| `docs` | linux-64, osx-arm64, win-64 | Python 3.13 with the `docs` group only (documentation workflow, release notes) |
| `nvidia` | linux-64 (CUDA 13 driver) | conda-forge `cupy`, `cufinufft` and `cuda-version 13.*`; `test`, `docs`, `notebooks`, `dask`, `optimization` |
| `amd` | linux-64 | PyPI `amd` extra (CuPy built against ROCm with `CUPY_INSTALL_USE_HIP=1`); same groups as `nvidia` without `docs` |
| `intel` | linux-64 | PyPI `intel` extra (`dpnp`, `dpctl`); same groups as `amd` |
| `svml` | linux-64 | Intel SVML Numba from Anaconda `pkgs/main`; `test` only. Separate terms: [`LICENSE-SVML.md`](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE-SVML.md) |
| `viz` | linux-64 | One-off research viewers and digitizers (`pyvista`, `trame`, `pymupdf`, `scikit-image`) |

Design decisions (#1462):

- **Channels.** conda-forge for every environment. Anaconda `pkgs/main` is
  used only by `svml`, at a higher priority, because only that channel ships
  the SVML-patched `llvmlite`. Its terms and Intel's EULA for
  `intel-cmplr-lib-rt` are stricter than the project licence, so the `svml`
  build carries its own notice. CI builds and tests it but publishes nothing
  built from it.
- **conda versus PyPI.** conda supplies the interpreter and the native stacks
  that wheels cannot express in one toolkit. That is the CUDA 13 CuPy and
  cuFINUFFT pair in `nvidia`, which replaces the PyPI extra's CUDA 12 runtime
  wheels, and the SVML Numba in `svml`. Everything else, including the
  project, installs from PyPI at the versions `uv.lock` pinned before the
  move, so default numerical output did not change. `nvidia` never includes
  the PyPI `nvidia` extra, which would install CuPy twice.
- **No double installs.** pixi skips a PyPI requirement only when its
  conda-to-PyPI map names the conda package. conda-forge ships that map;
  `pkgs/main` does not, so `[tool.pixi.workspace.conda-pypi-map]` maps
  `numpy`, `numba` and `llvmlite` for it. `tests/dev/test_pixi_lock.py`
  fails if any locked environment installs one distribution from both
  sources.
- **Platforms.** Every environment on linux-64. The CPU-only `default`,
  `py314` and `docs` environments also lock osx-arm64 and win-64. GPU, `svml`
  and `viz` are Linux-only.
- **Python.** 3.13 (`py313` feature) everywhere except `py314`. CI runs each
  primary test domain in `default` and `py314`.
- **Floors.** `numpy>=2.2` and `numba>=0.61` are the oldest releases the code
  supports, which lets `svml` keep its conda builds. The other environments
  still resolve the latest releases.
- **uv.** It is not used to manage environments. The `docs` feature carries a
  `uv` binary only because marimo's HTML-WASM export runs Ruff through
  `uv tool run`. Its tool and cache directories stay under `.pixi/`, so
  sandboxed builds need no setup. Historical evidence records keep the `uv
  run` commands that produced them; read them as `pixi run`.

Select an environment with `-e`, for example `pixi run -e nvidia pyrite-dev
test tests/montecarlo/spectrum/test_sinc_bin_integration_cuda.py`. After
editing dependencies, run `pixi lock` and commit `pixi.lock` with the
manifest.

## Extras versus dependency groups

Dependencies imported by shipped code for a user-facing feature are package extras, published in wheel metadata; contributor, CI, and validation-only tools are dependency groups, which wheels never carry. pixi turns each extra and each group into a feature of the same name. `plots` holds the runtime of `pyrite.plots` (and the lazily imported matplotlib figures in `perf/` and `validation/`); `apps` is `pyrite-mc[plots]` plus the marimo stack for `src/pyrite/apps/`. The `notebooks` group is `pyrite-mc[apps]` plus notebook conversion tools (`nbconvert`, `pandoc`) that nothing in `src/` imports, so the `default` environment stays complete. Entry points that need an extra check it with `pyrite._util.extras.require_extra` (CLI: `pyrite.cli._extras.require_cli_extra`), which names `pyrite-mc[<extra>]` and the checkout's pixi environment (`pyrite._util.extras.checkout_environment`) without importing the dependency. `pyrite-dev package-smoke` builds the wheel and installs it with plain `venv` and `pip`, without extras to assert those hints, then with `[apps]` to smoke-launch each app.

## Why source was not split across distributions

The import and data inventory has one natural wheel owner. Core material and Monte Carlo modules feed `sweep`, `results`, profiles, configuration, and run drivers; plotting consumes results and Monte Carlo APIs; the lazy CLI dispatches drivers, remote orchestration, plotting, and apps. All public imports live under `pyrite`, and `pyrite.DATA_DIR` owns the bundled catalogs, CIFs, detector data, and validation fixtures.

Splitting that namespace between wheels would make installation order decide which files and data survive. Renaming implementation namespaces would add a large compatibility layer without producing isolation in a shared workspace environment. Therefore source, apps, and both entry points remain in the root wheel. A third analysis/app member is deferred.

## Commands

Commands below assume `pyrite` and `pyrite-dev` are on your `PATH`: run them inside `pixi shell`, or prefix each with `pixi run`.

```bash
# Normal contributor setup: the default environment (every dependency group).
pixi install --locked

# A lab GPU environment, as remote jobs install it.
pixi install --locked -e nvidia

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

# Identity pins only; after a *_MODEL / tables-*-N bump, re-mint them in one pass.
pyrite-dev test-suite identity-pins
pyrite-dev test-suite identity-pins --update-pins

# Validate and render committed tutorial captures alongside repository checks.
pyrite-dev verify

# Full slow-inclusive CPU suite: explicit audit/release operation.
pyrite-dev test --slow

# Clean wheel and editable-install compatibility check.
pyrite-dev package-smoke
```

Agent and git hooks:

- `pixi run pre-commit install` installs both configured stages: commit-time Ruff,
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

Use the project runner rather than bare `pytest` or an environment-specific Python path. If pixi's package cache is not writable, set `PIXI_CACHE_DIR=/tmp/pyrite-pixi-cache` instead of switching interpreters.

See [CPU startup latency](compute/startup-latency.md) for benchmark cache policies,
profiling, and the recorded baseline.

Suite ownership is structural (`pyrite.devtools.dev_cli`). Monte Carlo tests live in suite directories under `tests/montecarlo/` (`transport/`, `collisions/`, `cascades/`, `runner/`, `spectrum/`, `coherent/`, `brem/`, `channeling/`, `quantum/`), and `tests/energy-grid/`, `tests/materials/` and `tests/xsgen/` own their suites, so a new module joins its directory's suite with no routing edit. A short explicit list of cross-directory exceptions (the public-boundary `cli`, `apps` and `packaging` selections, plus channeling tests under `tests/materials/`) takes precedence; `core` owns the remainder. Only listed exceptions may sit directly in `tests/montecarlo/`. Regression tests require all primary suites to partition every test module exactly once and require CI's matrix to match that inventory. Empty selections fail rather than accidentally invoking the full suite. The `integration` suite intentionally overlaps primary suites; it samples public imports/data, exports, CLI contract, remote, sweep/run, and a headless app path. The `identity-pins` suite also overlaps: it runs only tests marked `identity_pin` (auto-applied to users of the `identity_pin` fixture, `tests/helpers/identity_pins.py`) in the modules listed in `IDENTITY_PIN_TESTS`. Pinned dataset/case digests and named stems live in `tests/data/identity-pins.json`; the physics-baseline identity fields stay in `tests/data/physics-baseline/resolved.json` and `channeling.json`. `--update-pins` rewrites every moved pin in one run instead of failing (case payloads are still asserted), so a deliberate identity generation change is re-minted by one command and reviewed as a JSON diff. A regression test requires every module that requests the fixture to be listed.

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

The suites read hash-pinned fetched data rather than packaged data (ADR-0014): the EPICS2025 EEDL and EADL files, the derived EPDL photon table, SBETHE's `sdbase/` and catalogue stopping tables, and the ELSEPA and BremsLib tables, with the positron SBETHE and ELSEPA pins alongside the electron ones. Install all of it once per machine with `pyrite tables fetch` (about 140 MB; one code with `pyrite tables fetch CODE`, a positron pin with `pyrite tables fetch {sbethe-tables,elsepa} --projectile positron`, or `--archive PATH` from a local copy); a test that needs one fails naming the command. Positron tests check every table their pinned release lists (`tests/helpers/positron_tables.py`): none installed skips, naming the fetch command (fails instead when `CI` is set, so CI never skips a positron anchor silently); some installed fails, naming the command. Tests never see a developer's `PYRITE_HOME` or config store, so they read the default user-data location, `~/.local/share/pyrite/datasets/` on Linux; fetch without `PYRITE_HOME` set. PyRITE's own archives (EPDL, SBETHE tables, ELSEPA, BremsLib, and the EEDL/EADL mirrors) are public GitHub Release assets, so fetching them needs no GitHub token. CI caches the datasets, `xsgen/tables` and `xsgen/reference-data` directories under a key derived from `src/pyrite/datasets.py`, `src/pyrite/xsgen/fetch.py` (the `sdbase/` pin) and `src/pyrite/data/xsgen/*-tables.json`, and runs `pyrite tables fetch`; they verify a cached copy and need no network on a hit. The `elastic_model="mott"` tests read the synthetic SRD 64-format tables in `tests/data/mott_srd64_synthetic/` (regenerate with `pixi run python -m tests.helpers.mott_synthetic`).

`pyrite-dev test` and `test-suite` use `pytest-xdist` workers by default: `min(CPUs, available memory / 2 GiB, 6)`. This is a sizing heuristic, not a hard memory bound; individual heavy tests can exceed it. Set `PYRITE_TEST_WORKERS=N` to choose the count (`1` runs serially). `test` runs that name test paths remain serial by default; both commands honor explicit `-n`/`--numprocesses` or `-p no:xdist`. Whenever a run uses two or more xdist workers (default or explicit `-n`), the pytest processes default to `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` of `1`; numpy and scipy each load their own OpenBLAS, and default pools per worker oversubscribe the cores and slow small dense linear algebra by more than an order of magnitude (#1021). Values already set in the environment win; serial and targeted runs are not capped. Pytest keeps `tmp_path` directories only for failed tests. Run full slow audits remotely, starting with `OPENBLAS_NUM_THREADS=1` for parallel CPU work; see the [measured memory and axial-tail audit](compute/slow-test-audit.md) before choosing more threads or running a high-memory node locally.

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

Measured before the pixi migration, on this WSL worktree with uv 0.11.28, Python 3.14, warm cache, CPU backend, and the repository `.venv`:

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
