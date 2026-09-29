# Development environment and focused verification

The repository has one uv project and one publishable distribution. The root project owns the `pyrite-xray` distribution, `src/pyrite/`, packaged data, `pyrite`, `pyrite-dev`, and the test suite. Contributor tools are dependency groups in the root `pyproject.toml`; there is no uv workspace split or separate test-tools package.

## Why source was not split across distributions

The import and data inventory has one natural wheel owner. Core material and Monte Carlo modules feed `sweep`, `results`, profiles, configuration, and run drivers; plotting consumes results and Monte Carlo APIs; the lazy CLI dispatches drivers, remote orchestration, plotting, and apps. All public imports live under `pyrite`, and `pyrite.DATA_DIR` owns the bundled catalogs, CIFs, detector data, and validation fixtures.

Splitting that namespace between wheels would make installation order decide which files and data survive. Renaming implementation namespaces would add a large compatibility layer without producing isolation in uv's shared workspace environment. Therefore source, apps, and both entry points remain in the root wheel. A third analysis/app member is deferred.

## Commands

```bash
# Normal contributor setup: root package and every dependency group
# (`tool.uv.default-groups = "all"`; `uv run` syncs the same set).
uv sync --locked

# Runtime-only environment, used by remote installations.
uv sync --no-default-groups --locked

# Stable domain partitions. Together these contain every tests/test_*.py once.
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core --durations=30 --durations-min=0.5
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite apps
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite packaging

# One focused test module or selection.
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test path/to/test.py -k test_name

# Static and formatting checks.
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev format
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev precommit

# Additive cross-boundary sample and release gates.
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite integration
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify --skip-tests

# Clean wheel and editable-install compatibility check.
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev package-smoke
```

Use the project runner rather than bare `pytest` or an environment-specific Python path. If the project environment is not writable, add `UV_PROJECT_ENVIRONMENT=/tmp/pyrite-venv` instead of switching interpreters.

Suite ownership uses deterministic filename rules in `pyrite._dev`. A regression test requires the four domain suites to cover every test module exactly once, so a new test cannot silently disappear from focused coverage. The integration suite intentionally overlaps domain suites; it exercises public imports/data, exports, CLI contract, remote, sweep/run, and a headless app path. `pyrite-dev docs` performs the clean offline warnings-as-errors Sphinx build; `verify` includes that documentation gate along with skills, imports, generated repository structure, lint, types, and tests. CI runs the four domain suites once each with adaptive workers (and slow-test timings for core), then uses `verify --skip-tests` for the remaining checks; local `verify` still runs everything.

`pyrite-dev test`, `test-suite`, and the test step of `verify` use `pytest-xdist` workers by default: `min(CPUs, available memory / 2 GiB, 6)`, since a worker peaks at roughly 1.2–1.8 GiB. Set `PYRITE_TEST_WORKERS=N` to choose the count (`1` runs serially). `test` runs that name test paths remain serial by default; all commands honor explicit `-n`/`--numprocesses` or `-p no:xdist`. Pytest keeps `tmp_path` directories only for failed tests.

## Coverage

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --cov
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test --numba --cov
```

`--cov` is forwarded straight to `pytest-cov`, which reads `[tool.coverage.*]` in `pyproject.toml` for source/branch/report settings; any other `pytest-cov`/`coverage.py` flag (`--cov-report=html`, `-k`, ...) composes the same way. `--numba` must come before other forwarded arguments; it sets `NUMBA_DISABLE_JIT=1` so `@njit` bodies (`montecarlo/transport.py`, `geometry.py`, `groove.py`) run under the Python tracer instead of compiled, at roughly 2x wall clock. Read the resulting totals against the two compiled-code caveats documented next to `[tool.coverage.report]` in `pyproject.toml`: the CuPy kernel modules (`montecarlo/transport/_jit_*.py`, `montecarlo/spectrum/*_jit_kernel.py`) report 0% on any environment without the `nvidia` extra, and `@njit` bodies need `--numba` to be measured at all.

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
