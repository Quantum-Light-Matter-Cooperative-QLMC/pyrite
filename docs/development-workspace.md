# uv workspace and focused verification

The repository is a two-member uv workspace with one publishable distribution:

```text
cxr-mc-tests (internal dependency/test tooling; never published)
    └── cxr-mc (sole owner of cxr_mc/, packaged data, cxr, and cxr-dev)
```

`cxr-mc` remains the root/default project. `packages/cxr-mc-tests` is a
non-package workspace member: it owns pytest dependencies and depends one way
on `cxr-mc`. Both members share uv's single lockfile and environment. Selecting
a member controls dependency installation; it does not provide dependency or
import isolation.

## Why source was not split across distributions

The import and data inventory has one natural wheel owner. Core material and
Monte Carlo modules feed `sweep`, `results`, profiles, configuration, and run
drivers; plotting consumes results and Monte Carlo APIs; the lazy CLI dispatches
drivers, remote orchestration, plotting, and apps. All public imports live under
`cxr_mc`, and `cxr_mc.DATA_DIR` owns the bundled catalogs, CIFs, detector data,
and validation fixtures.

Splitting that namespace between wheels would make installation order decide
which files and data survive. Renaming implementation namespaces would add a
large compatibility layer without producing isolation in uv's shared workspace
environment. Therefore source, apps, and both entry points remain in the root
wheel. A third analysis/app member is deferred.

## Commands

```bash
# Normal contributor setup: all workspace members and test tools.
uv sync --locked

# Runtime/root-only environment, used by remote installations.
uv sync --package cxr-mc --no-dev --locked

# Explicit test-tool member path.
uv sync --package cxr-mc-tests --locked
uv run --package cxr-mc-tests cxr-dev test-suite core

# Stable domain partitions. Together these contain every tests/test_*.py once.
uv run --package cxr-mc-tests cxr-dev test-suite core
uv run --package cxr-mc-tests cxr-dev test-suite cli
uv run --package cxr-mc-tests cxr-dev test-suite apps
uv run --package cxr-mc-tests cxr-dev test-suite packaging

# Additive cross-boundary sample and unchanged release gate.
uv run --package cxr-mc-tests cxr-dev test-suite integration
uv run --package cxr-mc-tests cxr-dev verify

# Clean wheel and editable-install compatibility check.
uv run --package cxr-mc-tests cxr-dev package-smoke
```

Suite ownership uses deterministic filename rules in `cxr_mc._dev`. A
regression test requires the four domain suites to cover every test module
exactly once, so a new test cannot silently disappear from focused coverage.
The integration suite intentionally overlaps domain suites; it exercises public
imports/data, exports, CLI contract, remote, sweep/run, and a headless app path.

## Coverage

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test --cov
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test --numba --cov
```

`--cov` is forwarded straight to `pytest-cov`, which reads
`[tool.coverage.*]` in `pyproject.toml` for source/branch/report settings; any
other `pytest-cov`/`coverage.py` flag (`--cov-report=html`, `-k`, ...) composes
the same way. `--numba` must come before other forwarded arguments; it sets
`NUMBA_DISABLE_JIT=1` so `@njit` bodies (`montecarlo/transport.py`,
`geometry.py`, `groove.py`) run under the Python tracer instead of compiled,
at roughly 2x wall clock. Read the resulting totals against the two
compiled-code caveats documented next to `[tool.coverage.report]` in
`pyproject.toml`: the CuPy kernel modules (`montecarlo/*_jit_kernel.py`)
report 0% on any environment without the `nvidia` extra, and `@njit` bodies
need `--numba` to be measured at all.

## Measurements (2026-08-01)

Measured on this WSL worktree with uv 0.11.28, Python 3.14, warm cache, CPU
backend, and the repository `.venv`:

| Workload | Baseline pytest / wall | Final pytest / wall | Final peak RSS | Result |
|---|---:|---:|---:|---|
| collect full suite | 9.25 / 11.49 s | 2.61 / 3.83 s | 343 MB | 2,276 tests after 3 new tests |
| representative core | 1.20 / 2.00 s | 1.13 / 1.85 s | 236 MB | 11 passed |
| representative CLI/remote | 1.71 / 2.60 s | 1.25 / 2.04 s | 239 MB | 178 passed |
| representative apps | 0.21 / 0.91 s | 0.11 / 0.50 s | 88 MB | 27 passed |
| full suite | 97.57 / 100.22 s | 85.10 / 87.50 s | 1.34 GB | final: 2,237 passed, 39 skipped, 1 sandbox-only deselection |

Final domain partitions measured 945 passed + 39 skipped + one sandbox-only
deselection for core (52.27 / 53.97 s), 839 passed for CLI (27.11 / 28.05
s), 279 passed for apps (19.42 / 21.10 s), and 173 passed for packaging (4.66
/ 5.54 s). The additive integration sample ran 634 tests in 30.80 / 31.71 s.

The baseline full-suite failure was the known restricted-sandbox forkserver bind error:
`PermissionError: [Errno 1] Operation not permitted` in
`test_run_cases_engine_cpu_end_to_end_returns_finite_spectrum`. Collection and
focused final runs benefited from a hotter filesystem/import cache than the
baseline, so the observed deltas are not attributed to workspace metadata. No
speed or isolation claim follows from this report.

Outside the restricted sandbox, `marimo check` passed for all four apps and
`cxr app analysis --smoke` completed successfully. Strict Sphinx rendered the
new page but the repository-wide `-W` build remains red on 11 existing warnings
outside this change (including existing autosummary CLI invocation,
highlighting, and cross-reference warnings).
