# Test coverage baseline and gap closure

## Problem and scope

`pytest-cov` has been a `test`-group dependency with no config and no caller.
The first full-suite `--cov` run (2026-08-07, see
[`coverage-report-2026-08-07.md`](coverage-report-2026-08-07.md)) reports
74.2% line / 6 902 branches, but 84.0% once the two compiled-code measurement
artifacts are removed. Suite itself is green: 2704 passed, 41 skipped.

In scope: coverage plumbing (config, a documented way to run it, optional
verify-gate wiring) and closing the *real* gaps the baseline exposes, largest
first — `plots/` figure builders, backend-fallback dispatch, `energy_grid`
apply/derive.

Out of scope: chasing the headline percentage. Anything that only moves the
number by hiding compiled modules (`omit` on `*_jit_kernel.py`) is rejected —
the caveats belong in comments and docs, not in an `omit` list. No physics
changes; no coverage threshold that would make CI fail on GPU-env differences
without a plan for the two-environment split.

## Baseline facts to build on

- `montecarlo/*_jit_kernel.py` (5 modules, 1 387 stmts) import `cupy` at module
  scope → 0.0% on any CPU-only environment. This box has an RTX 3060 Ti but no
  `cupy` installed.
- `@njit` bodies bypass the tracer. `NUMBA_DISABLE_JIT=1` moves
  `transport.py` 37.1→86.5%, `groove.py` 39.1→90.5%, `geometry.py`
  60.4→86.1%, at ~2.2x wall clock (85.6 s → 189.6 s).
- Genuinely untested, ranked: `plots/` builders (~734 stmts across five
  modules, worst `plots/interactive.py` at 6.3%), `montecarlo/_backend.py`
  69.4% fallback dispatch, `energy_grid/apply.py` 77.7%, `cli/_dashboard.py`
  81.1%, `_remote/lifecycle.py` 80.5%, four 0% entry shims (34 stmts).

## Implementation path

`pyproject.toml` `[tool.coverage.*]` (landed on `main` with the baseline
commit). Remaining work is `src/cxr_mc/_dev.py` (a coverage entry point) plus
new tests under `tests/plots/`, `tests/montecarlo/`, `tests/energy-grid/`.

Likely owners: `_dev.py` `cmd_test`/`cmd_verify`; `tests/plots/test_exports.py`
as the model for what *not* to stop at (it asserts module identity only).

## Checklist

- [x] A — Run full suite under `--cov`, both default and `NUMBA_DISABLE_JIT=1`;
      write the baseline report.
- [x] B — Add `[tool.coverage.run|paths|report]` to `pyproject.toml`, caveats
      in comments.
- [ ] C — `cxr-dev coverage` (or `test --coverage`) wrapping the documented
      invocation, including a `--numba` switch that sets `NUMBA_DISABLE_JIT=1`.
      Document in `AGENTS.md` canonical commands and `docs/`.
- [ ] D — `plots/` smoke tests: call every public figure builder on a small
      synthetic result with the Agg backend, assert a `Figure` and non-empty
      axes. Target `plots/` ≥ 85%.
- [ ] E — `montecarlo/_backend.py` fallback-dispatch tests (non-NVIDIA path);
      coordinate with the Active "Compute performance optimization" item, which
      already asks for exactly this confirmation.
- [ ] F — `energy_grid/apply.py` + `derive.py` gap review; add tests or record
      why a branch is unreachable.
- [ ] G — Decide the entry-shim policy (`_compile_nb.py`, `_entry/scan.py`,
      `cli/energy_grid.py`, `cli/__main__.py`): smoke-test or `pragma: no
      cover` with a reason.
- [ ] H — Re-measure; update the report with the after numbers. Decide then,
      not now, whether a `fail_under` belongs in `verify`.

## Decisions and open questions

- **Decided:** no `omit` for compiled modules; document instead.
- **Open:** does a coverage run belong in `cxr-dev verify`? It costs ~2x if it
  implies `NUMBA_DISABLE_JIT=1`. Leaning no — keep it an explicit command until
  slices D–F land.
- **Open:** is a GPU-environment coverage run (with the `nvidia` extra) worth
  standing up, given `cupyx.jit` bodies stay unmeasurable either way? Only the
  host-side wrappers would be recovered.
- **Open:** `fail_under` threshold, and whether it is per-package rather than
  global — a global floor is hostage to which extras the runner installed.

## Delegation slices and required skills

- C → `implement-task-lite`; `cli-ui-ux` for the `cxr-dev` surface,
  `documentation-maintenance` for `AGENTS.md`/`docs`.
- D → `implement-task`; `regression-testing`.
- E → `implement-task`; `monte-carlo` + `performance`; check against the Active
  compute-optimization item before starting.
- F, G → `implement-task-lite`; `regression-testing`.
- H → whoever closes the last of D–F.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test --cov
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache NUMBA_DISABLE_JIT=1 uv run cxr-dev test --cov
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
```

- Suite stays green; no new skips.
- `plots/` ≥ 85% and `montecarlo/_backend.py` ≥ 90% under
  `NUMBA_DISABLE_JIT=1`.
- No `omit` entry added for `*_jit_kernel.py`.
- Report refreshed with post-change numbers and the same environment caveats.
