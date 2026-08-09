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
- [x] C — `cxr-dev coverage` (or `test --coverage`) wrapping the documented
      invocation, including a `--numba` switch that sets `NUMBA_DISABLE_JIT=1`.
      Document in `AGENTS.md` canonical commands and `docs/`.
- [x] D — `plots/` smoke tests: call every public figure builder on a small
      synthetic result with the Agg backend, assert a `Figure` and non-empty
      axes. Target `plots/` ≥ 85%. Landed: `tests/plots/test_detectors.py`,
      `test_interactive.py`, `test_trajectory_builders.py`. `plots/` now 90.4%
      aggregate (`detectors.py` 12.5%→94.2%, `interactive.py` 6.3%→94.6%,
      `trajectories.py` 50.0%→94.8%, plus incidental gains to `spectra.py`
      68.4% and `sweeps.py` 81.2% from the shared drawers). Remaining weak
      spot: `render_trajectories.py` 50.0% (untouched — its gap is
      `render_reveal_animation`'s missing-dependency error path, not a figure
      builder; out of D's scope). 2831 passed, 57 skipped, no new skips.
- [x] E — `montecarlo/_backend.py` fallback-dispatch tests (non-NVIDIA path).
      CPU fakes now cover CuPy CUDA/ROCm probing and allocator behavior, SYCL
      device selection/queue failures, loader error paths, backend selection,
      and FP64 fallback; focused Numba-disabled coverage is 98.1%. The active
      compute-performance remainder only needs CUDA-box profiling/CLI work, so
      this CPU-only contract coverage does not overlap its remaining ownership.
- [x] F — `energy_grid/apply.py` + `derive.py` gap review. Added regression
      coverage for both artifact setters' catalog/provenance rollback on a
      stamping failure. The remaining uncovered paths are CLI rendering,
      filesystem cleanup failures, and real Monte Carlo execution; the latter
      is deliberately replaced by deterministic runner fakes in the CPU suite.
- [x] G — Entry-shim policy: smoke-test the installed/module compatibility
      surfaces (`_entry/scan.py`, `cli/energy_grid.py`, `cli/__main__.py`).
      Mark `_compile_nb.py` no-cover: it is a legacy developer script whose
      hard-coded notebook inventory no longer exists, not a supported runtime
      entry point.
- [x] H — Re-measure; update the report with the after numbers. Decide then,
      not now, whether a `fail_under` belongs in `verify`.
      The unrestricted default run is green: 2,854 passed, 57 skipped, 75.9%
      total. The completed Numba-disabled run is also green: 2,854 passed, 57
      skipped, 78.9% total / 81.0%
      statements (18,191 / 22,457 statements; 7,108 branches). Its former 59%
      stall was the 48,000-electron aggregate comparison in
      `test_transport_per_electron.py`; coverage mode now retains all six
      observables with the existing 120-electron minimal case and four fixed
      seeds, while normal runs retain 3,000 electrons and eight seeds. Focused
      coverage runtime is 9.85 s. No `fail_under` or `verify` wiring: keep both
      coverage environments explicit until GPU/JIT-dependent totals have a
      stable two-environment policy.

## Decisions and open questions

- **Decided:** no `omit` for compiled modules; document instead.
- **Decided:** coverage stays an explicit `cxr-dev test --cov` workflow rather
  than part of `cxr-dev verify`; the Numba-disabled measurement is materially
  slower and measures a different execution mode.
- **Open:** is a GPU-environment coverage run (with the `nvidia` extra) worth
  standing up, given `cupyx.jit` bodies stay unmeasurable either way? Only the
  host-side wrappers would be recovered.
- **Decided:** no `fail_under` yet. A global floor is hostage to installed GPU
  extras and JIT tracing; package floors need a defined CPU/GPU split first.

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
