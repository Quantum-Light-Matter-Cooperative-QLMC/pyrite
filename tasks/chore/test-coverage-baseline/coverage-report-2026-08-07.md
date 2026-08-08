# Test-suite coverage baseline — 2026-08-07

First `--cov` run of the full suite. Snapshot only; no source changed.
`pytest-cov>=7.1.0` was already in the `test` dependency group but nothing in
the repo configured or invoked it.

## How it was run

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test --cov
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache NUMBA_DISABLE_JIT=1 uv run cxr-dev test --cov
```

Host: WSL2, RTX 3060 Ti present but **`cupy` not installed** in the dev env
(no `nvidia` extra). `CXR_ONLINE_TESTS` unset.

## Suite result

`2704 passed, 41 skipped, 0 failed`. 85.6 s default; 189.6 s with
`NUMBA_DISABLE_JIT=1`. Coverage instrumentation cost is negligible (~95 s in
the first line-only run vs. 85 s with branch coverage — noise).

## Headline numbers

| Measurement | Line coverage |
| --- | --- |
| Default run (branch coverage on) | **74.2%** (16 577 / 21 757 stmts) |
| `NUMBA_DISABLE_JIT=1` | **76.6%** |
| `NUMBA_DISABLE_JIT=1`, excluding `montecarlo/*_jit_kernel.py` | **84.0%** (17 102 / 20 370) |

1 096 partial branches of 6 902.

**Read the 74.2% as an artifact of the measuring environment, not as the real
gap.** Two compiled-code effects account for nearly the whole difference
between 74.2% and 84.0%:

1. **CuPy kernel modules are never imported.** The five
   `src/cxr_mc/montecarlo/*_jit_kernel.py` modules `import cupy` at module
   scope, so on a CPU-only environment every statement is missing — 1 387
   statements, 0.0% each (`coherent_stream` 391, `coherent` 322, `brem` 270,
   `spectrum` 265, `line_prologue` 139). Installing the `nvidia` extra would
   recover their host-side statements; `cupyx.jit` device-function bodies never
   execute under the Python tracer and stay unmeasurable in any environment.
2. **`@njit` bodies bypass the tracer.** Numba compiles them away, so exercised
   code reports as missing. Disabling the JIT proves the tests do reach it:

   | Module | default | `NUMBA_DISABLE_JIT=1` |
   | --- | --- | --- |
   | `montecarlo/groove.py` | 39.1% | **90.5%** |
   | `montecarlo/transport.py` | 37.1% | **86.5%** |
   | `montecarlo/geometry.py` | 60.4% | **86.1%** |

   Nothing else moved. `transport.py` looking like the worst-covered module in
   the tree is entirely this effect.

## Per-package (`NUMBA_DISABLE_JIT=1`)

| Package | Covered / stmts | % |
| --- | --- | --- |
| `montecarlo` | 2 316 / 4 215 | 54.9% (→ 82% excluding the CuPy kernel modules) |
| `plots` | 1 584 / 2 432 | 65.1% |
| `energy_grid` | 1 747 / 2 077 | 84.1% |
| `results` | 344 / 404 | 85.1% |
| `_remote` | 1 734 / 2 003 | 86.6% |
| `cxr_mc` top-level | 4 254 / 4 907 | 86.7% |
| `detectors` | 396 / 451 | 87.8% |
| `cli` | 3 463 / 3 867 | 89.6% |
| `materials` | 1 244 / 1 376 | 90.4% |

## Real gaps, ranked

Compiled-code artifacts removed; these are genuinely unexercised.

1. **`plots/` figure builders.** The largest true hole in the tree.
   `plots/detectors.py` 12.5% (221/260 missing), `plots/interactive.py` 6.3%
   (155/169), `plots/spectra.py` 45.8% (149/274), `plots/trajectories.py` 50.0%
   (129/284), `plots/sweeps.py` 63.8% (80/222). `tests/plots/test_exports.py`
   asserts the public names resolve to the right modules but never calls the
   figure builders, so the bodies are untouched. ~734 uncovered statements
   total — every one is user-facing output.
2. **`montecarlo/spectrum.py`** 74.3% (170/745) and **`montecarlo/runner.py`**
   72.3% (148/614). Mixed: some is GPU-branch dispatch that a CPU-only env
   cannot reach, some is scheduling/OOM-retry error handling. Needs a
   line-level split before it is actionable.
3. **`montecarlo/_backend.py`** 69.4% (48/177) — backend selection/fallback.
   The non-NVIDIA fallback branches the Active backlog item asks to
   "double check" are exactly what is unmeasured here.
4. **`energy_grid/apply.py`** 77.7% (104/542) and **`energy_grid/derive.py`**
   78.3% — largest gaps outside `montecarlo`/`plots` in shipped logic.
5. **`cli/_dashboard.py`** 81.1% (108/683) and **`_remote/lifecycle.py`** 80.5%
   (112/652) — terminal rendering and remote-session paths, both hard to drive
   from tests and both previously bug-prone.
6. **Entry shims at 0%**: `_compile_nb.py` (24), `_entry/scan.py` (5),
   `cli/energy_grid.py` (3), `cli/__main__.py` (2). Trivial; either smoke-test
   or accept.

## What this does not measure

- GPU execution paths (no `cupy` here) and `cupyx.jit` device bodies.
- `checks/` physics validation — outside `testpaths`.
- Online tests (`CXR_ONLINE_TESTS` unset; 41 skips include these).
- marimo notebooks/apps.

## Config landed alongside

`pyproject.toml` gained `[tool.coverage.run|paths|report]`: branch coverage,
`source = ["src/cxr_mc"]`, `parallel = true` for xdist, `skip_covered`,
`show_missing`, `precision = 1`, `exclude_also` for `TYPE_CHECKING` and
`__main__` guards, and no `omit` — with the two compiled-code caveats above
written into the comments so a future reader does not misread the total.
`uv run cxr-dev test --cov` now works with no extra flags.
