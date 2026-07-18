# Regime-split scheduling for `run_cases` (2026-07-18)

## Problem

`analyze_line_grid_bounds.py` runs two workload regimes per beam energy:

- **coarse** (`COARSE_NE=500`, ~980 cases) — selects the top-5 candidate geometries.
- **refine** (`REFINE_NE=5000`, top-5) — produces the reported max-`E_line_grid` numbers.

`run_cases` (`src/cxr_mc/montecarlo/runner.py`) picks its execution branch from
*hardware alone*: `if _GPU` it runs the GPU-pipeline branch (CPU transport farmed
to a worker pool of `ncpu//2`, spectra drained **in strict input order** and
computed **serially in the main-process CUDA context**); only when no GPU is
present does it use the full-case CPU pool (`ncpu*3//4` workers, `as_completed`,
CPU spectrum).

On the `qlmc` box (GPU present) the coarse regime is therefore forced onto the
GPU pipeline even though at ne=500 the spectrum is trivially cheap (494 MiB,
2–4% GPU utilisation — see `docs/superpowers/plans/2026-07-18-line-grid-gpu-throughput-findings.md`).
Coarse is **structurally CPU-bound** yet runs with only half the cores and a
strict-order drain that stalls the pipeline behind any slow transport
(head-of-line blocking). Coarse dominates wall-clock.

## Goal

Let the *caller* choose the execution branch so the CPU-bound coarse regime runs
the full-case CPU pool (more workers, `as_completed`, no main-process
serialisation) while the refine regime keeps the GPU pipeline. Refine produces
the reported numbers and its behaviour must not change.

## Non-goals

- No change to transport or spectrum physics.
- No Numba (tracked as the follow-up prototype; see "Numba seam" below).
- No spectrum-batching-across-cases or CUDA-stream double-buffering (larger,
  separate GPU-side tasks).

## Design

### 1. `engine` selector on `run_cases`

Add `engine="auto"` to
`run_cases(cases, max_workers=None, progress=True, callback=None, should_stop=None, engine="auto")`:

| `engine` | Branch |
| --- | --- |
| `"auto"` (default) | Today's behaviour exactly: GPU pipeline if `_GPU`, else CPU pool. |
| `"gpu"` | Force the GPU-pipeline branch. If no GPU is present, warn and fall back to the CPU pool. |
| `"cpu"` | Force the full-case CPU pool **even when `_GPU` is true**. |

Branch selection becomes:

```python
if engine not in ("auto", "gpu", "cpu"):
    raise ValueError(...)
use_gpu = _GPU if engine == "auto" else (engine == "gpu")
if engine == "gpu" and not _GPU:
    warnings.warn("engine='gpu' requested but no GPU present; using CPU pool")
    use_gpu = False
```

Everything currently guarded by `if _GPU:` becomes `if use_gpu:`. **Default
`"auto"` preserves every existing caller and golden test bit-for-bit.**

### 2. Force CPU spectrum inside workers (the one real wrinkle)

The CPU-pool branch runs `run_case` → `_spectrum_case`, which reads the
module-global `_GPU`. In a worker process on a GPU box cupy imports fine, so
`_GPU` is `True` and each worker would initialise its own CUDA context — the
exact contention the single-context design avoids.

Fix: `_worker_init` gains a `force_cpu=False` parameter that, when true, sets the
worker module's `_GPU = False` (and any derived `xp`/flags) so `_spectrum_case`
takes the NumPy path. The `engine="cpu"` branch constructs its pool with
`initializer=_worker_init, initargs=(True,)`. No env-var global state, no
per-case flag plumbing — the pool is born CPU-only.

**Verification item (implementer must confirm, not assume):** `_spectrum_case` /
`mc_spectrum` / the brem path have a working NumPy code path when cupy is
*installed but forced off*. The `xp` abstraction should cover it; check by
running one `engine="cpu"` case with `_GPU` monkeypatched true and asserting a
finite spectrum comes back. If any code path hard-imports cupy unconditionally,
fix it to route through the existing `xp` selector.

### 3. Script wiring — `analyze_line_grid_bounds.py`

- Thread an `engine` argument through `_run_specs` and `_scan`.
- Coarse calls (near-zero scan + spot-check, lines ~194/197) pass
  `engine=<coarse_engine>`.
- Refine call (~209) passes `engine="auto"` (GPU on the box; unchanged).
- Add CLI flag `--coarse-engine {cpu,auto}` (default `cpu`) so the coarse-CPU
  route can be A/B'd against the old GPU-pipeline path in a single command.

### 4. Reproducibility & validation

- `engine="auto"` default → catalog path, golden-fingerprint tests, and every
  other caller are untouched, bit-for-bit.
- Only the coarse **diagnostic** scan moves to CPU spectrum. CPU vs GPU spectra
  are not bit-identical (float summation order), so the coarse top-5 candidate
  set can shuffle at the margin. This is acceptable: the **refine** phase (still
  GPU, unchanged) produces the reported max-`E_line_grid` numbers, and the whole
  pipeline is validated against literature via `cxr check`.
- Implementer confirms no golden-fingerprint test pins the coarse diagnostic
  output. If one does, surface it before proceeding.

### 5. Testing (must run GPU-free in CI)

CI has no GPU (`_GPU` is false), so tests exercise the *dispatch logic* by
monkeypatching, not a real device:

- `engine="auto"` with `_GPU` monkeypatched false → CPU pool (current behaviour).
- `engine="cpu"` with `_GPU` monkeypatched true → CPU-pool branch selected, and
  workers run CPU spectrum (assert via a spy on `_worker_init` / that no CUDA
  context is requested). At minimum assert the CPU-pool code path is taken.
- `engine="gpu"` with `_GPU` false → emits the fallback warning and runs the CPU
  pool.
- `engine="bogus"` → `ValueError`.
- A small end-to-end `engine="cpu"` run returns finite spectra (covers the
  §2 NumPy-path verification).

### 6. Files touched

- `src/cxr_mc/montecarlo/runner.py` — `run_cases` signature + branch selection;
  `_worker_init` `force_cpu` param; CPU-pool branch `initargs`.
- `scripts/analyze_line_grid_bounds.py` — thread `engine`; `--coarse-engine`.
- `tests/` — dispatch-logic tests above (locate the existing runner test module
  via repo-map; follow its patterns).

## Success criteria

- `engine="auto"` golden/fingerprint tests unchanged.
- New dispatch tests pass GPU-free in CI.
- `cxr check` passes with coarse on CPU (literature validation).
- Coarse wall-clock drops materially vs the GPU pipeline — **measured on `qlmc`
  when it is back online; deferred verification**, noted here so it is not
  mistaken for done.

## Numba seam (follow-up, not built here)

#1 isolates transport (`_transport_case`, pure NumPy, farmed to the pool) from
scheduling. The Numba prototype later swaps only the transport kernel — the
CPU-pool route already carries its segments to the CPU spectrum, so no further
scheduling change is needed. Numba breaks bit-for-bit reproducibility by design;
it is validated against literature via `cxr check`, which is why it is a
separate, explicitly-validated phase rather than folded in here.
