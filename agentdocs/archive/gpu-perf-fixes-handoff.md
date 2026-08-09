# GPU perf fixes handoff (2026-07-28)

Archived historical handoff; do not treat it as current instructions.

Landed the 5 fixes from the perf review (see convo), uncommitted, on `main`.

## What changed

1. **fp32 chunk-budget bug** — `montecarlo/runner.py:44`: `_REAL_BYTES =
   np.dtype(REAL).itemsize` replaces the hardcoded 8 in `_adaptive_chunk`.
   GPU (fp32) chunks double; CPU (fp64) path bit-for-bit unchanged.
2. **parallel_materials vs single GPU** —
   - `_remote/config.py`: `DEFAULT_PARALLEL_MATERIALS` 2 -> 1.
   - `montecarlo/runner.py`: `_GPU_POOL_SHARE` (env `CXR_MC_GPU_SHARE`,
     default 1); `_ensure_pool_limit` caps at `_GPU_POOL_FRAC / _GPU_POOL_SHARE`
     so N co-tenant scans sum to 0.85 VRAM, not N*0.85.
   - `_remote/scripts.py::_queue_script` exports `CXR_MC_GPU_SHARE=$parallel_materials`.
3. **_FREE_EVERY default 1 -> 8** (`runner.py`): pool cap + OOM-retry made the
   per-case `free_all_blocks()` non-load-bearing. `CXR_MC_FREE_EVERY=1` = old
   cadence.
4. **Skip `uv sync` on resubmit slices** — `_remote/scripts.py::_uv_sync_block(once=True)`
   guards the sync behind `$JOBDIR/.synced`; applied at all 7 call sites.
   Delete the sentinel to force re-sync mid-chain.
5. **Host-RAM retention** — `run_cases(..., keep_results=False)` nulls
   `results[i]` post-callback; `run.py::run_sweep` passes it (callback owns
   storage). Default True = old behaviour for every other caller.

## Test edits (interface follow-ons)

- `tests/test_run.py`: run_cases fakes/stubs accept `keep_results`.
- `tests/remote/test_remote.py`: default assertions 2 -> 1; new assert on the
  `CXR_MC_GPU_SHARE` export.

## Dropped

Unrelated `cli/sweep.py` line-grid reconcile change (stale stash pop,
pre-CLI-refactor) was re-stashed: `git stash list` -> "stale stash pop:
sweep.py line-grid reconcile (pre-CLI-refactor)". Recover with `git stash pop`
if still wanted.

## Not done (from the review, by choice)

- Caching question (chi_g/U_g, E_tab union grid, Henke loads per material):
  investigated as small-win; not implemented.
- Fused `sinc(x)**2` elementwise kernel, CUDA streams, pinned H2D, out-of-order
  completion, shared-memory transport handoff: left as-is (clarity/complexity
  tradeoffs noted in the review).
- `_halve_case_chunks` halves spec+brem together on OOM: harmless, kept.

## Verify

`uv run python scripts/dev.py test` (offline suite).
