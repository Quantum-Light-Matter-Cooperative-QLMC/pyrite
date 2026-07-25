# TODO — bugfix/rebrem-gpu-memory

## `rebrem` GPU-memory regression

**Problem.** `cxr remote rebrem --all --ne-brem 500 --step 20` leaks VRAM: remote
logs show CuPy reserved pool climbing after hundreds of records till card fills.
Cause: live sweep `_spectrum_case` releases pool per case (`_maybe_free_pool`,
`runner.py`), but `repair_brem_wide` loop (`run.py`) calls `_brem_for_case`
direct and never triggers the inter-case pool free — reserved pool grows +
fragments unbounded.

**Path.**
1. Failing cadence regression: assert `repair_brem_wide` fires the runner
   pool-free once per repaired record (monkeypatch `runner._GPU=True` +
   count `_maybe_free_pool`). Red without fix.
2. Bounded-memory fix: after each `_brem_for_case` in the loop, run the same
   guarded A2 cadence the live sweep uses (`if runner._GPU: runner._maybe_free_pool()`).
   Reuses existing `_should_free` / `_FREE_EVERY` / watermark — resumability
   (skip-at-target, save_cb, max_seconds deadline) untouched.
3. Remote benchmark: rerun `rebrem --all --ne-brem 500 --step 20`, confirm
   bounded reserved-pool watermark across hundreds of records.

**Scope.** `rebrem` only. `reline` (`repair_line_spec`) shares the same gap —
note as follow-up, do not widen here unless trivial parity.
