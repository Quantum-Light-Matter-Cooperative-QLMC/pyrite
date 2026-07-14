# Task 5: finite footprint sweep and runner report

## Scope delivered

- Added optional `Sweep.crystal_width_mm` and `Sweep.crystal_height_mm` fields
  with the existing `ScalarOrSeq` convention.
- `build_cases()` validates the paired dimensions, takes their Cartesian
  product, writes both public mm keys into every case, and appends
  `footprint=<width>x<height>mm` only for finite cases.
- Omitted dimensions preserve the legacy case name exactly and store both keys
  as `None`.
- `geometry_table()` now exposes `width [mm]` and `height [mm]`.
- `_transport_case()` forwards the values to both line and brem trajectory
  calls; `_brem_for_case()` forwards them to repair-path transport. `run_case()`
  documents the optional case keys and their all-None limit.

## TDD evidence

### RED

```text
uv run python scripts/dev.py test tests/test_sweep.py tests/test_run.py \
  -k 'footprint or transport_case or brem_for_case' -v
```

Before implementation, all seven selected tests failed for the intended
missing behavior: `Sweep.__init__` rejected the new footprint keys and the
runner spies received no `crystal_width_mm`/`crystal_height_mm` keyword.

### GREEN

After the minimal sweep expansion and runner forwarding changes, the same
focused command passed: **7 passed**. A legacy-name/all-None regression was
then added, and the focused command passed with **8 passed**.

## Independent review follow-up

The reviewer found that the initial sweep validation accepted positive
infinities, producing a finite-named case that downstream transport later
rejected. Added failing `+inf` width and height regressions, then changed the
array validation to require both `np.isfinite(...)` and `> 0.0`. The focused
Task 5 suite then passed with **10 passed**; the full sweep/run suite passed
with **56 passed**.

## Verification

- `uv run python scripts/dev.py test tests/test_sweep.py tests/test_run.py -v`:
  **56 passed**.
- Finite-crystal integration batch (`test_finite_transverse_geometry`,
  `test_montecarlo`, `test_spectrum_escape_helpers`, `test_chunk_invariance`,
  `test_multilayer`, `test_montecarlo_exports`): **70 passed**.
- `uv run python scripts/dev.py test -q`: completed successfully with no pytest
  failure traceback.
- `uv run python scripts/dev.py lint`: passed.
- Scoped Pyright on `src/cxr_mc/sweep.py` and
  `src/cxr_mc/montecarlo/runner.py`: **0 errors, 0 warnings**.
- Ruff format check and `git diff --check`: passed.

## Self-review and concerns

- The all-None branch deliberately uses one `(None, None)` footprint pair, so
  legacy names and Cartesian ordering remain unchanged except for the new
  explicit `None` case fields.
- A finite footprint is appended after any substrate suffix, making
  `footprint=...mm` the terminal, checkpoint-distinguishing suffix.
- No checkpoint grouping or identity logic changed beyond those finite names.
- No outstanding concerns in Task 5 scope.
