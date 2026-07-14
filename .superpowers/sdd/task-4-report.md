# Task 4: Finite spectrum escape-distance report

## Scope delivered

- Added `_segment_escape_distance(segments, n_hat, *, xp)`, using the shared
  rectangular-prism first-exit primitive for finite footprints and the existing
  z-only helper when both dimensions are absent.
- Kept the public `mc_spectrum` and `mc_brem_spectrum` signatures unchanged.
- Applied the finite distance to coherent attenuation, including capped
  `_stack_tau` for layered absorbers.
- Applied the finite distance to bremsstrahlung attenuation, including
  `_layer_path_length` per absorber layer in the finite branch.
- Added side-exit attenuation, layered residence, and finite chunk-invariance
  regressions. Physics docstrings now state six-face, fixed-far-field, and
  all-None-limit behavior and cite `finite-transverse-crystal`.

## TDD evidence

### RED: new helper contract

Command:

```text
uv run python scripts/dev.py test tests/test_spectrum_escape_helpers.py -v
```

Observed expected collection failure:

```text
ImportError: cannot import name '_segment_escape_distance'
```

### GREEN: helper implementation

The minimal helper was added around `first_prism_exit` and then the same command
passed:

```text
collected 4 items
tests/test_spectrum_escape_helpers.py ....
4 passed in 1.04s
```

### RED: finite self-absorption behavior

After adding the finite-width spectrum regression, before the spectrum branches
were implemented:

```text
collected 5 items
FAILED test_finite_side_exit_shortens_coherent_and_brem_self_absorption
assert 6.285764642030211e-10 > 6.285764642030211e-10
1 failed, 4 passed
```

This demonstrates the old code ignored stored transverse dimensions and produced
identical coherent spectra for narrow and wide crystals.

### GREEN: spectrum branches and regressions

After adding finite coherent/brem branches, the behavior suite passed:

```text
collected 5 items
tests/test_spectrum_escape_helpers.py .....
5 passed in 1.24s
```

The layered finite-path regression was added afterward as an additional guard:

```text
collected 6 items
tests/test_spectrum_escape_helpers.py ......
6 passed in 1.43s
```

The chunk-invariance regressions exercise `chunk=1` and one-shot execution with
the same finite side-exit fixture. They are an invariant property rather than a
new API, so the behavior RED above is the primary implementation red test.

## Focused verification

```text
uv run python scripts/dev.py test \
  tests/test_spectrum_escape_helpers.py \
  tests/test_chunk_invariance.py \
  tests/test_multilayer.py -v

collected 40 items
40 passed in 8.00s
```

## Full verification and review

```text
uv run python scripts/dev.py lint
All checks passed!

uv run python scripts/dev.py typecheck
BLOCKED by two pre-existing optional-dependency errors:
src/cxr_mc/plots/interactive.py:82,110 -- Import "ipywidgets" could not be resolved.

uv run python -m pyright src/cxr_mc/montecarlo/spectrum.py
0 errors, 0 warnings, 0 informations

uv run python scripts/dev.py test -v
collected 540 items; command completed successfully with no failure traceback.

uv run python scripts/dev.py test -q
command completed successfully with no failure traceback.

git diff --check
no output (clean)
```

Self-review confirmed the legacy all-None coherent and brem z-only branches
retain their former calculations, finite no-layer paths use `L_esc * mu`, finite
coherent layers delegate to capped `_stack_tau`, and finite brem layers compute
each capped ray/layer residence directly. The focused suite includes an exact
array-equality regression between omitted and explicitly all-None dimensions.

## Concerns

- The full-suite tool output was progress-truncated by the command runner, but
  both full-suite commands completed without a subprocess error or pytest
  failure traceback.
- Repository-wide Pyright is blocked by the two unresolved optional
  `ipywidgets` imports in `plots/interactive.py`; the touched production module
  passes direct Pyright validation with zero errors.
- This task intentionally does not change public APIs, sweeps, transport, or
  validation-ledger documents; those belong to adjacent plan tasks.

## Review-finding fix: pure lateral finite bremsstrahlung escape

### RED

Added `test_finite_brem_pure_lateral_escape_has_no_divide_by_zero_warning`,
which uses a finite footprint and `n_hat=[1, 0, 0]`.  Warnings are treated as
errors, and the test requires a finite, positive bremsstrahlung spectrum.

```text
uv run python scripts/dev.py test tests/test_spectrum_escape_helpers.py -k pure_lateral -v

FAILED test_finite_brem_pure_lateral_escape_has_no_divide_by_zero_warning
RuntimeWarning: divide by zero encountered in divide
src/cxr_mc/montecarlo/spectrum.py:591: in mc_brem_spectrum
    L_esc = _escape_length(z_mid, thickness, n_hat[2])
```

### GREEN

`mc_brem_spectrum` now evaluates `_segment_escape_distance` directly for a
finite footprint.  Only the unchanged legacy no-footprint branch evaluates
the z-only `_escape_length`, eliminating the invalid `n_z=0` division.

```text
uv run python scripts/dev.py test tests/test_spectrum_escape_helpers.py -v
8 passed in 1.01s

uv run python scripts/dev.py test \
  tests/test_spectrum_escape_helpers.py \
  tests/test_chunk_invariance.py \
  tests/test_multilayer.py -v
41 passed in 7.58s
```
