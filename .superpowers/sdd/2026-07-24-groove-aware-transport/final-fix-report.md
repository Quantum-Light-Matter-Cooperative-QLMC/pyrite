# Groove-Aware Transport Final Fix Report

## Scope

Consolidated final-review corrections for commit base `dc8f52b`.

Scoped implementation and regression files:

- `src/cxr_mc/montecarlo/groove.py`
- `src/cxr_mc/montecarlo/transport.py`
- `src/cxr_mc/montecarlo/spectrum.py`
- `src/cxr_mc/plots/trajectories.py`
- `tests/test_groove.py`
- `tests/test_trajectories.py`
- `docs/physics-validation-ledger.md`
- `docs/superpowers/plans/2026-07-24-groove-aware-transport.md`
- `docs/superpowers/specs/2026-07-24-groove-aware-transport-design.md`
- `docs/validation/blazed-groove-geometry.md`

The independent verifier owns
`docs/validation/blazed-groove-geometry.md`; implementation work did not edit
its content. After remote GREEN, the verifier re-checked both former
discrepancies and updated that document to verdict `rederived`. Per the
approved plan, the ledger remains `unverified`; no suggested status transition
was applied. Unrelated dirty remote, run, agent-guidance, issue-note,
`.gitignore`, and retired flat-report changes remain excluded.

## RED Evidence

Tests were written or corrected before production changes, then executed on
qlmc by the controller:

```text
3 failed, 3 passed, 28 deselected in 3.87s
```

Expected failures:

1. Analytic valley crossing returned `inf` instead of `1.0 Ang`.
2. Valid groove exit/re-entry consumed `max_steps`, leaving
   `n_transmitted == 0`.
3. Perpetual valid surface events exhausted the outer loop without raising.

Fast reference marching, tangent tolerance, and Matplotlib collection
contracts already passed against pre-fix production.

## Corrections

### Transport event limits

Ungrooved transport retains the legacy lockstep path. Grooved transport now
tracks material iterations and valid surface re-entries separately per
electron. Re-entry repeats event processing without consuming the material
`max_steps` budget. A separate bounded groove-event counter raises
`RuntimeError("grooved surface event limit exhausted")` rather than returning
a live event-limit survivor as stopped. The existing two-event zero-length
guard remains.

`zero_surface_events` is narrowed to a non-optional local inside the groove
branch, resolving the reported static-type diagnostics.

### Geometry endpoint

`first_surface_event` accepts only a geometry-scaled floating-point band:

```text
band_tol = 32 * eps_float64 * max(spacing, depth)
```

This admits an analytic apex or valley displaced by a few ULP during plane
arithmetic while continuing to reject intersections farther outside
`[0, depth]`. The analytic `Lambda=2 Ang`, `tp=0.61` vertical valley crossing
now has a focused regression.

### Independent references and tangency

The expensive escape reference now brackets the first material-state change
with a period-scaled step and bisects it for 48 iterations. It remains
independent of production geometry helpers. The fixture uses a
dimensionally equivalent small-period groove. Tolerance
`5e-11 * spacing` plus `5e-11` relative covers modulo/facet floating-point
conditioning, not a spatial march step.

The analytic tangent precondition uses an eight-ULP absolute dot-product
tolerance. Existing tiny-nonzero vertical-direction and facet-rate tests remain
unchanged, so near-tangent physical events are not erased.

### Matplotlib and typing

Vacuum legs remain one separate `LineCollection`, with `alpha=0.35` and
per-segment energy values. Material track arrays are unchanged by rendering.
Passing `vacuum_segments.tolist()` satisfies Matplotlib's typed
`Sequence[ArrayLike]` boundary.

### Finite photon edge scaling

Specification, ledger, and spectrum documentation now state:

```text
Delta x = L_esc cos(tp) = z cot(tp) + O(Lambda)
f_edge(z) = min(z cot(tp) / W + O(Lambda / W), 1)
```

`O(Lambda/W)` is only the periodic-phase correction, not the total finite-side
error in general. Ledger status remains `unverified`; no signed-off or
discrepancy transition was made. Ledger and specification name implemented
helpers exactly:
`blazed_groove_spec`, `surface_depth_ang`, `in_material`,
`first_surface_event`, `escape_distance_ang`, and `entry_points`.

## GREEN Evidence

All verification ran remotely on qlmc, per user instruction.

Focused:

```text
34 passed in 18.59s
```

Broader transport, spectrum, runner, and trajectory plots:

```text
192 passed in 34.75s
```

Typecheck:

```text
All checks passed!
```

Scoped Ruff:

```text
All checks passed!
```

Local `git diff --check` also passed. No local tests, lint, or typecheck ran.

## Self-Review

- Flat `groove=None` branch preserves existing event order and RNG draws.
- Valid re-entry increments only the separate surface counter.
- Permanent entrance exit and finite-footprint side exit retain existing
  counters.
- Vacuum legs still preserve energy and direction, advance only the clock, and
  never enter material segment arrays.
- Event bound is per electron, preventing one pathological ray from consuming
  another electron's allowance.
- Endpoint tolerance scales with geometry and does not reuse the larger
  transition-probing epsilon as a physical band extension.
- Tiny nonzero facet rates remain production-visible.
- Matplotlib regression inspects real collection state, not a mock.
- Independent reference derives material membership from `_z_surf`, not
  production helpers.
- Verifier document remained implementation-author untouched; independent
  re-review resolved both discrepancies and recorded verdict `rederived`.
- Ledger remains `unverified` per the approved plan despite the verifier's
  suggested transition.
- Unrelated working-tree changes remain unstaged and outside commit scope.
