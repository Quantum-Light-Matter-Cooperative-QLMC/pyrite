# Correct line-absorption tabulation

## Problem and scope

`line-absorption-tabulation` linearly interpolates the compound attenuation
coefficient `mu(E)` on the shared line table.  This preserves units and
Beer--Lambert signs but disagrees with the pinned xraydb 4.5.8 Chantler
interpolant at absorption edges: HOPG is high by 27.2% at 283.7351 eV and the
relative transmission error reaches 52.3% for a 10000 Angstrom escape path.
The discrepancy is catalog-wide and has no regression anchor.

The owning correction is the single-slab and finite-footprint line path.  Keep
layered and grooved paths on their existing exact per-point evaluation.  Do not
change the Beer--Lambert equation, scattering amplitudes, resonance kinematics,
catalog data, or detector attenuation.

## Implementation path and owners

- `src/pyrite/montecarlo/spectrum/lines.py`: build per-element attenuation
  tables, include explicit absorber-composition elements in the edge-resolved
  grid, and interpolate each elemental contribution in log(mu) versus log(E)
  before summing.  Use a float32-stable log-energy fraction such as `log1p`;
  avoid cancellation between closely spaced edge nodes.
- `src/pyrite/montecarlo/spectrum/coherent_stream_jit_kernel.py`: keep the
  streamed coherent route numerically consistent with the array routes.
- `tests/montecarlo/`: add a fast CPU regression derived independently from
  xraydb's pinned interpolation rule.  Cover elemental HOPG, compound MoS2,
  an explicit absorber element absent from the crystal basis, endpoint/limit
  behavior, and backend-appropriate tolerances.
- `docs/validation/radiation-physics/line-absorption-tabulation.md`: record the
  source behavior, derivation, units, limits, numeric comparison, and remaining
  evidence.
- `docs/validation/physics-validation-ledger.md`: retain `discrepancy` until a
  fresh-context verifier confirms the change; update checks, anchor, and notes.

The key derivation is elemental: xraydb interpolates non-`f1` Chantler data
linearly in `log(f2)` versus `log(E)`, and `mu_i` is proportional to `f2_i/E`,
so `log(mu_i)` is linear on the same intervals.  The compound coefficient is
`sum_i mu_i`; interpolating `log(sum_i mu_i)` is not equivalent and is out of
scope as a fix.

## Checklist

- [ ] Add a failing edge-focused regression before changing production code.
- [ ] Implement stable per-element log-log attenuation interpolation in every
      tabulated line route without restoring per-segment CPU/device transfers.
- [ ] Include all absorber-composition elements when constructing the native
      edge grid and preserve endpoint clamping and out-of-range policy.
- [ ] Verify units, positivity, zero-path transmission, elemental and compound
      limits, and exact-node identity.
- [ ] Measure identical CPU workloads before/after; record table size, warm-up,
      repeats, spread, and any unavailable CUDA evidence.
- [ ] Add the derivation record and update the ledger without marking it
      `signed-off`.
- [ ] Run focused line/interpolation tests, neighboring Monte Carlo tests,
      lint/typecheck, and docs proportional to the touched paths.
- [ ] Review the scoped diff and checkpoint independently valid work.

## Decisions and open questions

- Decision: interpolate each elemental contribution, then sum.  Do not accept
  the current edge error and do not interpolate the compound total in log
  space; both leave the physical accuracy contract unresolved.
- Decision: use the repository-pinned xraydb implementation and native
  Chantler nodes as the source behavior.  No external data-source migration.
- Decision: target float64 agreement at rounding level and float32 relative
  error no worse than `1e-4` at adversarial edge midpoints.  The tolerance is
  set by backend precision and the measured stable formulation, not the old
  discrepancy.
- Open evidence: on-device performance and CUDA numerical behavior require an
  available CUDA runner.  Do not launch a remote job without separate
  authority; absence is reportable and does not justify weakening CPU physics
  coverage.

## Delegation

One lead worker owns the integrated milestone because the table representation
crosses array and streamed kernels, physics validation, regression design, and
performance evidence.  Required skills: `lead-task`, `physics-review`,
`physics-validation`, `regression-testing`, `performance`, and
`documentation-maintenance`.  The reviewed task is self-contained enough for
a Serena one-shot contract.  No sub-delegation is authorized.

## Acceptance checks

- The new regression fails against the pre-fix linear-total-mu method and
  passes against the corrected method.
- HOPG, MoS2, and MoSe2 adversarial midpoint maxima satisfy the stated dtype
  tolerances; an explicit non-basis absorber uses its own native edge nodes.
- Single-slab and finite-footprint coherent/incoherent routes share the
  corrected method; layered and grooved exact paths remain unchanged.
- Focused and neighboring CPU suites pass.  CUDA checks pass when locally
  available or are explicitly reported as unrun.
- The performance report demonstrates that the fix does not restore the
  eliminated per-segment CPU/device synchronization.
- The derivation record and ledger identify the exact old divergent convention,
  the corrected expression, assumptions, limits, anchor, and human sign-off
  requirement.

