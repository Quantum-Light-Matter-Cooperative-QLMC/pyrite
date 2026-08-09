# Energy-controlled electron transport

## Problem and scope

The current transport freezes energy, speed, stopping power, elastic hazard,
and radiation kinematics at the start of each physical flight. This is a
left-endpoint rule with no convergence control; the exponential flight tail can
produce large energy/hazard changes and coherent clock error.

Source: [`docs/electron_transport_physics_recommendations.docx`](../../../../docs/electron_transport_physics_recommendations.docx), Stages 1 and 3.

This task introduces a controlled propagation model while keeping physical
flights distinct from numerical energy-integration substeps. It is gated on
`fix/electron-transport-correctness`, which establishes cutoff, timing, and
termination semantics first. Reference elastic/stopping tables are separate
tasks and may land before or after this one through a stable model interface.

In scope:

- `E_start`, `E_end`, representative energy, and midpoint/integrated time;
- midpoint predictor-corrector stopping and clock integration;
- an energy-loss step limit, initially targeting `|Delta E| / E <= 1%`;
- elastic optical-depth integration or a convergent piecewise-constant hazard;
- `electron_id`, physical `flight_id`, and numerical `substep_id` identity;
- CXR field combination within a physical flight and bremsstrahlung
  quadrature without artificial radiation/decoherence from substep count; and
- transport/radiation error diagnostics and convergence tests.

Out of scope: changing the underlying Browning/NIST angular law, replacing
Joy--Luo data, energy-loss straggling, hard inelastic events, channeling, and
anisotropic bremsstrahlung.

## Implementation path and likely owners

`montecarlo/transport.py` and `transport_jit_kernel.py` own propagation and the
segment schema. `montecarlo/spectrum.py` owns representative-state CXR and
bremsstrahlung integration. `montecarlo/runner.py`, results/checkpoint storage,
plots, and apps are schema consumers and must be inventoried before migration.

Advance each material state by the minimum of physical collision distance,
nearest physical boundary, cutoff distance, and numerical energy-loss limit.
When only the numerical limit wins, update energy/time/hazard without
scattering and retain the same `flight_id`. Sample collisions in optical depth
or demonstrate convergence of the chosen hazard approximation.

Numerical substeps are integration detail:

- coherent CXR sums substep fields into their physical flight before later
  coherent reductions;
- default incoherent CXR does not treat substeps as independent emitters; and
- bremsstrahlung integrates along the flight by midpoint or refined quadrature.

## Checklist

- [ ] A -- Inventory every segment-schema consumer and write the migration
      contract for `E_keV`, `t_ang`, identifiers, checkpoints, and device
      staging.
- [ ] B -- Add per-flight diagnostics: fractional loss, hazard change, clock
      estimate, cutoff overshoot, and percentile summaries.
- [ ] C -- Implement `E_start`/`E_end` plus midpoint predictor-corrector stopping
      and midpoint/integrated clock while retaining one radiation object per
      physical flight.
- [ ] D -- Add the CXR endpoint resonance-drift and bremsstrahlung quadrature
      estimators; select warning thresholds from observed convergence rather
      than treating 0.05/1% suggestions as universal constants.
- [ ] E -- Establish 2%, 1%, and 0.5% fractional-loss convergence matrices over
      thin/thick, low/high-Z, and 1--300 keV cases.
- [ ] F -- Introduce physical-flight and numerical-substep identity and an
      energy-controlled propagator with collision optical-depth handling.
- [ ] G -- Make CXR and bremsstrahlung invariant to numerical substep refinement
      at fixed physical flights.
- [ ] H -- Port the accepted algorithm to lockstep, grooved, per-electron, and
      CUDA paths without weakening deterministic/statistical parity contracts.
- [ ] I -- Update public docs, validation ledger, checkpoint/schema handling,
      and golden data; run fresh-context physics validation.

## Decisions and open questions

- **Decided:** numerical substeps must never create collision events,
  independent radiation intensities, or artificial decoherence merely because
  a tolerance was tightened.
- **Decided:** physical boundary/collision identity remains distinct from
  numerical integration identity.
- **Open:** exact compatibility lifetime and meaning of `E_keV` during
  migration (`E_start` alias versus representative/midpoint energy).
- **Open:** integrated optical-depth inversion versus bounded piecewise-constant
  hazard. Choose from correctness, convergence, Numba/CUDA feasibility, and
  measured cost.
- **Open:** storage shape: emit substeps in the existing segment arrays, retain
  a compact physical-flight table plus quadrature state, or reduce substeps
  before return.
- **Open:** accepted convergence tolerances for transport metrics, spectra, and
  coherent complex fields. Establish empirically before dispatching later
  slices as `one-shot`.

## Delegation slices and required skills

- A--B require `lead-task`, `repo-orientation`, `monte-carlo`, and
  `scientific-library`; not Serena `one-shot` while schema decisions remain.
- C--E require `lead-task`, `monte-carlo`, `regression-testing`, and
  `physics-review`; each checkpoint needs independent numerical evidence.
- F--H require `lead-task`, `monte-carlo`, `performance`, and
  `remote-gpu-jobs`; not `one-shot` because CPU/GPU algorithm identity and
  coherence semantics cross subsystem boundaries.
- I requires fresh-context `physics-validation`, `regen-golden`,
  `documentation-maintenance`, and `run-cxr-mc`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite core
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
```

- Transport ranges, exit fractions, energy deposition, spectra, and coherent
  complex fields converge across the documented 2%/1%/0.5% step refinements.
- Splitting one physical flight into numerical substeps leaves its final CXR
  field and default incoherent yield invariant within the accepted tolerance.
- Bremsstrahlung midpoint/trapezoid estimates converge on the refined result.
- Tightening the energy tolerance does not change physical collision statistics
  beyond the documented Monte Carlo confidence interval.
- All four execution paths implement the same accepted algorithm; CUDA evidence
  comes from `cxr remote`, not a local heavy run.
- New physics equations and numerical claims carry validation markers, ledger
  rows, assumptions/limits, fresh-context validation, and human-only sign-off.
