# Energy-controlled electron transport

## Problem and scope

The current transport freezes energy, speed, stopping power, elastic hazard,
and radiation kinematics at the start of each physical flight. This is a
left-endpoint rule with no convergence control; the exponential flight tail can produce large energy/hazard changes and coherent clock error.

Source: Stages 1 and 3 of the August 2026 electron-transport review. The
temporary review DOCX was retired in `fdac4ef`; this record preserves its
actionable findings and decisions.

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

`src/pyrite/montecarlo/transport.py` and `transport_jit_kernel.py` own
propagation and the segment schema. `montecarlo/spectrum/lines.py` and
`montecarlo/spectrum/brem.py` own representative-state CXR and bremsstrahlung integration. `montecarlo/runner/__init__.py`, results/checkpoint storage, plots, and apps are schema consumers and must be inventoried before migration.

Repository naming in this record follows the PyRITE migration: importable code lives under `src/pyrite/`, developer commands use `pyrite-dev`, and remote runtime checks use `pyrite remote`. Scientific CXR terminology is unchanged.

Advance each material state by the minimum of physical collision distance,
nearest physical boundary, cutoff distance, and numerical energy-loss limit.
When only the numerical limit wins, update energy/time/hazard without
scattering and retain the same `flight_id`. Sample collisions in optical depth or demonstrate convergence of the chosen hazard approximation.

Numerical substeps are integration detail:

- coherent CXR sums substep fields into their physical flight before later
  coherent reductions
- default incoherent CXR does not treat substeps as independent emitters;
- bremsstrahlung integrates along the flight by midpoint or refined quadrature

## Checklist

- [x] A -- Inventory every segment-schema consumer and write the migration
      contract for `E_keV`, `t_ang`, identifiers, checkpoints, and device
      staging.
- [x] B -- Add per-flight diagnostics: fractional loss, hazard change, clock
      estimate, cutoff overshoot, and percentile summaries.
- [x] C -- Implement `E_start`/`E_end` plus midpoint predictor-corrector stopping
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
  independent radiation intensities, or artificial decoherence merely because a tolerance was tightened.
- **Decided:** physical boundary/collision identity remains distinct from
  numerical integration identity.
- **Decided:** emitted rows remain numerical integration rows in the existing
  segment arrays. Later radiation code groups them by `(electron_id,
  flight_id)` before applying physical-flight coherence/incoherence semantics;
  checkpoints continue to receive reduced spectra rather than raw transport
  rows. A compact second flight table would duplicate geometry/state and make
  every current masking/staging consumer dual-schema.
- **Decided:** `E_keV` remains an unscheduled public compatibility alias of
  `E_start_keV`; it must never silently change to midpoint/representative
  energy. New propagation adds `E_end_keV` and an explicit `E_repr_keV` for
  radiation/quadrature. The line and brem kernels migrate to the explicit
  representative field only when their substep-invariance behavior lands.
- **Decided:** `t_ang` remains an unscheduled compatibility alias of
  `t_start_ang` (relative age, c=1). New propagation adds integrated
  `t_end_ang`; representative emission time is explicit rather than changing
  `t_ang`. `t0_ang` remains the separate per-electron bunch offset.
- **Decided:** `electron_id` is the canonical new spelling and `elec_id`
  remains its compatibility alias. `flight_id` is zero-based and monotonic
  within each electron, so the stable physical key is `(electron_id,
  flight_id)` independent of batching/backend row order. `substep_id` is
  zero-based within that flight. Collision, material/vacuum boundary, and
  terminal events close a physical flight; a numerical energy-limit event does
  not.
- **Decided:** every per-row schema transform (layer filtering, population
  cutoff clipping, device staging, and later flight grouping) uses one owning
  field registry. New float fields stage with backend `REAL`; identifiers retain
  integer dtype. Fixed-size diagnostics and per-electron arrays remain on host.
  Until H ports the accepted propagator, energy-controlled CUDA mode must fail
  closed rather than return a partial schema; legacy CUDA transport remains
  available.
- **Decided:** raw transport segments are not checkpoint payloads. Current
  checkpoints/results store reduced spectra, counts, and case metadata; no
  checkpoint migration is needed for A--H unless a later slice deliberately
  persists diagnostics or raw flights. External direct callers of
  `simulate_trajectories` receive the aliases above.
- **Decided:** the propagation rule is selected by `energy_model`
  (`"frozen"` default, `"midpoint"`), following the existing `elastic_model` /
  `transport_core` string-mode convention. New per-row fields appear only under
  the mode that produces them, so the frozen schema is never partially
  extended; unported cores raise on a midpoint request.
- **Decided:** `_SEG_ARRAYS` in `montecarlo/spectrum/lines.py` is the owning
  field registry named by decision A. Every per-row transform loops it with a
  presence guard, so optional fields are safe to register — but an unregistered
  per-row array survives a row mask at full length and silently desynchronizes,
  which is why the new fields are registered in the same slice that adds them.
- **Decided:** `_clip_segments_to_cutoff` drops `E_end_keV`/`t_end_ang` when it
  shortens a flight. Its left-endpoint clip rule cannot reconstruct a
  midpoint-integrated end state for the shortened flight, and a stale end state
  is worse than an absent one.
- **Open:** integrated optical-depth inversion versus bounded piecewise-constant
  hazard. Choose from correctness, convergence, Numba/CUDA feasibility, and
  measured cost.
- **Open:** accepted convergence tolerances for transport metrics, spectra, and
  coherent complex fields. Establish empirically before dispatching later
  slices as `one-shot`.

## A -- segment-schema consumer inventory

Evidence is against the post-PyRITE package layout at `398ad4b` plus this
slice. `simulate_trajectories` is the sole producer. Its eight material-row
arrays are `r_mid`, `v_hat`, `L_ang`, `E_keV`, `t_ang`, `t0_ang`, `elec_id`,
and `layer`; groove `vacuum_*`, incident `initial_*`, geometry, and count fields
are separate shapes/scalars.

- `montecarlo/spectrum/lines.py`: `_SEG_ARRAYS` owns layer filtering, cutoff
  clipping, and host/device staging. `mc_spectrum` consumes position,
  direction, length, start energy, electron identity, and—for coherent mode—
  start age plus bunch offset. `_segment_escape_distance` consumes geometry.
- `montecarlo/spectrum/brem.py`: consumes electron identity, midpoint, length,
  and start energy after shared cutoff clipping; layer splitting is driven by
  the runner. It currently treats every row as an independent path integral.
- `montecarlo/runner/__init__.py`: `_transport_case`, `_brem_for_case`, and
  `_transport_lines_for_case` produce/mask shared populations; line and brem
  helpers filter by layer, stage one device copy, and reduce `L_ang.size` to
  `n_segments`. Transport-pool segment pickling is transient IPC, not durable
  checkpoint storage.
- `campaign/config.py`: `gate_cases_by_penetration` calls transport and consumes
  only exit/penetration summaries, not row energy/time semantics.
- `plots/mpl/trajectories.py`: direct producer call; reconstructs endpoints,
  orders tracks by `(elec_id, t_ang)`, colors by `E_keV`, and exports the row
  payload to Altair/Plotly-ready trajectory data. Altair and Plotly consumers
  then use the normalized `E`, time, geometry, and electron-id columns rather
  than the raw transport mapping.
- `apps/trace_app.py`: consumes normalized trajectory and incident phase-space
  data from the matplotlib owner. `apps/anchor_figures.py` calls transport
  directly and passes rows to line/brem kernels or reads exit counts.
- `results/` and `checkpoints/`: no raw segment consumer or serializer found.
  `results/store.py` stores reduced line/coherent/brem arrays and metadata.
- Tests and fixtures under `tests/helpers/segments.py`, `tests/montecarlo/`,
  `tests/plots/`, and `tests/notebooks/` construct or assert the current mapping;
  schema/staging/coherence/cutoff tests must migrate with their owning paths.

## B -- bounded transport diagnostics

`simulate_trajectories(..., collect_diagnostics=True)` performs a deterministic
post-transport pass over each current physical-flight row. It predicts end
energy using the same left-endpoint stopping law, compares start/end elastic
hazard, compares left-endpoint and midpoint `L/beta` clocks, and measures
positive cutoff undershoot as overshoot. It returns only p50/p90/p99/max plus
`n_flights` under `transport_diagnostics`; empty summaries use `None`. Default
calls add no key or pass. No RNG draws, propagation state, spectra, or resident
device payload change; an explicit diagnostic request is the only path that
downloads resident segment fields.

## C -- midpoint predictor-corrector stopping and clock

`simulate_trajectories(..., energy_model=...)` selects the propagation rule.
`"frozen"` (default) is the historical left-endpoint rule, bit-for-bit
unchanged. `"midpoint"` advances each physical flight by the implicit midpoint
rule `E_end = E_start + (dE/ds)((E_start+E_end)/2)·s`, evaluated by one
predictor-corrector pass, and advances the clock by `s/β((E_start+E_end)/2)`.
The cutoff truncation distance is solved for `E_end = E_cut` rather than
extrapolated, so a cutoff-stopped flight lands on the floor exactly and the
frozen rule's range overshoot disappears.

Measured local truncation error against a 20k-substep RK4 reference (one
boundary-truncated carbon flight at 25 keV; 600/300/150 Å, 0.51%/0.25%/0.13%
fractional loss): midpoint ratios 8.04/8.02 (energy) and 7.69/7.85 (clock)
versus frozen 4.01/4.00 — third-order local, second-order global, one order
above the frozen rule. Full numbers and the derivation are in
`docs/validation/beam-transport/transport-midpoint-stopping.md`;
`Validation: transport-midpoint-stopping`.

Scope held deliberately tight: elastic hazard stays frozen at `E_start` (step
F), radiation kernels still read the start energy (steps D and G), and the
flight decomposition is untouched, so one radiating row per physical flight
still holds. Only the ungrooved lockstep core carries the propagator; grooved,
per-electron, and CUDA midpoint requests raise (step H).

## Delegation slices and required skills

- A--B require `lead-task`, `repo-orientation`, `monte-carlo`, and
  `scientific-library`; not Serena `one-shot` while schema decisions remain.
- C--E require `lead-task`, `monte-carlo`, `regression-testing`, and
  `physics-review`; each checkpoint needs independent numerical evidence.
- F--H require `lead-task`, `monte-carlo`, `performance`, and
  `remote-gpu-jobs`; not `one-shot` because CPU/GPU algorithm identity and coherence semantics cross subsystem boundaries.
- I requires fresh-context `physics-validation`, `regen-golden`,
  `documentation-maintenance`, and `run-cxr-mc`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite core
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
```

- Transport ranges, exit fractions, energy deposition, spectra, and coherent complex fields converge across the documented 2%/1%/0.5% step refinements.
- Splitting one physical flight into numerical substeps leaves its final CXR
  field and default incoherent yield invariant within the accepted tolerance.
- Bremsstrahlung midpoint/trapezoid estimates converge on the refined result.
- Tightening the energy tolerance does not change physical collision statistics beyond the documented Monte Carlo confidence interval.
- All four execution paths implement the same accepted algorithm; CUDA evidence comes from `pyrite remote`, not a local heavy run.
- New physics equations and numerical claims carry validation markers, ledger rows, assumptions/limits, fresh-context validation, and human-only sign-off.
