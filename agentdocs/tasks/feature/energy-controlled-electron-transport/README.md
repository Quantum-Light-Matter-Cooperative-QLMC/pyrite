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
- [x] D -- Add the CXR endpoint resonance-drift and bremsstrahlung quadrature
      estimators; select warning thresholds from observed convergence rather
      than treating 0.05/1% suggestions as universal constants.
- [x] E -- Establish 2%, 1%, and 0.5% fractional-loss convergence matrices over
      thin/thick, low/high-Z, and 1--300 keV cases. Measured; the headline
      result is that a fractional-loss cap is the wrong control variable and
      the binding tolerance is an absolute emission phase.
- [x] F -- Introduce physical-flight and numerical-substep identity and an
      energy-controlled propagator with collision optical-depth handling.
      Implemented on both lockstep cores; ensemble evidence measured and
      ledgered as `energy-controlled-propagation`.
- [x] G -- Make CXR and bremsstrahlung invariant to numerical substep refinement
      at fixed physical flights. Flight-grouped incoherent CXR plus an explicit
      representative energy; measured and ledgered as
      `substep-radiation-invariance`. Host-only, non-batched path; the ports are
      slice H.
- [x] H -- Port the accepted algorithm to lockstep, grooved, per-electron, and
      CUDA paths without weakening deterministic/statistical parity contracts.
- [x] I -- Update public docs, validation ledger, checkpoint/schema handling,
      and golden data; run fresh-context physics validation. All five rows are
      now `rederived` by independent fresh contexts. No checkpoint or golden
      migration was needed: checkpoints store reduced spectra, and the new row
      fields appear only under the non-default `energy_model="midpoint"`.
      Human sign-off is the only remaining ledger transition.

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
- **Decided (E):** the step-control variable is an **absolute emission-phase
  tolerance**, not a fractional energy-loss cap. A coherent kernel weights each
  row by `e^{i omega t_abs}`, the age is 10^3--10^4 Ang, and the clock error
  accumulates along a trajectory while `|dE|/E` is per-flight, so the two are
  only loosely related. Measured accepted tolerance: accumulated per-electron
  `|dphi|` p99 < 0.1 rad at the case's resonance.
- **Decided (E):** the midpoint rule is a **prerequisite** for coherent CXR, not
  an optimization. At one row per flight it gives accumulated phase error
  1e-3--0.2 rad against 10--10^3 rad for the frozen rule, three to four orders
  better at equal cost; the frozen rule does not reach the tolerance anywhere
  on the 2%/1%/0.5% ladder. Slices F--H must therefore treat `energy_model` as
  load-bearing for radiation, not just for transport.
- **Decided (E):** `elec_id` is **not** a grouping key in the line kernel.
  `lines.py` uses it only as the row mask `elec_id < Ne`, and
  `mc_spectrum(coherent=True)` sums ONE global complex field over every
  surviving row; per-electron decoherence is emergent from the `t0_ang` spread.
  Slice G must add explicit `(electron_id, flight_id)` grouping to the kernel
  itself -- remapping `elec_id` to per-flight ids does not produce a
  flight-incoherent reduction, it silently masks nearly every row away.
- **Decided (E):** the requested 2%/1% rungs are **no-ops** on these cases. The
  elastic mean free path already holds per-flight fractional loss below 2% for
  ~99% of flights (p99 = 1e-2 to 1e-1 across the matrix), so those rungs
  reproduce the unrefined row set almost exactly. Only 0.5% and below bind, and
  then only on the top decile of flights.
- **Open (raised by E):** the production global-coherent reduction appears
  **ill-conditioned** to any per-row change. `|sum_j E_j|^2` is a small residual
  of a large cancellation, so it does not converge under the ladder even with
  accurate phases (54% residual in the 100 keV thick carbon case). The
  physical-flight-incoherent reduction on the same rows converges roughly 3x
  better and monotonically. This is evidence for slice G's grouping, but the
  conditioning of the global sum itself needs its own decision.
- **Decided (E):** the CSDA clock the whole slice converges to is a
  **mean-value construct**, and the tolerance bounds numerical error against it,
  not physical phase fidelity. Fresh-context literature review (2026-08-11)
  confirms the microphysics is discrete stochastic loss at constant velocity
  between events, and that the "many inelastic events per flight" justification
  fails here -- inelastic and elastic mean free paths are comparable (about 1.0
  inelastic event per flight in C at 25 keV, 0.11 in W). Smooth in-flight
  deceleration is a modeling choice, not a limit theorem. It is nonetheless the
  right choice: the review independently reproduces the slice-E result that
  freezing `beta` costs hundreds to thousands of radians per micron
  (365 rad/um at 25 keV in C, matching the measured 11--1300 rad over
  0.2--2.6 um trajectories).
- **Decided (E):** the 0.1 rad tolerance stands because the frozen rule's error
  is **systematic** (one-signed, accumulates coherently across the ensemble)
  whereas unmodeled straggling jitter is **random** (suppresses the line via
  `exp(-sigma_phi^2/2)` rather than displacing it). The two are not
  commensurate and must not be compared as if they were.
- **Open (raised by E):** energy-loss **straggling is unmodeled**, and by
  Jensen's inequality it biases the *mean* arrival time, not just its variance
  (~0.3 rad at 25 keV over 1 um at 1 keV) -- above the 0.1 rad numerical
  tolerance. Numerical precision has outrun the transport model there. Out of
  scope for F--H; needs its own task if coherent absolute phase is ever claimed
  to that accuracy.
- **Decided (F):** the collision draw is a **per-physical-flight optical-depth
  budget**, consumed across substeps at each substep's own hazard
  (`tau -= ds / lambda(E_substep)`), rather than an analytic inversion of
  `int ds / lambda(E(s))`. Browning/Mott `lambda(E)` is tabulated, so no closed
  form exists to invert; the budget is one extra float of per-electron state,
  needs no RNG per substep, and reduces exactly to the current draw when the cap
  is disabled. This is the discretized integrated inversion, not the
  bounded-piecewise-constant alternative: the *total* optical depth is
  integrated, only the hazard within one substep is held constant.
- **Decided (F):** the collision-statistics acceptance criterion is **met at a
  stated resolution, not proved**. Over C 25 keV (4000 Ang and thick), W 25 keV
  thick, and C 100 keV thick at Ne = 1000 x 12 independent seeds per rung, all
  160 paired shifts against the finest rung are below 2.3 sigma and none reaches
  the 3 sigma flag. The predicted bias is real and one-signed -- freezing
  `lambda` at `E_start` understates a hazard that rises as `E` falls, so
  unrefined flights are slightly too long -- and its largest appearance (2.25
  sigma on mean flight length, C 25 keV thick, correct sign, the case slice E
  found has the matrix's largest per-flight fractional loss) is consistent with
  a small convergent bias that a larger ensemble would resolve rather than
  contradict. Recording it as a null result at a resolution, not as invariance,
  is the honest reading.
- **Decided (F):** rungs are compared by the **paired** per-seed difference.
  Paired and unpaired shifts differ by at most ~0.7 sigma with the paired ones
  larger, so a little variance survives the shared seed and pairing is the more
  sensitive test; the small margin independently confirms the decorrelation
  claim below.
- **Resolved (was open, raised by F):** refining `max_dE_frac` **decorrelates trajectories**,
  so per-realization flight counts do not converge — measured 2568 / 2882 / 2599
  / 2553 / 2605 distinct flights at f = 1e-2 / 5e-3 / 2e-3 / 1e-3 / 5e-4 (C,
  25 keV, 4000 Ang, Ne = 200, seed 7). Changing the substep grid changes the
  energies at which the hazard is evaluated, which changes the sampled collision
  point, which changes the whole downstream trajectory. The acceptance criterion
  "tightening the tolerance does not change physical collision statistics" must
  therefore be measured the way slice E measured Part A: ensemble means with
  Monte Carlo standard errors over seed replicates, not a single-seed count.
  Measured; see the F decisions above.

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

## F -- flight/substep identity and the energy-controlled propagator

`simulate_trajectories(..., energy_model="midpoint", max_dE_frac=f)` caps one
row's fractional energy loss. When that cap binds before any physical event the
core emits a row and resumes the same physical flight: same direction, same
`flight_id`, `substep_id + 1`, no scatter, no new collision draw. `f = 0.0`
(default) leaves one row per flight, and `max_dE_frac > 0` requires
`energy_model="midpoint"` — substepping a frozen flight is exactly the
mis-phased configuration slice E rejected, so the two are not independently
selectable.

The collision is drawn once per physical flight as an optical depth
`tau = -log(U)` and consumed as `tau -= ds / lambda(E_substep)` per substep;
the flight closes when `tau` is exhausted, at a boundary, at the cutoff, or on
termination. With the cap disabled every iteration is a whole flight, one draw
each, and `-lam * log(U)` is replaced by the bit-identical `(-log(U)) * lam`,
so **frozen mode is bit-for-bit unchanged** (verified by hashing `r_mid`,
`v_hat`, `L_ang`, `E_keV`, `t_ang`, `elec_id`, `layer` and the exit counts over
C/W x 25/100 keV against `main`).

Rows carry `flight_id` (zero-based, monotonic per electron) and `substep_id`
(zero-based within the flight) under midpoint mode only; `electron_id` is added
as the canonical spelling of `elec_id`. All three are registered in
`_SEG_ARRAYS`, so layer filtering, cutoff clipping, and device staging carry
them with the rows.

Owned by `_transport_core_ungrooved` and `_transport_core_ungrooved_lut`. The
grooved, per-electron, and CUDA cores still raise on a midpoint request; that
port is slice H.

`checks/collision_statistics_refinement.py` supplies the acceptance evidence:
per-electron flight counts, mean physical flight length, path, retained energy,
transit clock, and exit fractions over the ladder `none`/1%/0.5%/0.2%/0.1%/0.05%
at Ne = 1000 x 12 independent seeds per rung, compared by paired per-seed
difference. Derivation, tables, assumptions, and limits are in
`docs/validation/beam-transport/energy-controlled-propagation.md`;
`Validation: energy-controlled-propagation`. The row's claim is scoped to
collision statistics -- substep invariance of emitted CXR and bremsstrahlung is
slice G and is explicitly not claimed there.

## G -- substep-invariant radiation

Two rules, both in the radiation kernels rather than in transport.

**Representative energy.** Midpoint transport now returns `E_repr_keV =
(E_start + E_end)/2`, the energy the implicit midpoint rule already evaluates
`dE/ds` and `beta` at. `mc_spectrum` and `mc_brem_spectrum` read it when
present and fall back to `E_keV` otherwise, so frozen rows are bit-for-bit. This
turns each row's one-point path integral from a left-endpoint rule into a
midpoint rule, and makes the line kernel's `t_L = L/beta(E_repr)` exactly the
transported flight duration `t_end - t_start`.

**Flight grouping.** The default incoherent CXR reduction now sums rows of one
`(electron_id, flight_id)` as complex field and squares the sum; only whole
flights add incoherently. Without it, splitting a flight into `N` substeps gives
`N` rows each carrying `(t_L/N)^2` instead of one carrying `t_L^2`, so the line
peak falls roughly as `1/N` -- tightening a numerical tolerance would dismantle
the line. At frozen energy and clock the grouped sum recovers the unsplit row
exactly by the Dirichlet-kernel identity, so all residual under refinement is
the physical variation of `E` and `beta` along the flight.

Implementation notes:

- Grouping is keyed by `(electron_id, flight_id)` value (a `lexsort`), not by
  adjacency. Slice G first used adjacency because `subdivide_flights` emits a
  flight's substeps contiguously; slice H found that the lockstep core does not.
  All singleton groups means one row per flight, whose grouped reduction is
  algebraically the incoherent one, so the proven path is kept and `grouped`
  stays `False`. Production frozen runs never take the new path.
- The accumulation is `np.add.reduceat` over blocks whose boundaries are snapped
  to flight boundaries (`_flight_blocks`), so no flight is split across two
  squarings. Rejected: cumsum-and-difference (precision) and a dense
  group-indicator matmul (memory).
- Grouped runs take the per-`hkl` `_accumulate` loop and reuse the coherent
  prologue (`cdtype`, `seg_t_mid`, `d_all`, `omega_grid`, `delta_omega_grid`,
  `pol_A`). The polarization amplitude list had been populated only under
  `coherent`; the incoherent path uses the fused real kernel, so grouped runs
  produced exactly zero until that guard was widened.
- Fails closed rather than silently degrading: non-NumPy backend, `components=
  True`, and `xray_dispersion="refractive"` with `layers` all raise on
  substepped rows. Batched, grooved, per-electron, and CUDA ports are slice H.

**Reversed decision.** `_clip_segments_to_cutoff` no longer drops `E_end_keV`/
`t_end_ang`. Rows carrying an end state came from midpoint transport, so the
truncation distance solves `E_end == E_cut` under that same rule and the end and
representative states are reconstructed exactly rather than dropped. The
radiation kernels need `E_repr_keV` to survive the population cutoff clip, and
frozen rows keep the left-endpoint solve bit-for-bit.

**`subdivide_flights`.** New public helper in `spectrum/diagnostics.py`: splits
fixed physical flights along their own straight rays into equal-length substeps
and rebuilds the full midpoint schema. This is what makes the measurement a
quadrature measurement -- re-running transport at a tighter `max_dE_frac` also
moves the sampled collision points and decorrelates the trajectories. It
duplicates the research helper `_subdivide` inside
`checks/energy_step_convergence_matrix.py`, which was deliberately left
untouched so slice E's already-ledgered evidence stays reproducible.

Acceptance evidence is `checks/substep_invariance.py`; derivation, tables,
assumptions, and limits in
`docs/validation/beam-transport/substep-radiation-invariance.md`;
`Validation: substep-radiation-invariance`.

## H -- porting the propagator to every core

Slice F landed the controlled propagator on both lockstep cores. H carries it to
the per-electron cores, the two CUDA rawkernels, and the grooved core, and drops
both `energy_model="midpoint"` fail-closed gates.

**Per-electron cores.** Same statement order as lockstep. `_dEds_packed_scalar`
is new: the per-electron and CUDA cores read the padded `(n_layers,
max_elements)` tables whose padding is zero-filled, so `_dEds_compound_scalar`
would evaluate `log(1.166 (E + 0)/0)`; the loop is bounded by `n_el` instead.
`_alloc_scratch` grew a `midpoint` flag and allocates the four end-state/identity
buffers at length zero under the frozen rule, so the core signature is fixed
while the frozen row stays exactly seven fields wide.

**CUDA.** The transpiler has no `continue`, so step 5 became
`if limited_j: substep_id += 1` / `else: <exit, boundary, collision>` with the
flight-close reset hoisted into the non-terminal branch. Two new device
functions, `_dEds_packed` and `_lut_lerp_at`, because the midpoint rule
evaluates `dE/ds` and the LUT rows at three energies per flight rather than one
(the frozen path indexes the energy grid once and reuses the index). The
launcher signatures still match the per-electron cores positionally, pinned by
`test_launcher_signature_tracks_the_reference_core`.

**Grooved.** Electrons are visited round-robin, so the optical-depth budget and
flight identity are per-electron arrays rather than loop locals: a substep must
resume the flight the previous outer pass opened. A substep also suppresses the
groove surface event, so `full_j` excludes it and no collision is sampled
mid-flight. Substeps consume the material step budget, as in the ungrooved cores.

**Frozen bit-for-bit.** `-lambda log(U)` became `(-log(U)) lambda`, which is
exact (negation is exact, multiplication commutative). Verified against `main`
by SHA-256 over the full row set: per-electron over C/W x 25/100 keV x LUT
on/off, grooved over C at 30/60 keV x both elastic models including vacuum legs.

**Found and fixed: flight grouping was adjacency-keyed.** Slice G's grouped
incoherent CXR reduction detected groups from adjacent-row key changes, which
holds for `subdivide_flights` output but not for the lockstep core -- it emits
step-major, so a flight's substeps are separated by every other electron's row
for that step. The reduction saw all-singleton groups, fell back to the plain
incoherent path, and divided the line peak by the substep count: the exact
failure the grouping exists to prevent, on the default core. Grouping is now
keyed by `(electron_id, flight_id)` value and the reduction gathers each
flight's rows with a stable sort, which leaves already grouped input and each
group's internal order untouched.

**Remainder, deliberately not in H.** The grouped reduction stays host-only and
stays on the per-`hkl` loop:

- The device port needs a segmented complex reduction with no CuPy `reduceat`;
  cumsum-and-difference was already rejected on precision grounds in G. It fails
  closed with an actionable message, so no result is silently wrong.
- The batched path falls back to the proven per-`hkl` loop on grouped rows. That
  is a launch-count cost on one configuration, not a correctness gap.

Both need GPU evidence through `pyrite remote` and are performance ports of an
already-correct path. The `components=True` and
`xray_dispersion="refractive"`-with-`layers` guards are *not* ports: they are
genuine modelling gaps (the PXR/CBS cross term survives the intra-flight sum;
the per-layer dispersive propagation phase is unmodelled) and stay closed.

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

## D -- radiation error estimators

`montecarlo/spectrum/diagnostics.py` adds two host-side, opt-in, read-only
estimators exported through `pyrite.montecarlo.spectrum`:
`cxr_endpoint_resonance_drift` (per-flight sweep of the kernel's own resonance
condition between endpoint speeds, in units of the flight's sinc linewidth,
worst over the caller's reflection set) and `brem_endpoint_quadrature_error`
(per-flight grid-integrated and worst-bin relative difference between the
kernel's left-endpoint evaluation and a midpoint one). Both read the
transported `E_end_keV` under `energy_model="midpoint"` and otherwise predict
the end state with the same left-endpoint Joy--Luo rule as the slice-B
transport diagnostics, so they apply to frozen runs unchanged. No default call
path computes them; kernel migration to a representative energy stays with
step G.

Warning thresholds are calibrated, not assumed:
`checks/radiation_error_estimator_calibration.py` measures both estimators
against the ACTUAL spectral change from midpoint evaluation on identical
segments over a C/W x thin/thick x 25/100 keV matrix. Drift p99 of 2.0--2.6
linewidths corresponded to 4--10% line-region L1 changes, so the default warns
at p99 = 1.0 linewidths; measured brem L1 ran at ~0.4x the p99 estimate, so
the default warns at p99 = 1e-2 (~0.4% continuum change, above every measured
case). The review's 0.05/1% figures match the measured TYPICAL errors, not
useful warning levels. Full table and derivation:
`docs/validation/beam-transport/radiation-error-estimators.md`;
`Validation: radiation-error-estimators`.

## E -- convergence matrices

`checks/energy_step_convergence_matrix.py` measures four things over a
C/W x thin/thick x 5--300 keV matrix at `E_cut = 1 keV`, seed 7. Full tables and
derivation: `docs/validation/beam-transport/energy-step-convergence.md`;
`Validation: energy-step-convergence`.

**Part A -- transport, frozen vs midpoint (Ne = 4000).** The rules diverge
trajectory-by-trajectory, so every observable is quoted with its Monte Carlo
standard error and the shift in units of the combined error. Across 14 cases
and 7 metrics, every shift is within 1.6 sigma except one: the mean path length
of `C 5 keV thick`, +4.3 sigma (3460 vs 3424 Ang, 1.2%). Replicated at seeds
7/101/2024/31337 at +4.3/+4.9/+4.6/+4.9 sigma, so it is a real bias and not the
expected tail of 98 comparisons. Sign and location are both predicted: Joy--Luo
`|dE/ds|` grows as `E` falls, so the left-endpoint rule understates the loss,
needs more path to reach `E_cut`, and **overstates the CSDA range**. The effect
tracks per-flight fractional loss, which peaks in exactly that case (p99 = 9.9%,
matrix maximum) and is below 1% for every case at or above 100 keV, where all
shifts are within 0.6 sigma.

**Part B -- radiation ladder at fixed physical flights (Ne = 200).** One frozen
transport run supplies the flights; only the numerical sampling of the emission
integral is refined, so trajectory divergence cannot confound the measurement.
Substeps and the reference rung both use the midpoint rule, so the reference is
not itself mis-phased. Bremsstrahlung converges cleanly and first order
(1.9e-3 to 8.2e-4 down the ladder, floor 1.5e-5). Coherent CXR does not, and
the frozen-integrated repeat of the same ladder (`cxrCfz`) stalls a factor 2--3
above the midpoint one, which is the propagation rule showing up directly in a
spectrum.

**Part B phase criterion -- the operative result.** The accumulated
per-electron clock error against the midpoint rule at `f = 0.125%`, converted to
radians at the case resonance. Under the frozen rule it is 11--1300 rad at one
row per flight and falls only first order, reaching 3--900 rad at `f = 0.5%`:
the frozen rule is unusable for coherent CXR at every rung of the requested
ladder. Under the midpoint rule it is 1.4e-3--6.0e-2 rad p99 at one row per
flight, already inside a 0.1 rad tolerance in all five cases, and `f = 2%`
takes the worst case max to 5.9e-2 rad.

**Part C -- physical-flight-incoherent CXR (Ne = 60).** One coherent kernel call
per flight, which is what slice G owes the kernels. It converges (1.6e-1 to
5.3e-2) while the production global-coherent reduction on the same rows is
about 3x worse and non-monotone. Row counts across the ladder are 409, 409, 421,
474 -- direct evidence that the 2% and 1% rungs subdivide essentially nothing.

**Part D -- row-splitting floor vs take-off geometry.** At frozen energy AND
clock, subdividing a flight is an exact algebraic identity for the coherent sum
(Dirichlet-kernel composition), so any residual is the kernel's own per-row
approximation. At the default 119 degree take-off it is 2.41e-3 (f=1%) to
8.94e-3 (f=0.125%) grid L1. At the near-grazing `n_hat = (1, 0, 0.01)` of
`tests/montecarlo/test_coherent_emission.py` it is 8.71e-2 to 3.38e-1, because the
Beer--Lambert escape path is the depth divided by `n_z` and so swings ~2x within
a single flight. This is a per-row escape-factor quadrature defect, independent
of `energy_model` and of the slice-E step control, and it bounds what any
energy-step ladder can resolve in that geometry. The slice-E CXR matrix
therefore uses the production default take-off, not the test geometry.

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
