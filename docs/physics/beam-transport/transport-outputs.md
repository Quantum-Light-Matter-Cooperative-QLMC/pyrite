# Transport outputs

What a transport run hands to the radiation kernels, and what those quantities do and do not mean. This is the interface every spectrum in the package is built on.

## Segments are histories, not events

A segment is one straight flight of one electron through one material. It is **not** a detector event, a photon, or an independent emitter. The line and bremsstrahlung kernels each read the same segment table and reduce it their own way; nothing in transport anticipates either.

Two consequences:

- segment counts have no physical normalization of their own: refining the numerical step raises the row count without changing any observable;
- the physically meaningful key is `(electron_id, flight_id)`, the physical flight, which is stable under batching and independent of row order.

## Per-segment arrays

```{list-table} Per-segment quantities, one row per segment.
:name: tbl-outputs-segment-schema
:header-rows: 1

* - Field
  - Shape
  - Meaning
* - `r_mid`
  - (M, 3) Å
  - flight midpoint position
* - `v_hat`
  - (M, 3)
  - unit direction, constant along the flight
* - `L_ang`
  - (M,) Å
  - flight length
* - `E_keV` / `E_start_keV`
  - (M,) keV
  - kinetic energy at the flight start
* - `t_ang` / `t_start_ang`
  - (M,) Å, $c=1$
  - transport age at the flight start
* - `t0_ang`
  - (M,) Å, $c=1$
  - this electron's bunch arrival offset
* - `elec_id` / `electron_id`
  - (M,)
  - originating electron
* - `layer`
  - (M,)
  - index of the emitting layer
```

`E_keV`, `t_ang`, and `elec_id` are compatibility aliases of the canonical spellings and keep flight-start semantics under every propagation rule.

### Fields added by midpoint propagation

`energy_model="midpoint"` adds `E_end_keV`, `t_end_ang`, the representative energy

```{math}
:label: eq-outputs-representative-energy

E_{\rm repr} = \tfrac12\left(E_{\rm start} + E_{\rm end}\right),
```

the `flight_id` / `substep_id` identifiers, and the row-end `event_kind`. {eq}`eq-outputs-representative-energy` is the point at which the radiation kernels evaluate their one-point path integrals, turning each row from a left-endpoint into a midpoint rule.

New fields appear **only** under the rule that produces them, so a consumer that finds `E_repr_keV` can rely on the rest of the midpoint fields being present, and one that does not can rely on every row being a whole physical flight.

### Fields added by the shell soft/hard mode

`inelastic_model="shell-soft-hard"` (midpoint only) adds per-row `hard_W_keV` and `hard_channel` and an `inelastic` metadata dict. A row ending in `HARD_INELASTIC` carries the transfer $W$, and the next row of that electron starts exactly $W$ lower. A `CUTOFF` row whose collision absorbed the primary also carries its $W$. Every other row has $W=0$ and channel $-1$. The result's `stopping_tables` are then the soft tables. See [Shell soft/hard inelastic transport](shell-soft-hard-transport.md).

### Fields added by secondary transport

`secondary_threshold_eV` (shell mode only) joins the rows of every generation into one result. Rows gain `track_id` (unique per trajectory; a primary's equals its electron index), `parent_id` ($-1$ for primaries), `generation` (0 for primaries) and `hard_secondary_v_hat`, the secondary's launch direction on each hard row and zero elsewhere. `electron_id`/`elec_id` stay the primary history, so `electron_limit` selections, normalization by `Ne` and `t0_ang` act on whole showers; the flight key is (`track_id`, `flight_id`), and `check_segment_event_contract` applies per track. Row order is generation-major. `n_backscattered`, `n_transmitted`, `n_side_exited`, `n_cutoff_stopped`, `straggle_dE_keV` and the `initial_*` arrays remain primary-only; select `generation == 0` for other primary-only observables. The result adds a `secondary_tracks` table (per track: identity, parent hard ordinal and launch state) and a `secondaries` dict (threshold, generation count, tracks and termination counts per generation). `transport.secondaries.secondary_energy_balance` evaluates the run's energy balance. See [secondary electron transport](shell-soft-hard-transport.md#secondary-electron-transport).

## Row events and physical segments

Every midpoint row records, in `event_kind` (`int8`), the event that ended it. The event sits at the row's far end, so the row itself carries the pre-event state — `v_hat` is the incoming direction, `E_end_keV` and `t_end_ang` the incoming energy and age — and the next row of the same electron carries the post-event state. An event's position is `r_mid + L_ang v_hat / 2`.

```{list-table} Row-end event codes, pyrite.montecarlo.transport.SegmentEvent.
:name: tbl-outputs-event-kinds
:header-rows: 1

* - Code
  - Name
  - Kind
  - Closes the flight
  - Direction after
* - 0
  - `ELASTIC`
  - physical interaction
  - yes
  - resampled
* - 1
  - `SUBSTEP`
  - numerical node
  - no
  - unchanged
* - 2
  - `LAYER_BOUNDARY`
  - geometry
  - yes
  - unchanged
* - 3
  - `GROOVE_SURFACE`
  - geometry
  - yes
  - unchanged (vacuum leg or escape)
* - 4, 5, 6
  - `EXIT_TOP`, `EXIT_BOTTOM`, `EXIT_SIDE`
  - geometry, terminal
  - yes
  - —
* - 7
  - `CUTOFF`
  - terminal
  - yes
  - —
* - 8
  - `HARD_INELASTIC` (reserved)
  - physical interaction
  - yes
  - resampled
* - 9
  - `HARD_RADIATIVE` (reserved)
  - physical interaction
  - yes
  - unchanged
* - 10
  - `DELTA` (reserved)
  - numerical node
  - no
  - unchanged
```

**Physical segment boundaries.** Elastic collisions, hard inelastic collisions, and hard radiative events are the physical interactions. Each one closes the flight: the next row opens `flight_id + 1` at `substep_id = 0`, and the default radiation reductions add the two flights incoherently. A hard radiative event keeps the electron's direction, following the PENELOPE-2024 convention (NEA/MBDAV/R(2024)1, §3.3), which leaves the photon recoil unmodelled and assigns the angular deflection to the elastic model; its discrete energy loss still ends the flight, because the emission resonance and phase velocity jump with the energy. Geometry events and the cutoff also close the flight. They are bookkeeping boundaries rather than interactions: the collision budget is redrawn after them, which the memoryless free-path law makes exact.

**Numerical nodes.** A `SUBSTEP` row, and a future fictitious (`DELTA`) interaction of a majorant free-path sampler, are integration detail. The next row keeps `flight_id`, advances `substep_id` by one, and continues the direction, energy, clock, position, and collision budget unchanged. Such rows never decohere a flight: the flight-grouped CXR reduction sums them as one complex amplitude (see `substep-radiation-invariance`).

**Invariants.** `check_segment_event_contract` enforces, on rows in any order:

- every electron opens flight 0 at substep 0;
- flight-continuing rows continue their flight, and every other row opens the next one;
- energy is continuous across every event except the hard events, where it may only fall;
- direction is unchanged across `SUBSTEP`, `DELTA`, `LAYER_BOUNDARY`, `GROOVE_SURFACE`, and `HARD_RADIATIVE`;
- clock and position are continuous across every event except a groove surface, whose vacuum leg advances both;
- a terminal event is its electron's last row, and every electron's last row is terminal or a groove surface through which it escaped.

The exit codes tally one-to-one against `n_backscattered`, `n_transmitted`, `n_side_exited`, and `n_cutoff_stopped`. The opt-in coupled radiative exact cores emit `HARD_RADIATIVE`; the default cores retain their existing event behavior.

**Event metadata.** Ordinary events need only the pair of rows around them. The pre-event state is that row's `E_end_keV`, `t_end_ang`, and `v_hat`; the post-event state is the next row's `E_start_keV`, `t_start_ang`, and `v_hat`, next in `(electron_id, flight_id, substep_id)`, not in array order, which is step-major on the lockstep core. The energy transferred is the row's `E_end_keV` less the next row's `E_start_keV`: the emitted photon's energy for `HARD_RADIATIVE`, the collision's loss for `HARD_INELASTIC`, and zero at every other event by the continuity invariant. The deflection is the angle between the two `v_hat`. The coupled radiative mode adds per-row `hard_radiative_k_eV`, `hard_radiative_Z`, `hard_radiative_direction`, and `hard_radiative_target_momentum_eV_c` so a terminal `CUTOFF` row can still represent the hard photon it emitted. The vectors are zero on other rows; photon directions have unit length on event rows. `check_segment_event_contract` checks their shape and correspondence with the electron energy jump when the next row exists.

A transported secondary (a delta ray, or a photon followed as its own track) is a new `electron_id`, not a new field on the primary's rows. Every invariant above is per track, so a secondaries slice adds only a parent link: the parent's `electron_id` and the `flight_id` of the event that created it.

**Downstream transforms.** `_clip_segments_to_cutoff` marks as `CUTOFF` every row a consumer's energy floor shortens or ends. `subdivide_flights` marks the inserted pieces `SUBSTEP` and keeps the parent's event on its last piece, so subdivision changes neither the event sequence nor the incoherent partition.

**Frozen rows** carry no `event_kind`. Each is a whole flight ending in a closing event; which one is not recorded. Hard-event transport requires the midpoint schema.

## Per-electron incident diagnostics

`initial_r_ang`, `initial_v_hat`, `initial_E_keV`, and `initial_t0_ang` carry one row per **sampled** electron, including those that missed a finite footprint and produced no segment. They are the realized phase space at the entrance plane, the sampled counterpart of the specification in [Beam phase space](beam-phase-space.md) and [Longitudinal bunch structure](longitudinal-structure.md), and they are what `beam_metrics.sampled_beam_metrics` reduces to emittance, Twiss parameters, and spot size.

Because misses are retained, these arrays record the incident distribution that was *attempted*, not only the part that entered the crystal.

## Vacuum legs

Grooved runs return non-radiating vacuum flights separately: `vacuum_start_ang`, `vacuum_end_ang`, `vacuum_E_keV`, `vacuum_t_ang`, `vacuum_t0_ang`, and `vacuum_elec_id`. They never enter the material segment sum. They exist because the transport clock advances across a gap even though nothing is emitted there, so a coherent phase reconstruction needs them and an intensity sum must not see them.

## Tallies and normalization

The exit tallies are listed in [Transport geometry](../geometry/transport-geometry.md#termination). All of them count electrons, and all yields in the package are **per incident electron**; the $N_e$ denominator includes missed entries. A run that overruns its sample therefore reports a reduced yield, which is the physical answer.

What the outputs deliberately do **not** carry:

- **no statistical weights.** Every electron counts once: no variance reduction, no splitting, no Russian roulette;
- **no absolute flux.** `bunch_charge_pc` and `rep_rate_hz` are inert to the transport draw and no photons-per-second normalization is applied anywhere. A spectrum is per incident electron until a caller supplies a multiplier;
- **no physical electron count.** A 1 pC bunch is $6.24\times10^{6}$ electrons against a few hundred macro-particles; the transport never conflates the two. Coherent emission does scale with the physical count, so the boundary is tested rather than incidental.

## Optional diagnostics

`collect_diagnostics=True` runs a deterministic, read-only pass after transport and returns fixed-size percentile summaries of per-flight fractional energy loss, relative elastic-hazard change, the left-endpoint-versus-midpoint clock difference, and cutoff overshoot. It consumes no random draws, alters no propagation, and retains no per-flight arrays.

Two matching radiation-side estimators, `cxr_endpoint_resonance_drift` and `brem_endpoint_quadrature_error`, report the spectral consequence of one-point evaluation. None of these are on a default call path; they are instruments for deciding whether a step control is fine enough.

## Checking convergence

Nothing in the output signals whether a run is converged. Two independent axes have to be checked:

- **electron count and seed.** Ordinary Monte Carlo error, falling as $1/\sqrt{N}$. Aggregate observables must be compared with their standard errors, never realization by realization.
- **numerical step refinement.** Refining `max_dE_frac` *decorrelates* trajectories, so per-realization quantities such as flight counts move for reasons that carry no information. Convergence must be read from ensemble means with Monte Carlo errors.

Statistical technique (estimators, error bars, seed handling, and the convergence protocol the validation records use) is collected in the [computation section](../../computation/index.md).

## Validation

`Validation: electron-transport` for the segment semantics and the row-event contract, `transport-midpoint-stopping` and `substep-radiation-invariance` for the midpoint fields and their reduction, `radiation-error-estimators` for the diagnostic estimators, and `beam-phase-space-metrics` for the incident-diagnostic reduction. See the [physics validation ledger](../../validation/physics-validation-ledger.md).
