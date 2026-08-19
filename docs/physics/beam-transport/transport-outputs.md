# Transport outputs

What a transport run hands to the radiation kernels, and what those quantities do
and do not mean. This is the interface every spectrum in the package is built on,
so its semantics are part of the physical model rather than an implementation
detail.

## Segments are histories, not events

A segment is one straight flight of one electron through one material. It is
**not** a detector event, not a photon, and not an independent emitter. The line
and bremsstrahlung kernels each read the same segment table and reduce it their
own way; nothing in transport anticipates either.

Two consequences follow immediately:

- segment counts have no physical normalization of their own — refining the
  numerical step raises the row count without changing any observable;
- the physically meaningful key is `(electron_id, flight_id)`, the physical
  flight, which is stable under batching and independent of row order.

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

`E_keV`, `t_ang`, and `elec_id` are compatibility aliases of the canonical
spellings and keep flight-start semantics under every propagation rule. Nothing
in the schema is reinterpreted when a new rule is selected.

### Fields added by midpoint propagation

`energy_model="midpoint"` adds `E_end_keV`, `t_end_ang`, the representative energy

```{math}
:label: eq-outputs-representative-energy

E_{\rm repr} = \tfrac12\left(E_{\rm start} + E_{\rm end}\right),
```

and the `flight_id` / `substep_id` identifiers. {eq}`eq-outputs-representative-energy`
is the point at which the radiation kernels evaluate their one-point path
integrals, turning each row from a left-endpoint into a midpoint rule.

New fields appear **only** under the rule that produces them. The frozen schema is
never partially extended, so a consumer that finds `E_repr_keV` can rely on the
rest of the midpoint fields being present, and one that does not can rely on
every row being a whole physical flight.

## Per-electron incident diagnostics

`initial_r_ang`, `initial_v_hat`, `initial_E_keV`, and `initial_t0_ang` carry one
row per **sampled** electron, including those that missed a finite footprint and
produced no segment. They are the realized phase space at the entrance plane —
the sampled counterpart of the specification in
[Beam phase space](beam-phase-space.md) and
[Longitudinal bunch structure](longitudinal-structure.md) — and they are what
`beam_metrics.sampled_beam_metrics` reduces to emittance, Twiss parameters, and
spot size.

Because misses are retained, these arrays also record the incident distribution
that was *attempted*, not just the part that entered the crystal.

## Vacuum legs

Grooved runs return non-radiating vacuum flights separately: `vacuum_start_ang`,
`vacuum_end_ang`, `vacuum_E_keV`, `vacuum_t_ang`, `vacuum_t0_ang`, and
`vacuum_elec_id`. They never enter the material segment sum. They exist because
the transport clock advances across a gap even though nothing is emitted there,
so a coherent phase reconstruction needs them and an intensity sum must not see
them.

## Tallies and normalization

The exit tallies are listed in
[Transport geometry](../geometry/transport-geometry.md#termination). All of them
count electrons, and all yields in the package are **per incident electron** — the
$N_e$ denominator includes missed entries. A run that overruns its sample
therefore reports a reduced yield, which is the physical answer.

What the outputs deliberately do **not** carry:

- **no statistical weights.** Every electron counts once. There is no variance
  reduction, no splitting, no Russian roulette, and consequently no weight
  bookkeeping to get wrong;
- **no absolute flux.** `bunch_charge_pc` and `rep_rate_hz` are inert to the
  transport draw and no photons-per-second normalization is applied anywhere. A
  spectrum is per incident electron until a caller supplies a multiplier;
- **no physical electron count.** A 1 pC bunch is $6.24\times10^{6}$ electrons
  against a few hundred macro-particles; the transport never conflates the two.
  Coherent emission genuinely scales with the physical count, so this is a
  deliberate, tested boundary rather than an oversight.

## Optional diagnostics

`collect_diagnostics=True` runs a deterministic, read-only pass after transport
and returns fixed-size percentile summaries of per-flight fractional energy loss,
relative elastic-hazard change, the left-endpoint-versus-midpoint clock
difference, and cutoff overshoot. It consumes no random draws, alters no
propagation, and retains no per-flight arrays.

Two matching radiation-side estimators, `cxr_endpoint_resonance_drift` and
`brem_endpoint_quadrature_error`, report the spectral consequence of one-point
evaluation. None of these are on a default call path; they are instruments for
deciding whether a step control is fine enough, not part of any result.

## Checking convergence

Nothing in the output signals whether a run is converged. Two independent axes
have to be checked, and they behave differently:

- **electron count and seed.** Ordinary Monte Carlo error, falling as
  $1/\sqrt{N}$. Aggregate observables must be compared with their standard
  errors, never realization by realization.
- **numerical step refinement.** Refining `max_dE_frac` *decorrelates*
  trajectories, so per-realization quantities such as flight counts move for
  reasons that carry no information. Convergence must be read from ensemble means
  with Monte Carlo errors.

Statistical technique — estimators, error bars, seed handling, and the
convergence protocol the validation records use — is collected in the
[computation section](../../computation/index.md).

## Validation

`Validation: electron-transport` for the segment semantics,
`transport-midpoint-stopping` and `substep-radiation-invariance` for the
midpoint fields and their reduction, `radiation-error-estimators` for the
diagnostic estimators, and `beam-phase-space-metrics` for the incident-diagnostic
reduction. See the
[physics validation ledger](../../validation/physics-validation-ledger.md).
