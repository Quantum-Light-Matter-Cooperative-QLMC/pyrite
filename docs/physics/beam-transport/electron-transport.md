# Electron transport

PyRITE transports independent incident electrons through crystalline or layered matter as piecewise-linear segments. Each segment records position, direction, kinetic energy, path length, material, and elapsed flight time for the radiation kernels.

This page states the transport loop and the propagation rules that control it. The models it calls are documented separately in [Elastic scattering](elastic-scattering.md), [Stopping power and the energy cutoff](stopping-power.md), and [Inelastic scattering and energy-loss straggling](inelastic-scattering-events.md); the beam it starts from in [Beam phase space](beam-phase-space.md) and [Longitudinal bunch structure](longitudinal-structure.md), the boundaries that confine it in [Transport geometry and boundaries](../geometry/transport-geometry.md), and what it hands downstream in [Transport outputs](transport-outputs.md).

## Model

At each material step the transport samples an elastic free path, advances the electron, applies either the mean energy loss or a fluctuation about that mean, and samples an elastic deflection. The default elastic model uses ELSEPA partial-wave tables; Mott/Browning remains selectable. Each layer's collision stopping comes from its SBETHE material table. The opt-in shell soft/hard model also samples discrete inelastic collisions. In a stack, the active layer determines rates, stopping and straggling; flights are truncated at interfaces before continuing with the next medium.

Vacuum legs do not scatter, stop, or radiate, but their distance advances the transport clock. Transport ends when the electron exits permanently, falls below the model cutoff, or exhausts the bounded step budget.

### Stochastic energy loss

In the default continuous inelastic mode, `straggling=False` gives deterministic loss from the layer's SBETHE table, with positive stopping magnitude $S(E)=\lvert dE/ds\rvert$. With `straggling=True`, that loss is replaced by an unrestricted Urban compound-Poisson draw whose mean is scaled to $S(E)$. The construction follows the Geant4 Physics Reference Manual and Bichsel's thin-detector treatment{cite:p}`geant4prm,bichsel1988`:

$$
\Delta E
=n_1E_1+n_2E_2+\sum_{k=1}^{n_3}\epsilon_k,
\qquad n_i\sim\operatorname{Poisson}(s\Sigma_i),
$$

where the continuum marks follow a $1/\epsilon^2$ density from $E_0=10$ eV to the Moller ceiling $T_{\max}=E/2$. The two excitation levels obey $f_1+f_2=1$ and $f_1\ln E_1+f_2\ln E_2=\ln I$; each element's historical channel contribution is scaled so the compound mean matches SBETHE. Consequently,

$$
\langle\Delta E\rangle=S(E)s,
\qquad
\operatorname{Var}(\Delta E)
=s\left(\Sigma_1E_1^2+\Sigma_2E_2^2
        +\Sigma_3E_0T_{\max}\right).
$$

The implementation samples Poisson counts by inverse CDF. A mean above 64 is decomposed into equal independent chunks no larger than 64 and the counts are summed. Poisson additivity therefore preserves the exact count law, including its higher moments and infinite divisibility, without underflowing the recurrence's starting probability. The measured production regime stays below about $\lambda=2.5$ per channel, so chunking is a guard for deliberately oversized direct-sampler steps.

The model is local and unrestricted: it changes the loss distribution without creating or transporting knock-on electrons. When the nominal K-shell level $E_2=10Z^2$ eV is above $T_{\max}$ or has a non-positive logarithmic factor, the two sum rules are re-solved with $f_1=1$ and $E_1=I$. This preserves the mean exactly while leaving a documented high-$Z$, low-energy width limitation. As $s\to0$, every Poisson mean vanishes, so $P(\Delta E=0)\to1$ and both moments vanish linearly. Disabling straggling removes the keys, draws and output field entirely and reproduces the deterministic transport bit-for-bit.

The derivation, domain limits and measured observable response are [`energy-loss-straggling`](../../validation/beam-transport/energy-loss-straggling.md). `Validation: energy-loss-straggling`

## Energy-controlled propagation

`energy_model` selects how one physical flight's energy and clock advance along it. `"frozen"` (the default) is the historical left-endpoint rule: energy, speed, stopping power, and elastic hazard are held at the flight-start value for the whole flight. `Numerics`, campaign profiles and the CLI expose this setting, `max_dE_frac`, and `straggling`.

`"midpoint"` advances the flight with explicit midpoint RK2:

$$
E_{\rm pred}=E_{\rm start}
+\frac{dE}{ds}(E_{\rm start})s,
\qquad
E_{\rm end}=E_{\rm start}
+\frac{dE}{ds}\!\left(\tfrac12(E_{\rm start}+E_{\rm pred})\right)s,
$$

The clock advances by $s/\beta(\tfrac12(E_{\rm start}+E_{\rm end}))$. The update is third-order local and second-order global, one order above the frozen rule. A cutoff-stopped flight separately evaluates the stopping rate at $(E_{\rm start}+E_{\rm cut})/2$ and assigns $E_{\rm end}=E_{\rm cut}$, so the frozen rule's CSDA range overshoot disappears.

With straggling enabled, the random draw replaces both deterministic energy updates: $E_{\rm end}=E_{\rm start}-\Delta E$. `energy_model` still selects the clock's representative energy — the start energy for `"frozen"`, the realized $(E_{\rm start}+E_{\rm end})/2$ for `"midpoint"` — and therefore the returned row schema. The Urban mean is a left-endpoint $S(E_{\rm start})s$ quadrature; `max_dE_frac` controls its energy-drift error.

For coherent emission the midpoint rule is a prerequisite: holding $\beta$ fixed across a flight costs hundreds of radians of emission phase per micron of trajectory, against $10^{-3}$--$10^{-1}$ rad for the midpoint rule on the same flights.

### Physical flights and numerical substeps

**What.** A *physical flight* is the straight path between two physical events: an elastic collision, a geometry event (layer interface, exit face, or groove facet), the energy cutoff, or termination. Under the continuous-slowing-down approximation (CSDA, `straggling=False`) the electron loses energy continuously along that path. `max_dE_frac` $=f$ lets transport split one flight into several output rows, *numerical substeps*, each covering a predicted mean fractional loss of at most $f$. A substep boundary is a quadrature node of the flight's own integrals, not an event: the electron keeps its direction, `flight_id`, and collision budget, advances `substep_id + 1`, and is not deflected.

**Why.** A row gives its consumers one or two point evaluations of quantities that vary along the flight: the stopping power in the energy update, $\beta$ in the clock, the elastic hazard $1/\lambda$ in the collision draw, and the representative energy at which the radiation kernels evaluate their path integrals. Each resulting error is controlled by the flight's fractional loss $\Delta E/E_{\rm start}$, not by its length. Flight length is set by the elastic mean free path, so nothing else bounds that fraction. Most flights are already short in this sense: about 99% lose less than 2% across the validation matrix, but low-energy and thick stopping cases reach a per-flight p99 of 3–10% (`energy-step-convergence`). Two one-point errors survive the midpoint rule at one row per flight:

- the hazard is held at $\lambda(E_{\rm start})$ for the whole flight, while the elastic cross section rises as the electron slows, so unrefined flights are slightly too long, first order in the fractional loss. The bias is measured below Monte Carlo error at 12 000 electrons (`energy-controlled-propagation`);
- the line kernel evaluates one resonance per row, while the resonance drifts as $\beta$ falls, so the unrefined line peak is 2.6–5.7% high and converges under refinement. Bremsstrahlung evaluated at $E_{\rm repr}$ is already a midpoint rule and converges from a much smaller error (`substep-radiation-invariance`).

Substepping does not control coherent emission phase. That clock error is removed by the midpoint rule itself, and a fractional-loss cap on frozen-energy rows misses the phase tolerance at every refinement (`energy-step-convergence`); hence the cap requires `energy_model="midpoint"`.

**How.** Each row starts at $E_{\rm start}$ with remaining optical depth $\tau$; a new flight first draws $\tau=-\ln U$. The row length is the shortest of:

1. the collision distance $\tau\lambda(E_{\rm start})$;
2. the distance to the next geometry event;
3. the cutoff distance, solved with the stopping rate at $(E_{\rm start}+E_{\rm cut})/2$, which also wins an exact tie that involves no geometry event;
4. unless the cutoff already binds, the cap $s_{\rm cap}=fE_{\rm start}/C(E_{\rm start})$, which wins only when strictly shorter.

The row advances energy and clock with the midpoint update above and debits the budget at its own start hazard, $\tau\to\tau-s/\lambda(E_{\rm start})$, clamped at zero. If the cap bound, the next row continues the same flight from the new state. Otherwise the flight closes, a collision scatters, and the next flight redraws $\tau$. The debit is the discretized inversion of $\int_0^{s^*}ds/\lambda(E(s))=-\ln U$, which has no closed form for tabulated $\lambda(E)$.

The cap bounds the left-endpoint prediction $C(E_{\rm start})s/E_{\rm start}$, not the realized loss: the midpoint loss is slightly larger when $C(E)$ rises as the electron slows. Because each binding cap removes close to a fraction $f$, a flight falling from $E_{\rm start}$ to $E_{\rm end}$ yields roughly $\ln(E_{\rm start}/E_{\rm end})/f$ rows, so row count grows as $1/f$ only in flights that actually lose energy. At `max_dE_frac=0` (the default) every row is a whole flight and the draw reduces exactly to the historical one, so frozen runs stay bit-for-bit unchanged.

**Choosing $f$.** Leave it at zero for transport observables and bremsstrahlung yields; their refinement shifts sit inside Monte Carlo error on the validation matrix. Set it when the CXR line peak or lineshape matters at the percent level. `cxr_endpoint_resonance_drift` reports, before any spectrum is computed, how far each flight's resonance moves in units of its own linewidth. In the 25 keV carbon validation cases, $f=10^{-3}$ converged the grouped line peak to $2\times10^{-5}$ for a 3–5× row increase. Refinement decorrelates trajectories, because a changed substep grid moves every later collision, so compare ensemble means with their errors, never single-seed counts. The flight-grouped line reduction that substepped rows need is host-only and rejects `components=True` and layered stacks.

**With straggling**, the cap remains deterministic and is applied **before** the draw using the same mean-rate $s_{\rm cap}$. Choosing a substep from the realized loss would make the partition depend on its increments and invalidate the compound-Poisson partition argument. Each accepted row then receives an Urban draw addressed by `(electron_id, flight_id, substep_id)`. The precise salt and counter construction live in [Random number streams](../../computation/random-streams.md#the-straggling-namespace).

At frozen energy, substep invariance of the loss is exact in distribution for any fixed partition because compound-Poisson increments are infinitely divisible:

$$
\sum_m \operatorname{CP}(s_m\Sigma)
\;\overset{d}{=}\;
\operatorname{CP}\!\left(\sum_m s_m\Sigma\right).
$$

It is not pathwise identity: different substeps use different addresses. Once energy evolves between substeps, the next jump kernel depends on the previous loss. For two short substeps $h_1,h_2$, with jump-intensity measure $\nu_E(d\epsilon)$ and $C(E)=\int\epsilon\nu_E(d\epsilon)$, the mean shift from one frozen unsplit draw contains

$$
h_1h_2\int\left[C(E-\epsilon)-C(E)\right]\nu_E(d\epsilon)
=-h_1h_2C(E)C'(E)+h_1h_2R(E),
$$

where

$$
R(E)=\int\left[C(E-\epsilon)-C(E)+\epsilon C'(E)\right]
\nu_E(d\epsilon).
$$

Urban jump sizes do not shrink with the substep, so $R(E)$ is generally nonzero at the same order as the deterministic quadrature term. The increments are only conditionally independent: the second draw uses $E-X_1$, producing

$$
\operatorname{Cov}(X_1,X_2)
=h_2\operatorname{Cov}\!\left(X_1,C(E-X_1)\right),
$$

which is generally nonzero. Evolving-energy subdivision is therefore a state-dependent jump-process discretization; its measured convergence cannot be reduced to deterministic stopping-power quadrature alone. The regression anchor evaluates the full generator integral independently and resolves the finite-jump remainder rather than testing only the linearized $-CC'$ term.

`(electron_id, flight_id)` is the stable physical key, independent of batching and backend row order; `flight_id` is zero-based and monotonic per electron and `substep_id` is zero-based within its flight. Collision, boundary, and terminal events close a flight; a numerical energy-limit event does not. Each row names its end event in `event_kind`; the codes, the reserved hard-event codes, and the invariants every core satisfies are in [Transport outputs](transport-outputs.md#row-events-and-physical-segments).

### Stochastic cutoff crossing

An Urban loss is nonnegative along the row. Apart from event precedence at an exact tie, first passage through the population cutoff occurs by the row end if and only if

$$
\Delta E\ge E_{\rm start}-E_{\rm cut}.
$$

The implementation defines exact equality at a simultaneous geometry event to belong to geometry; equality without a geometry event counts as a cutoff. Under this explicit convention `n_cutoff_stopped` is exact. The jump position is not retained, so a winning cutoff uses the realized-loss fluid interpolation

$$
s_{\rm cut}
=s\frac{E_{\rm start}-E_{\rm cut}}{\Delta E},
\qquad E_{\rm end}=E_{\rm cut}.
$$

It consumes no extra random number, handles arbitrary overshoot, and reduces algebraically to the deterministic cutoff-distance solve when the fluctuation vanishes. Its limitation is explicit: within the terminal row, spreading the realized loss continuously biases the location relative to the true jump time, by at most one flight length. In a grooved geometry the sampler sees only the material-side distance; if the cutoff is reached there, stopping precedes the facet and no vacuum leg is emitted.

## Initial conditions and geometry

The selected beam supplies kinetic-energy spread, entrance position, direction, and optional bunch timing. These inputs are sampled on independent random streams so enabling an otherwise inert distribution does not perturb transport draws. Coordinate and phase-space conventions are documented in [Beam phase space](beam-phase-space.md), and arrival-time structure in [Longitudinal bunch structure](longitudinal-structure.md).

Finite transverse dimensions, surface tilt, multilayer interfaces, and optional blazed-groove facets constrain crossings. The flat, single-material slab is the compatibility limit. See [Transport geometry and boundaries](../geometry/transport-geometry.md), [Multilayer materials](../materials/multilayer-materials.md), and [Tilt convention](../geometry/tilt-convention.md).

## Outputs and coupling

The full returned schema and its normalization semantics are documented in [Transport outputs](transport-outputs.md).

Segments are Monte Carlo histories, not detector events. The line and bremsstrahlung kernels consume them separately. Numerical substeps within one physical flight are phase-summed; default spectra then add physical-flight and electron intensities under a random-phase approximation. Experimental coherent emission additionally uses segment midpoint times and bunch offsets.

Midpoint transport adds `E_end_keV`, `t_end_ang`, the representative energy `E_repr_keV` $=(E_{\rm start}+E_{\rm end})/2$ that the radiation kernels evaluate their one-point path integrals at, the `flight_id`/`substep_id` identifiers, and the row-end `event_kind`. New fields appear only under the rule that produces them.

When straggling is enabled, `straggle_dE_keV` stores the summed sampled loss per electron. On a cutoff row it includes the sampled overshoot even though the applied row loss ends exactly at $E_{\rm cut}$; this distinction makes the sampler replayable from the recorded history.

`collect_diagnostics=True` adds a deterministic, read-only post-transport pass returning percentile summaries of per-flight fractional loss, elastic-hazard change, clock estimate, and cutoff overshoot. `montecarlo.spectrum` exports two matching radiation-side estimators, `cxr_endpoint_resonance_drift` and `brem_endpoint_quadrature_error`. None of these are on a default call path.

## Assumptions and limits

- independent electrons; no space charge or collective beamline evolution;
- condensed-history local energy loss between elastic scatters; optional Urban straggling samples the loss but does not create secondary particles or an explicit collision cascade;
- refining `max_dE_frac` decorrelates trajectories, so its convergence must be read from ensemble means with Monte Carlo errors, not per-realization counts;
- transport data limited to catalog-supported elements and model energy range;
- no radiation reaction or energy removal from emitted photons;
- material interfaces are sharp and static;
- Monte Carlo convergence must be checked against electron count and seed.

The canonical ledger claim is `electron-transport`; phase-space injection and multilayer behavior have separate rows. The propagation rule is `transport-midpoint-stopping`, the optical-depth collision budget is `energy-controlled-propagation`, the step-control measurements are `energy-step-convergence`, and the diagnostic estimators are `radiation-error-estimators`. Stochastic loss, cutoff and distributional substep semantics are `energy-loss-straggling`. Consult the [validation ledger](../../validation/physics-validation-ledger.md) before scientific use. Implementation owner: `pyrite.montecarlo.transport.simulate_trajectories`.
