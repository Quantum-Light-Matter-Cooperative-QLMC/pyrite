# Electron transport

PyRITE transports independent incident electrons through crystalline or
layered matter as piecewise-linear segments. Each segment records position,
direction, kinetic energy, path length, material, and elapsed flight time for
the radiation kernels.

This page is the section overview: it states the transport loop and the
propagation rules that control it. The models it calls are documented separately
in [Elastic scattering](elastic-scattering.md) and
[Stopping power and the energy cutoff](stopping-power.md), the beam it starts
from in [Beam phase space](beam-phase-space.md) and
[Longitudinal bunch structure](longitudinal-structure.md), the boundaries that
confine it in
[Transport geometry and boundaries](../geometry/transport-geometry.md), and what
it hands downstream in [Transport outputs](transport-outputs.md).

## Model

At each material step the transport samples an elastic free path, advances the
electron, applies either the mean energy loss or an Urban fluctuation about that
mean, and samples an elastic deflection. The implementation uses tabulated NIST
SRD 64 Mott/Browning transport data and the per-element Joy--Luo/Berger--Seltzer
stopping-power splice. Compound rates are assembled from element number
densities. In a stack, the active layer determines rates, stopping and
straggling; flights are truncated at interfaces before continuing with the next
medium.

Vacuum legs do not scatter, stop, or radiate, but their distance advances the
transport clock. Transport ends when the electron exits permanently, falls
below the model cutoff, or exhausts the bounded step budget.

### Stochastic energy loss

With `straggling=False` (the default), a material row loses the deterministic
mean $Cs$, where $C=\lvert dE/ds\rvert$ is the spliced stopping magnitude at
the row's start energy and $s$ is its material path length. With
`straggling=True`, that loss is replaced by the unrestricted Urban
compound-Poisson draw

$$
\Delta E
=n_1E_1+n_2E_2+\sum_{k=1}^{n_3}\epsilon_k,
\qquad n_i\sim\operatorname{Poisson}(s\Sigma_i),
$$

where the continuum marks follow a $1/\epsilon^2$ density from
$E_0=10$ eV to the Moller ceiling $T_{\max}=E/2$. The two excitation levels
obey $f_1+f_2=1$ and $f_1\ln E_1+f_2\ln E_2=\ln I$; each element is sampled
independently using its own stopping contribution. Consequently,

$$
\langle\Delta E\rangle=Cs,
\qquad
\operatorname{Var}(\Delta E)
=s\left(\Sigma_1E_1^2+\Sigma_2E_2^2
        +\Sigma_3E_0T_{\max}\right).
$$

These are the analytic Urban moments. For channel means $\lambda<100$, the
implementation samples the Poisson counts by inverse CDF. At
$\lambda\ge100$ it uses a rounded-Gaussian count approximation, so the sampled
law no longer has exact Poisson higher moments or infinite divisibility. The
measured production regime stays below about $\lambda=2.5$ per channel; the
broader function contract remains a validation discrepancy.

The model is local and unrestricted: it changes the loss distribution without
creating or transporting knock-on electrons. When the nominal K-shell level
$E_2=10Z^2$ eV is above $T_{\max}$ or has a non-positive logarithmic factor,
the two sum rules are re-solved with $f_1=1$ and $E_1=I$. This preserves the
mean exactly while leaving a documented high-$Z$, low-energy width limitation.
As $s\to0$, every Poisson mean vanishes, so
$P(\Delta E=0)\to1$ and both moments vanish linearly. Disabling straggling
removes the keys, draws and output field entirely and reproduces the historical
transport bit-for-bit.

The derivation, domain limits and measured observable response are
[`energy-loss-straggling`](../../validation/beam-transport/energy-loss-straggling.md).
`Validation: energy-loss-straggling`

## Energy-controlled propagation

`energy_model` selects how one physical flight's energy and clock advance along
it. `"frozen"` (the default) is the historical left-endpoint rule: energy,
speed, stopping power, and elastic hazard are held at the flight-start value for
the whole flight. `Numerics`, campaign profiles and the CLI expose this setting,
`max_dE_frac`, and `straggling`; their defaults preserve historical case and
dataset identities.

`"midpoint"` advances the flight by the implicit midpoint rule

$$
E_{\rm end}=E_{\rm start}
+\frac{dE}{ds}\!\left(\tfrac12(E_{\rm start}+E_{\rm end})\right)s,
$$

evaluated by one predictor--corrector pass, and advances the clock by
$s/\beta(\tfrac12(E_{\rm start}+E_{\rm end}))$. It is third-order local and
second-order global, one order above the frozen rule. A cutoff-stopped flight
solves its truncation distance for $E_{\rm end}=E_{\rm cut}$ rather than
extrapolating, so the frozen rule's CSDA range overshoot disappears.

With straggling enabled, the random draw replaces both deterministic energy
updates: $E_{\rm end}=E_{\rm start}-\Delta E$. `energy_model` still selects the
clock's representative energy — the start energy for `"frozen"`, the realized
$(E_{\rm start}+E_{\rm end})/2$ for `"midpoint"` — and therefore the returned
row schema. The Urban mean is a left-endpoint $C(E_{\rm start})s$ quadrature;
`max_dE_frac` controls its energy-drift error.

For coherent emission the midpoint rule is a prerequisite rather than an
optimization: holding $\beta$ fixed across a flight costs hundreds of radians of
emission phase per micron of trajectory, against $10^{-3}$--$10^{-1}$ rad for
the midpoint rule on the same flights.

### Physical flights and numerical substeps

`max_dE_frac` caps one row's fractional energy loss and requires
`energy_model="midpoint"`. When that cap binds before any physical event, the
transport emits a row and resumes the same flight: same direction, same
`flight_id`, `substep_id + 1`, no deflection, and no new collision draw.
Numerical substeps are quadrature nodes of a flight's own integrals — never
collision events, independent emitters, or a source of decoherence.

The collision is drawn once per physical flight as an optical depth
$\tau=-\ln U$ and consumed per substep at that substep's own hazard,
$\tau \to \tau-\Delta s/\lambda(E_{\rm substep})$. This is the discretized
inversion of $\int ds/\lambda(E(s))=-\ln U$; tabulated $\lambda(E)$ has no
closed-form inverse. At `max_dE_frac=0` (the default) every iteration is a whole
flight and the draw reduces exactly to the historical one, so frozen runs stay
bit-for-bit unchanged.

With straggling, the cap remains deterministic and is applied **before** the
draw using the mean rate,
$s_{\rm cap}=fE_{\rm start}/C(E_{\rm start})$. Choosing a substep from the
realized loss would make the partition depend on its increments and invalidate
the compound-Poisson partition argument. Each accepted row then receives an
Urban draw addressed by `(electron_id, flight_id, substep_id)`. The precise
salt and counter construction live in
[Random number streams](../../computation/random-streams.md#the-straggling-namespace);
the physics page does not duplicate that plumbing.

At frozen energy, substep invariance of the loss is exact in distribution for
any fixed partition **while every channel remains on the exact-Poisson branch**
and its inverse-CDF loop terminates normally, because compound-Poisson
increments are infinitely divisible:

$$
\sum_m \operatorname{CP}(s_m\Sigma)
\;\overset{d}{=}\;
\operatorname{CP}\!\left(\sum_m s_m\Sigma\right).
$$

It is not pathwise identity: different substeps use different addresses. Once
energy evolves between substeps, the next jump kernel depends on the previous
loss. For two short substeps $h_1,h_2$, with jump-intensity measure
$\nu_E(d\epsilon)$ and $C(E)=\int\epsilon\nu_E(d\epsilon)$, the mean shift from
one frozen unsplit draw contains

$$
h_1h_2\int\left[C(E-\epsilon)-C(E)\right]\nu_E(d\epsilon)
=-h_1h_2C(E)C'(E)+h_1h_2R(E),
$$

where

$$
R(E)=\int\left[C(E-\epsilon)-C(E)+\epsilon C'(E)\right]
\nu_E(d\epsilon).
$$

Urban jump sizes do not shrink with the substep, so $R(E)$ is generally
nonzero at the same order as the deterministic quadrature term. The increments
are only conditionally independent: the second draw uses $E-X_1$, producing a
cross-step covariance. Evolving-energy subdivision is therefore a
state-dependent jump-process discretization; its measured convergence cannot
be reduced to deterministic stopping-power quadrature alone.

`(electron_id, flight_id)` is the stable physical key, independent of batching
and backend row order; `flight_id` is zero-based and monotonic per electron and
`substep_id` is zero-based within its flight. Collision, boundary, and terminal
events close a flight; a numerical energy-limit event does not.

### Stochastic cutoff crossing

An Urban loss is nonnegative along the row. Apart from event precedence at an
exact tie, first passage through the population cutoff occurs by the row end if
and only if

$$
\Delta E\ge E_{\rm start}-E_{\rm cut}.
$$

The implementation gives an exact equality to a simultaneous geometry event;
geometry wins that tie, while equality without a geometry event counts as a
cutoff. Under this convention `n_cutoff_stopped` is exact. The jump position is
not retained, so a winning cutoff uses the realized-loss fluid interpolation

$$
s_{\rm cut}
=s\frac{E_{\rm start}-E_{\rm cut}}{\Delta E},
\qquad E_{\rm end}=E_{\rm cut}.
$$

It consumes no extra random number, handles arbitrary overshoot, and reduces
algebraically to the deterministic cutoff-distance solve when the fluctuation
vanishes. Its limitation is explicit: within the terminal row, spreading the
realized loss continuously biases the location relative to the true jump time,
by at most one flight length. In a grooved geometry the sampler sees only the
material-side distance; if the cutoff is reached there, stopping precedes the
facet and no vacuum leg is emitted.

## Initial conditions and geometry

The selected beam supplies kinetic-energy spread, entrance position,
direction, and optional bunch timing. These inputs are sampled on independent
random streams so enabling an otherwise inert distribution does not perturb
transport draws. Coordinate and phase-space conventions are documented in
[Beam phase space](beam-phase-space.md), and arrival-time structure in
[Longitudinal bunch structure](longitudinal-structure.md).

Finite transverse dimensions, surface tilt, multilayer interfaces, and optional
blazed-groove facets constrain crossings. The flat, single-material slab is the
compatibility limit. See
[Transport geometry and boundaries](../geometry/transport-geometry.md),
[Multilayer materials](../materials/multilayer-materials.md), and
[Tilt convention](../geometry/tilt-convention.md).

## Outputs and coupling

The full returned schema and its normalization semantics are documented in
[Transport outputs](transport-outputs.md).

Segments are Monte Carlo histories, not detector events. The line and
bremsstrahlung kernels consume them separately. Default spectra add segment and
electron intensities incoherently; experimental coherent emission additionally
uses segment midpoint times and bunch offsets.

`E_keV` and `t_ang` keep flight-start semantics under both propagation rules and
are compatibility aliases of `E_start_keV` and `t_start_ang`; `elec_id` is a
compatibility alias of `electron_id`. Midpoint transport adds `E_end_keV`,
`t_end_ang`, the representative energy `E_repr_keV` $=(E_{\rm start}+E_{\rm
end})/2$ that the radiation kernels evaluate their one-point path integrals at,
and the `flight_id`/`substep_id` identifiers. New fields appear only under the
rule that produces them, so the frozen schema is never partially extended.
Checkpoints store reduced spectra rather than raw rows, so no stored-result
migration follows from the added fields.

When straggling is enabled, `straggle_dE_keV` stores the summed sampled loss per
electron. On a cutoff row it includes the sampled overshoot even though the
applied row loss ends exactly at $E_{\rm cut}$; this distinction makes the
sampler replayable from the recorded history.

`collect_diagnostics=True` adds a deterministic, read-only post-transport pass
returning percentile summaries of per-flight fractional loss, elastic-hazard
change, clock estimate, and cutoff overshoot. `montecarlo.spectrum` exports two
matching radiation-side estimators, `cxr_endpoint_resonance_drift` and
`brem_endpoint_quadrature_error`. None of these are on a default call path.

## Assumptions and limits

- independent electrons; no space charge or collective beamline evolution;
- condensed-history local energy loss between elastic scatters; optional Urban
  straggling samples the loss but does not create secondary particles or an
  explicit collision cascade;
- refining `max_dE_frac` decorrelates trajectories, so its convergence must be
  read from ensemble means with Monte Carlo errors, not per-realization counts;
- transport data limited to catalog-supported elements and model energy range;
- no radiation reaction or energy removal from emitted photons;
- material interfaces are sharp and static;
- Monte Carlo convergence must be checked against electron count and seed.

The canonical ledger claim is `electron-transport`; phase-space injection and
multilayer behavior have separate rows. The propagation rule is
`transport-midpoint-stopping`, the optical-depth collision budget is
`energy-controlled-propagation`, the step-control measurements are
`energy-step-convergence`, and the diagnostic estimators are
`radiation-error-estimators`. Stochastic loss, cutoff and distributional
substep semantics are `energy-loss-straggling`. Consult the [validation
ledger](../../validation/physics-validation-ledger.md) before scientific use.
Implementation owner: `pyrite.montecarlo.transport.simulate_trajectories`.
