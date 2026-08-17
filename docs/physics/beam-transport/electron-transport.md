# Electron transport

PyRITE transports independent incident electrons through crystalline or
layered matter as piecewise-linear segments. Each segment records position,
direction, kinetic energy, path length, material, and elapsed flight time for
the radiation kernels.

## Model

At each material step the transport samples an elastic free path, advances the
electron, applies continuous energy loss, and samples an elastic deflection.
The implementation uses tabulated NIST SRD 64 Mott/Browning transport data and
a Joy--Luo stopping-power model. Compound rates are assembled from element
number densities. In a stack, the active layer determines rates and stopping;
flights are truncated at interfaces before continuing with the next medium.

Vacuum legs do not scatter, stop, or radiate, but their distance advances the
transport clock. Transport ends when the electron exits permanently, falls
below the model cutoff, or exhausts the bounded step budget.

## Energy-controlled propagation

`energy_model` selects how one physical flight's energy and clock advance along
it. `"frozen"` (the default) is the historical left-endpoint rule: energy,
speed, stopping power, and elastic hazard are held at the flight-start value for
the whole flight. It is a `simulate_trajectories` argument only; campaign
profiles and the CLI still run every case under the frozen default.

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

`(electron_id, flight_id)` is the stable physical key, independent of batching
and backend row order; `flight_id` is zero-based and monotonic per electron and
`substep_id` is zero-based within its flight. Collision, boundary, and terminal
events close a flight; a numerical energy-limit event does not.

## Initial conditions and geometry

The selected beam supplies kinetic-energy spread, entrance position,
direction, and optional bunch timing. These inputs are sampled on independent
random streams so enabling an otherwise inert distribution does not perturb
transport draws. Coordinate and phase-space conventions are documented in
[Beam phase space](beam-phase-space.md).

Finite transverse dimensions, surface tilt, multilayer interfaces, and optional
blazed-groove facets constrain crossings. The flat, single-material slab is the
compatibility limit. See [Multilayer materials](../materials/multilayer-materials.md)
and [Tilt convention](../geometry/tilt-convention.md).

## Outputs and coupling

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

`collect_diagnostics=True` adds a deterministic, read-only post-transport pass
returning percentile summaries of per-flight fractional loss, elastic-hazard
change, clock estimate, and cutoff overshoot. `montecarlo.spectrum` exports two
matching radiation-side estimators, `cxr_endpoint_resonance_drift` and
`brem_endpoint_quadrature_error`. None of these are on a default call path.

## Assumptions and limits

- independent electrons; no space charge or collective beamline evolution;
- condensed-history continuous stopping between elastic scatters; energy-loss
  straggling is unmodelled, which biases the mean arrival time and not only its
  variance, at a level above the numerical tolerance the midpoint rule reaches;
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
`radiation-error-estimators`. Consult the [validation
ledger](../../validation/physics-validation-ledger.md) before scientific use.
Implementation owner: `pyrite.montecarlo.transport.simulate_trajectories`.
