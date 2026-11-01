# Issue #623 — axial entrance-zone contract scaffolding

**Status: research-only; `cr-axial-zone-handoff-physical` remains unverified.**
The associated code does not enable axial channeling in standard PyRITE runs.
It adds fail-closed, testable numerical contracts for physical outgoing-flux
accounting, candidate lab-frame launch conversion and positive expected-value
photon assembly. These are *not* a substitute for the still-missing physical
matching-surface derivation or independent convergence evidence.

## Trace and depth budget

`axial_zone_handoff.AxialFluxAccount` enforces the incident-normalized partition

$$w_{\rm retained}+w_{\rm physical\ outgoing}+w_{\rm unresolved}=1.$$

All weights are finite and nonnegative. TDS finite-grid sinks, FFT aliasing,
periodic wrap and above-barrier state occupation **never** count as physical
outgoing merely because the state disappeared from the represented grid.
An external physics model must independently supply and justify each weight.

`require_axial_depth_budget` refuses missing evidence, mismatched cases, depths
above a measured envelope or excessive conservative upper error bounds:

$$\epsilon_{\rm return}^{+}+\epsilon_{\rm unresolved}^{+}+
\epsilon_{\rm source}^{+}\leq\epsilon_{\rm limit}.$$

`AxialDepthEvidence` records a case key, evidence identifier, depth cap and
those three bounds. A constructed Python object is not itself experimental
validation. The currently published Si [001] pilot in #623 is insufficient
for issuing eligible case evidence. Production axial transport must remain
refused until this gate can be integrated ahead of work and writes.

## Correlated outgoing candidate (not a native launch)

An external *physical* outgoing instrument first supplies a state
`phi_out` with norm $\int|\phi_{out}|^2dA=w_{\rm physical\ outgoing}$.
`propose_correlated_outgoing` converts it to the existing finite-Weyl sampler's
discrete normalization via `phi_out * sqrt(grid.cell_area_ang2)` and samples
the joint $(x,y,k_x,k_y)$ law using the existing per-incident Philox stream.
This retains position/momentum correlations; drawing marginals independently
would not.

Scattering-channel averaging generally produces a mixed outgoing density,
not a single wavefunction. The research helpers
`axial_phase_space.density_phase_space_weights` and
`sample_density_phase_space` accept that density directly in the discrete
position basis. For a normalized window and $M$ grid sites they evaluate

$$q_{ab}=M^{-1}\langle g_{ab}|\rho_{out}|g_{ab}\rangle,\qquad
\sum_{ab}q_{ab}=\operatorname{tr}\rho_{out}.$$

The sampler draws position from the window-smoothed density diagonal, then
momentum from the full localized density, retaining off-diagonal coherence.
It uses the same incident-addressed generator and two conditional draws;
no random pure-state decomposition is needed. A supplied coordinate-space
density must first be multiplied by `grid.cell_area_ang2`; a momentum-space
gain must first be transformed to the discrete position basis without
wrapping omitted modes. A gain density per angstrom yields measurement
weights per angstrom, which still require a physical depth/event law.
The dense reference refuses inputs exceeding its working-memory budget.
It does not select the outgoing instrument, resolve periodic images, supply
stopping/clocks, or connect to native launch/scoring/persistence. Standard
axial runs remain refused under #624.

For a sampled periodic pixel $(i,j)$ and FFT order $(m,n)$, supplied unwrapped
transverse origin $(x_0,y_0)$ and carrier $\mathbf{k}_t$,

$$\mathbf r_c=(x_i+x_0,y_j+y_0,z),\quad
\mathbf{k}_\perp=(k_{x,m}+k_{t,x},k_{y,n}+k_{t,y}).$$

With right-handed, orthonormal crystal-to-lab matrix $R$, incident lab origin
$r_0$ and on-shell momentum $p=\sqrt{E(E+2m_ec^2)}/(\hbar c)$,

$$\mathbf r_{lab}=r_0+R\mathbf r_c,\qquad
\hat{v}_{lab}=R\left(k_x/p,k_y/p,\sqrt{1-|\mathbf{k}_\perp|^2/p^2}\right).$$

The function refuses nonphysical or above-budget paraxial momenta, mismatched
outgoing norm, invalid coordinates/frames/identities or nonmonotonic clocks.
It returns incident ID, depth, energy in keV, both direction components,
lab position, **caller-provided** clock (`t_ang`, with c=1), `t0_ang` and the
unchanged native primary stream key. The supplied entrance origin must already
resolve the periodic-image ambiguity. Energy after stopping and clock at the
actual surface must come from an independently validated propagation; neither
is guessed from depth here. No `LaunchState` conversion is provided, because
nothing here establishes that the outgoing state was selected physically.

## Depth-resolved supplied jump instrument

`axial_jump_instrument.absorbing_axial_model` is a bounded CPU reference for
a **supplied**, constant finite absorbing model. It accepts a retained
Hamiltonian $H$, retained recycling jumps $A_a$, represented outgoing jumps
$B_b$, and a positive unresolved effect $E_O$. The outgoing jump rows must
already use the C-order discrete position basis of the chosen measurement
grid. These differential maps contain information absent from a total loss
effect; they cannot be reconstructed by relabelling the production grid sink.

The retained generator uses the loss from those same maps:

$$
\Gamma=\sum_a A_a^\dagger A_a+\sum_b B_b^\dagger B_b+E_O,
\qquad
\partial_z\rho=-\frac{i}{\hbar v}[H,\rho]
+\sum_a A_a\rho A_a^\dagger-\frac12\{\Gamma,\rho\}.
$$

`model.at_depth(z)` computes retained density, instantaneous outgoing gain
$\sigma_F(z)=\sum_b B_b\rho(z)B_b^\dagger$, integrated outgoing density
$Q_F(z)=\int_0^z\sigma_F(s)ds$, and integrated unresolved weight
$U_O(z)=\int_0^z\operatorname{tr}[E_O\rho(s)]ds$. An augmented matrix
exponential propagates $\rho(z)$ together with the occupation integral
$X(z)=\int_0^z\rho(s)ds$, without depth quadrature or an inverse of the
possibly singular retained generator; then $Q_F=\sum_b B_bXB_b^\dagger$ and
$U_O=\operatorname{tr}(E_OX)$. Dense work therefore scales with the retained
dimension $n$ as $(2n^2)^2$, not with the outgoing grid size $m$; only the
returned $m\times m$ snapshots grow with $m$. The occupation block is scaled
by $\kappa=\lVert\Gamma\rVert_2$ so its absolute exponential error does not
grow with depth for weak losses. Densities are Hermitized, and eigenvalues
within the propagator error bound $\sim64\,\epsilon\,n
(1+\lVert\mathcal L\rVert_1 z)$ times the incident weight are clipped to
zero; larger negative eigenvalues refuse. Without this, physical phases
$\lVert H\rVert z/\hbar v\sim10^4$ made the sampler refuse on roundoff. It preserves
$\operatorname{tr}\rho(z)+\operatorname{tr}Q_F(z)+U_O(z)
=\operatorname{tr}\rho(0)$, including a subunit entrance weight.
Retained recycling is retained gain, never an outgoing tally.

`sample_absorbing_outgoing(model, window, max_depth_ang, rng)` conditions on
an outgoing event within the depth window. It samples depth from the
cumulative outgoing trace and then measures the **instantaneous** mixed gain
at that depth. The correlated law is

$$
\Pr(dz,a,b\mid F\text{ by }Z)
=\frac{M^{-1}\langle g_{ab}|\sigma_F(z)|g_{ab}\rangle}
{\operatorname{tr}Q_F(Z)}\,dz.
$$

Sampling depth independently of the integrated outgoing density would lose
correlations. One depth draw and two conditional phase-space draws use the
caller-retained incident-addressed `phase_space_stream(seed, incident_id)`.
The returned `outgoing_weight` is the dimensionless incident weight in the
depth window; `rate_inv_ang` is the local outgoing rate, not another event
weight. A zero outgoing weight refuses before any draw. The absolute root
tolerance requires convergence for rapidly varying models. A local rate below
$10^{-10}$ of the window-mean rate at the sampled root refuses rather than
measuring propagator noise. Dense memory,
nonpositive inputs and nonfinite assembled operators are refused.

The [independent finite-model derivation](../../validation/channeling/cr-axial-absorbing-jump-instrument.md)
and analytic/ODE regressions validate this specified absorbing algebra. They
do not establish the material jump maps, completeness of the unresolved
effect, a physical retained surface, omitted-return bounds, stopping energy,
clock continuity or periodic-image choice. No native launch, photon scoring
or #624 production admission is provided.

## Incident-normalized photons and radiation ownership

`axial_source_assembly.integrate_incident_axial_source` accepts a *supplied*
nonnegative photon rate $q(z,E,\Omega)$ in
photons/(eV sr Å **incident electron**), a spectral absorption coefficient
$\mu(E)$ [Å$^{-1}$], and an azimuth- and depth-dependent path-to-exit
$L(z,\Omega)$ [Å]. Its positive trapezoidal integration is

$$\frac{d^2N}{dE\,d\Omega}=\int
q(z,E,\Omega)e^{-\mu(E)L(z,\Omega)}dz,$$

returned in photons/(eV sr incident electron). It never renormalizes by the
MCWF survivor count. There is no synthetic line broadening or frame rotation;
those must already be represented in input $q$. Finite/nonnegative shape,
energy and unit limits and bounded temporary memory are checked.

`AxialRadiationOwnership` rejects any *declared* overlap with native radiation
components. Tags must correspond to physical processes, not code labels.
Existing finite-grid QRT sources contain free/free transitions and therefore
cannot be claimed disjoint from native CBS by naming them “channeling radiation.”
This is not proof that a user-defined process partition is physically complete.

## Still required for closing #623

1. Derive the actual physical outgoing projector/instrument and depth-resolved
   outgoing states with transfer outside represented FFT/supercell grids.
2. Measure selected-cut clearance, coherent cross-cut return flux, unresolved
   weight, convergence under step, FFT grid and supercell changes, **including
   electronic kicks**, and determine case-specific upper-error depth budgets.
3. Validate stopping-energy and clock continuity, periodic-image resolution,
   on-shell lab directions, and native launch/stream-key integration.
4. Derive and validate the complete emitting current/subspace, recoil-sector
   closure, lab polarization/azimuth mapping, line shapes, source normalization
   and the exactly-once split from native sources.
5. Add full-case early refusal, independent reference tests, fresh-context
   source-to-code review before advancing the claim to `anchored`. Human
   sign-off is tracked separately in #277 and does not gate that status.

## Tests

```sh
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/montecarlo/test_channeling_axial_zone_handoff.py \
  tests/montecarlo/test_channeling_axial_source_assembly.py \
  tests/montecarlo/test_channeling_axial_phase_space.py
```

The first two cover the accounting, candidate conversion and supplied-source
assembly contracts; the last is the finite-Weyl oracle. Passing these checks
does not validate a physical outgoing instrument or material photon source.
