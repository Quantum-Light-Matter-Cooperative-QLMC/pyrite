# Issue #623 — axial entrance-zone contract scaffolding

**Status: research-only; `cr-axial-zone-handoff-accounting` remains unverified.**
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
