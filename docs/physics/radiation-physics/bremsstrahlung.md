# Bremsstrahlung

PyRITE's default `bremsstrahlung_model="auto"` uses the direction-resolved BremsLib double differential cross section when tables are available for every layer element. If any are unavailable, it warns and uses the isotropic Livermore Evaluated Electron Data Library (EEDL) {cite:p}`perkins1991eedl`; either source can also be selected explicitly. Both score the incoherent background along transported electron segments. The continuum shares the transport segments, escape geometry, and per-electron normalization with the [line kernel](coherent-radiation.md), but can use a separate photon-energy grid. Combining line and continuum requires compatible energy coordinates and density conventions; see [spectral observables](spectral-observables.md).

## EEDL fallback: MF=23/MT=527 and MF=26/MT=527

ENDF-6 organizes evaluated data by **file number** (`MF`, the kind of data) and **reaction number** (`MT`, the physical interaction). The pair therefore names a section of the EEDL evaluation; it is not a fitted parameter:

- `MF=23` contains integrated photo-atomic and electro-atomic cross sections. Its `MT=527` section is the total electro-atomic bremsstrahlung cross section {math}`\sigma(T)` as a function of incident-electron energy, tabulated in barns.
- `MF=26` contains secondary-particle energy/angle distributions. Its matching `MT=527` section describes the products of the same bremsstrahlung interaction. PyRITE uses the first subsection, whose `ZAP=0`, `LAW=1`, `LANG=1`, `NA=0` records give an isotropic tabulated photon-energy probability density. The second subsection uses `LAW=8` to report the outgoing electron's average energy loss and is not needed by the photon spectrum estimator.

The two sections are used together because they supply complementary pieces: MF=23 fixes the total probability of a bremsstrahlung event, while MF=26 fixes the conditional photon-energy shape. Their product is the required differential cross section. The section definitions follow ENDF-6 {cite:p}`trkov2018endf6`.

## EEDL cross section and spectrum

For element {math}`Z`, incident electron kinetic energy {math}`T`, and photon energy {math}`k`, the EEDL energy-differential cross section is

```{math}
\frac{d\sigma_Z}{dk}(T,k) = \sigma_Z^{23,527}(T)\,P_Z^{26,527}(k\mid T),
\qquad \int_0^T P_Z(k\mid T)\,dk=1.

```

`endf-parserpy` reads both sections from the checksum-pinned packaged `EEDL.endf`. PyRITE accepts the tape only when the total and photon tables are finite, ordered, non-negative, and declare the supported ENDF lin-lin laws. The MF=23 values are converted from barns to cm². Each native MF=26 photon density is normalized after parsing to remove only source-record rounding error.

The tape declares lin-lin interpolation for the total cross section, the secondary photon-energy axis (`LEP=2`), and the incident-energy panels (`INT=2`). PyRITE follows the first two, but not the declared Cartesian law between incident-energy panels. EEDL gives photon spectra at only 8–10 decade-spaced incident energies per element (carbon: 14.1 keV, 251 keV, 1.19 MeV and 12.2 MeV between 10 keV and 100 GeV). Interpolating adjacent panels at a fixed absolute photon energy leaves every {math}`k` above the lower panel's endpoint with only the upper panel's share, so the spectrum collapses towards {math}`k\to T`. At 30 keV in carbon it fell to 0.09–0.15 of Seltzer–Berger for {math}`k/T\ge0.5` (#174, `brem-source-comparison`).

PyRITE instead uses ENDF unit-base interpolation {cite:p}`trkov2018endf6`. Panel {math}`i`, with photon range {math}`[a_i,b_i]`, maps onto the reduced variable {math}`x=(k-a_i)/(b_i-a_i)\in[0,1]`, where its density becomes the unit-area {math}`q_i(x)=(b_i-a_i)P_i(k)`. Sub-panels are inserted at 32 geometrically spaced incident energies per decade:

```{math}
q(x\mid T) = (1-w)\,q_i(x) + w\,q_{i+1}(x),\qquad
w=\frac{\ln(T/T_i)}{\ln(T_{i+1}/T_i)},\qquad
[a,b](T) = [a_i,b_i] + \frac{T-T_i}{T_{i+1}-T_i}\bigl([a_{i+1},b_{i+1}]-[a_i,b_i]\bigr).

```

The photon range is interpolated linearly in {math}`T`, so {math}`b=T` because every EEDL panel ends at its incident energy. The shape weight is linear in {math}`\ln T`, the spacing of the native panels. Against Seltzer–Berger this weight is several times more accurate between panels than one linear in {math}`T`. All panels share the union of the native {math}`x` nodes, which represents each native panel exactly and keeps every mixture piecewise linear with unit area. Native panels are reproduced unchanged.

Between adjacent sub-panels the runtime interpolation stays Cartesian at fixed photon energy. Each sub-panel is held at its endpoint density up to the next sub-panel's endpoint, so the mixture stays continuous for {math}`k` between them. The implementation then imposes {math}`0<k\le T` and renormalizes the surviving piecewise-linear density exactly, so integrating the differential cross section over the physical photon range recovers the interpolated MF=23 total.

Against Seltzer–Berger, for the 24 catalogue elements from 10 keV to 1 MeV, the result is:

- within 0.96–1.34 pointwise for {math}`0.05\le k/T\le0.95`;
- within 0.98–1.81 for {math}`k/T>0.95`;
- within 0.99–1.07 in the radiative first moment.

The remaining error is panel sparsity, not interpolation.

The ENDF-6 electro-atomic format describes the bremsstrahlung photon subsection as an isotropic, angle-independent tabulated spectrum (`LAW=1`, `LANG=1`, `NA=0`) {cite:p}`trkov2018endf6`. PyRITE uses this isotropic angular model.

## Optional Bethe--Heitler backend

An analytic alternative is available with `cross_section_model="bethe-heitler"` on `mc_brem_spectrum`. It is the nonrelativistic, unscreened Born Bethe--Heitler form with the {cite:t}`elwert1939` correction, using relativistic momenta as a weakly-relativistic extension {cite:p}`kochmotz1959`:

```{math}
\frac{d\sigma}{dk}=\frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2}\ln\!\left(\frac{p_i+p_f}{p_i-p_f}\right)
\frac{\beta_i}{\beta_f}
\frac{1-e^{-2\pi\alpha Z/\beta_i}}{1-e^{-2\pi\alpha Z/\beta_f}}.

```

Momenta are carried in units of {math}`m_ec`, built from the exact relativistic relation {math}`p=\sqrt{T(T+2m_ec^2)}/(m_ec^2)` and {math}`\beta=p/(1+T/m_ec^2)`, with {math}`T_f=T_i-k`. This backend is also the fallback when EEDL lacks an element or incident-energy range. Every fallback emits a `RuntimeWarning` that identifies the missing coverage. Malformed or checksum-mismatched EEDL data are not treated as missing coverage and remain hard errors.

## Default BremsLib direction-resolved backend

`cross_section_model="bremslib"` replaces isotropic emission with the BremsLib v2.0 double differential cross section {cite:p}`poskus2025bremslib`, taking the photon-energy spectrum from the same evaluation. BremsLib tabulates, per element, the scaled single differential cross section (SDCS) {math}`\chi(T_1,x)=(k/Z^2)\,d\sigma/dk` in mb and, at every grid node, the scaled double differential cross section (DDCS) {math}`(k/Z^2)\,d^2\sigma/(dk\,d\Omega)` in mb/sr over the photon emission angle {math}`\theta` measured from the incident electron direction, with {math}`x=k/T_1`. The grids are {math}`T_1` from 10 eV to 30 MeV (ratios up to 1.33), 13 values of {math}`x` from 0 to just below 1, and 181--441 angles refined towards the forward peak. The cross sections PyRITE evaluates are

```{math}
\frac{d^2\sigma_Z}{dk\,d\Omega}(T,k,\theta)=\frac{Z^2}{k}\,\chi(T,x)\,S(T,x,\theta),
\qquad
\frac{d\sigma_Z}{dk}=\frac{Z^2}{k}\,\chi(T,x),
\qquad
\int_{4\pi}S\,d\Omega=1,

```

where the shape function {math}`S` is the DDCS divided by its own solid-angle integral, as the library defines it. Taking the shape and the spectrum from one evaluation keeps {math}`S` normalized against its own parent SDCS; PyRITE therefore never combines the BremsLib shape with the EEDL spectrum.

The interpolation is PyRITE's own. It is not a port of the vendor's `Interpolate_DCS` (a GPL-3 weighted least-squares spline fit), and is built from the library's published grids:

1. **Nodes.** Each node's DDCS is linear in {math}`\theta` between tabulated angles. It is rescaled so that the exact solid-angle integral of that linear interpolant, {math}`2\pi\sum_i\int_{\theta_i}^{\theta_{i+1}}f(\theta)\sin\theta\,d\theta`, equals the node's SDCS. All node grids are subsets of the finest one, so they are resampled onto it without changing any interpolant.
2. **Incident-energy refinement.** Backward emission falls steeply and convexly with {math}`T_1`, so linear interpolation across a library interval overshoots it. At load, each interval is split into four in {math}`\ln T_1`. Each sub-node's shape is the weighted geometric mean of its neighbours' shapes, renormalized to unit integral, and its SDCS is interpolated log-log. Library nodes are kept bit for bit.
3. **Evaluation.** The scaled DDCS is interpolated linearly in {math}`\theta`, in {math}`x=k/T` and in {math}`\ln T` on the refined grid. Each step is a convex combination of nodes whose angular integrals equal their SDCS, so the {math}`4\pi` integral of the interpolated DDCS is exactly the identically interpolated SDCS. The energy spectrum is recovered by construction, not by a separate renormalization.

Between the last node, {math}`x_{\rm top}` (0.99, {math}`1-50\,{\rm eV}/T_1`, or 0.9999 depending on {math}`T_1`, per the library manual), and the kinematic tip, the top node's value is held. Above {math}`k=T` the cross section is zero.

Tables come from `pyrite.xsgen`. The released catalogue tables install with `pyrite tables fetch bremslib`, and other elements are generated from a local BremsLib checkout. Because the physics core does not import `pyrite.xsgen`, a driver resolves them with `pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables(elements)` and passes them as `bremslib_tables`. Fallback coverage:

- With explicit `"bremslib"`, a released element whose table has not been fetched is an error that names the fetch command.
- An element with no table, or a segment energy outside the table's range, warns and uses isotropic EEDL for only that element or those segments.

Runs select the continuum through `Numerics.bremsstrahlung_model` (also `Settings`, profile numerics and `pyrite profile numerics set --bremsstrahlung-model`). The default `"auto"` uses BremsLib for a case when a table resolves for every element of its crystal and absorber layers, and otherwise warns and falls back to EEDL; `"bremslib"` requires the tables and `"eedl"` selects the packaged evaluation. A case records only the resolved choice, so `dataset_identity`, `case_content_key` and the provenance marker (`BREMSSTRAHLUNG_BREMSLIB_MODEL` or `BREMSSTRAHLUNG_MODEL`) name what actually ran, and the tables' manifest digests join the xsgen identity markers. The default followed the source comparison and recommendation in `brem-source-comparison` (#86): BremsLib is the recommended source for the 24 catalogue elements from 10 keV to 30 MeV, with its electron–electron shortfall at low Z and MeV energies recorded there. The low-level `mc_brem_spectrum` takes the same `cross_section_model="auto"` default: BremsLib when the segments carry directions `v_hat` and a table resolves for every composition element (the supplied `bremslib_tables`, else the installed release through the driver-side loader), otherwise a warned EEDL fallback. The runner pins the case's already-resolved choice, so `"auto"` never re-resolves it. Public archive hosting is deferred; archives are distributed internally until then.

## Coupled soft/hard radiative transport

The opt-in coupled mode adds soft radiative stopping and samples hard photon events that debit electron energy. Its partition, event record, straggling treatment, and scoring boundary are documented in [Hard BremsLib photon events](hard-bremsstrahlung-events.md). The default continuum remains an uncoupled track-length estimator.

## Per-segment yield

For the EEDL and Bethe--Heitler backends emission is taken **isotropic**. For a segment of length {math}`L` traversed in an element of number density {math}`n_Z`, the contribution to the observed spectrum is

```{math}
\frac{d^2N}{dE\,d\Omega}=\frac{1}{4\pi}\,n_Z\,L\,\frac{d\sigma}{dk}\,T_{\rm abs},

```

with {math}`T_{\rm abs}` the Beer--Lambert escape transmission from the segment midpoint along the observation direction. Contributions are summed over segments and electrons and divided by the electron count, so the returned quantity is **photons per eV per steradian per incident electron** — the same units as the line spectrum.

Isotropy approximates a scattering-randomized electron population in the intended weakly relativistic regime; it does not resolve directional emission from an individual electron. The BremsLib backend does: each segment's {math}`\frac{1}{4\pi}\frac{d\sigma}{dk}` is replaced by {math}`\frac{d^2\sigma}{dk\,d\Omega}(T,k,\theta)` with {math}`\cos\theta=\hat v\cdot\hat n`, the straight-flight electron direction against the observation direction. The small coherent fraction of the continuum — which is what forms the CBS lines — is **not** subtracted here, so adding the line and continuum spectra slightly double counts that fraction.

Compound targets sum each element's cross section at its own number density. The analytic backend carries explicit {math}`Z^2` scaling; EEDL supplies an evaluated cross section for each element. Self-absorption uses the summed attenuation of the whole composition. In a layered stack each layer's segments radiate with that layer's composition and every photon is attenuated across the whole stack.

## Runtime implementation

The EEDL tables are prepared once per element and requested photon-energy grid. Every native incident-energy panel is interpolated onto that grid once, while the incident-panel bracket, interpolation fraction, total cross section, and exact cutoff-normalization factor are stored as one-dimensional segment arrays. Those staged values are reused for every reduction chunk; the tabulated physics is not reparsed or recopied for each chunk.

On CUDA with float32 spectra, one fused kernel interpolates the two adjacent panels, applies the {math}`k\le T` support, selects Bethe--Heitler only for uncovered segments, applies the existing absorption model, and reduces directly into the output bin. This avoids a dense segment-by-energy cross-section scratch array. Other backends and float64 use the same equations through the portable chunked implementation. Its memory admission uses the larger EEDL working-set estimate, and delayed CUDA/ROCm allocation errors are recognized at the point where they surface so the runner can retry the bremsstrahlung phase with a smaller chunk. Non-memory runtime errors are not retried.

## Escape and geometry

The escape path reuses the line kernel's geometry: a flat slab uses the z-only distance to the exit face, a finite rectangular footprint takes the nearest of the prism's six faces along the fixed observation direction, a layered absorber sums {math}`\mu_i\,\Delta z_i` across the stack, and a blazed groove replaces the distance with the exact periodic working-facet path. Grooved escape is single-slab only and requires the exact working-facet normal; other combinations raise rather than silently using flat attenuation. Details and limits are in [Photon escape and in-medium dispersion](photon-escape-and-dispersion.md).

Transport supplies material segments only, so vacuum legs advance the clock but do not radiate.

## Energy range and cutoffs

- `k <= 0` and `k > T` are hard-zero bins for EEDL. The optional analytic backend also sets the endpoint {math}`k=T` to zero.
- The EEDL tables cover their declared incident and secondary-energy ranges. Requests outside the incident-energy range warn and use Bethe--Heitler for only the affected segments.
- The normalized EEDL spectrum extends to 0.1 eV in the packaged tape. A simulation grid with a higher lower bound intentionally records only the corresponding partial total cross section.
- Background calculations use a low transport cutoff (typically 1 keV): electrons below the line-radiation cutoff still radiate into the soft X-ray window.
- One transport can be shared by radiation populations with different cutoffs, so each kernel re-applies the stopping rule to a population-specific energy floor. Rows starting below the floor are dropped; a terminal flight that crosses it is shortened — start time and energy unchanged, length and midpoint truncated — with the truncation distance solving {math}`E_{\rm end}=E_{\rm cut}` under the midpoint rule, so the shortened flight's state is reconstructed rather than lost. A higher-cutoff consumer therefore cannot recover radiation from the lower-cutoff tail.
- Attenuation is available only within the optical tables' supported ranges. Where {math}`\mu` is unavailable, the background kernel substitutes zero attenuation. This is a numerical fallback, not evidence that the material is transparent. Production grids apply the material-dependent floor described in [energy-grid semantics](energy-grid-semantics.md); explicit grids still need their data support checked.

## Path-integral order

Each row contributes one evaluation of the integrand rather than a quadrature along the flight. Under `energy_model="midpoint"` that evaluation uses the representative energy `E_repr_keV`, making it a midpoint rule (second order in the flight length); frozen rows keep the left-endpoint one (first order). Refinement controls the residual path-quadrature error. `brem_endpoint_quadrature_error` measures the residual difference, opt-in and read-only.

The continuum needs no coherent flight grouping. Its contribution is linear in segment length when energy and attenuation are held fixed. Subdivision of an evolving, absorbing track changes the quadrature and is not exactly invariant.

## External backgrounds

`load_external_brem` interpolates a two-column external background (energy [eV], intensity) onto the spectral grid, for comparison against or subtraction from an independently simulated continuum — e.g. a NIST DTSA-II simulation. The file is treated as an **as-detected** spectrum: window efficiency and detector resolution are not re-applied, and energies outside the file's range interpolate to zero. The intensity must already be in the detected units of the plot it joins.

## Limits and assumptions

- EEDL and Bethe--Heitler emission is isotropic, appropriate only to the intended weakly relativistic regime; no backend carries polarization;
- the BremsLib backend takes the emission angle from each segment's straight-flight direction and one fixed observation direction (a point detector); the DDCS is azimuthally symmetric (unpolarized beam, unoriented target). Interpolation error is under about 1.5 % in the DDCS and 0.4 % in the SDCS at the library's own spacing, estimated from drop-one-node tests over 20 keV--3 MeV for C, Si and W. Against the vendor's `Interpolate_DCS` output for Au it is at most 0.8 % on the {math}`T_1` grid, and a 0.4 % median (5.4 % at the last point before the tip) between {math}`T_1` nodes on the 0-degree forward peak;
- the BremsLib portable path costs about four times the EEDL path on CPU; a fused CUDA reducer matches the portable CUDA path on the hardware-gated synthetic-table comparison (including an out-of-range fallback segment);
- EEDL is elemental atomic data; molecular bonding, density-dependent emission effects, and interactions below the configured transport cutoff are outside this model;
- the optional/fallback Bethe--Heitler form is unscreened and approximate, especially for high {math}`Z`; its Elwert factor is not a substitute for an evaluated relativistic cross section;
- segment contributions and incident electrons are summed incoherently, with no phase and therefore no coherent-emission counterpart;
- the coherent (CBS) fraction of the continuum is not subtracted;
- self-absorption uses the same layered escape model as line radiation, with no feedback on emission; see [Multilayer materials](../materials/multilayer-materials.md).

## Validation

Ledger rows: `bremslib-angular-model` for the default direction-resolved backend, `bremslib-radiative-partition` and `bremslib-radiative-event-spectrum` for the opt-in coupled mode, `brem-spectrum` for EEDL parsing, interpolation, normalization, fallback selection, and per-segment assembly, `external-brem-subtraction` for the weighted scale-only sideband fit that consumes a loaded external background, `substep-radiation-invariance` for the representative-energy evaluation, `radiation-error-estimators` for the opt-in quadrature estimator, and `finite-transverse-crystal` / `blazed-groove-geometry` for the escape variants. The full derivation, dimensional analysis, limits, and numeric comparison are in [Bremsstrahlung spectrum validation](../../validation/radiation-physics/brem-spectrum.md), with the literature comparison in [external bremsstrahlung comparison](../../validation/literature-ref/external-bremsstrahlung-comparison.md). Implementation owner: `pyrite.montecarlo.spectrum.brem.mc_brem_spectrum`.
