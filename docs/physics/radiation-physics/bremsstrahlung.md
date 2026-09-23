# Bremsstrahlung

PyRITE models the smooth incoherent background emitted along transported electron segments with the Livermore Evaluated Electron Data Library (EEDL) {cite:p}`perkins1991eedl`. The default cross section is the evaluated total in ENDF-6 MF=23/MT=527 multiplied by the normalized photon-energy probability density in MF=26/MT=527. It shares the transport segments, the escape geometry, and the per-electron normalization with the [line kernel](coherent-radiation.md), but can use a separate photon-energy grid. Combining line and continuum requires compatible energy coordinates and density conventions; see [spectral observables](spectral-observables.md).

## What MF=23/MT=527 and MF=26/MT=527 mean

ENDF-6 organizes evaluated data by **file number** (`MF`, the kind of data) and **reaction number** (`MT`, the physical interaction). The pair therefore names a section of the EEDL evaluation; it is not a fitted parameter:

- `MF=23` contains integrated photo-atomic and electro-atomic cross sections. Its `MT=527` section is the total electro-atomic bremsstrahlung cross section {math}`\sigma(T)` as a function of incident-electron energy, tabulated in barns.
- `MF=26` contains secondary-particle energy/angle distributions. Its matching `MT=527` section describes the products of the same bremsstrahlung interaction. PyRITE uses the first subsection, whose `ZAP=0`, `LAW=1`, `LANG=1`, `NA=0` records give an isotropic tabulated photon-energy probability density. The second subsection uses `LAW=8` to report the outgoing electron's average energy loss and is not needed by the photon spectrum estimator.

The two sections are used together because they supply complementary pieces: MF=23 fixes the total probability of a bremsstrahlung event, while MF=26 fixes the conditional photon-energy shape. Their product is the required differential cross section. The section definitions follow ENDF-6 {cite:p}`trkov2018endf6`.

## EEDL cross section and spectrum

For element {math}`Z`, incident electron kinetic energy {math}`T`, and photon energy {math}`k`, the default energy-differential cross section is

```{math}
\frac{d\sigma_Z}{dk}(T,k) = \sigma_Z^{23,527}(T)\,P_Z^{26,527}(k\mid T),
\qquad \int_0^T P_Z(k\mid T)\,dk=1.

```

`endf-parserpy` reads both sections from the checksum-pinned packaged `EEDL.endf`. PyRITE accepts the tape only when the total and photon tables are finite, ordered, non-negative, and declare the supported ENDF lin-lin laws. The MF=23 values are converted from barns to cm². Each native MF=26 photon density is normalized after parsing to remove only source-record rounding error.

The tape declares lin-lin interpolation for the total cross section, the secondary photon-energy axis (`LEP=2`), and the incident-energy panels (`INT=2`). PyRITE follows those laws: it interpolates adjacent photon spectra at fixed absolute photon energy. That interpolation can retain a small tail above an intermediate incident energy because adjacent panels have different upper endpoints, so the implementation imposes {math}`0<k\le T` and analytically renormalizes the surviving piecewise-linear density. Consequently, integrating the differential cross section over the full physical photon range recovers the interpolated MF=23 total.

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

## BremsLib direction-resolved backend

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

- A released element whose table has not been fetched is an error that names the fetch command.
- An element with no table, or a segment energy outside the table's range, warns and uses isotropic EEDL for only that element or those segments.

The run path does not select this backend yet. `"eedl"` stays the packaged default until #86 establishes the BremsLib accuracy claim. The model marker `BREMSSTRAHLUNG_BREMSLIB_MODEL` and `table_identity(tables)` supply the run identity for when it is selected.

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

Ledger rows: `brem-spectrum` for EEDL parsing, interpolation, normalization, fallback selection, and per-segment assembly, `bremslib-angular-model` for the BremsLib direction-resolved backend, `external-brem-subtraction` for the weighted scale-only sideband fit that consumes a loaded external background, `substep-radiation-invariance` for the representative-energy evaluation, `radiation-error-estimators` for the opt-in quadrature estimator, and `finite-transverse-crystal` / `blazed-groove-geometry` for the escape variants. The full derivation, dimensional analysis, limits, and numeric comparison are in [Bremsstrahlung spectrum validation](../../validation/radiation-physics/brem-spectrum.md), with the literature comparison in [external bremsstrahlung comparison](../../validation/literature-ref/external-bremsstrahlung-comparison.md). Implementation owner: `pyrite.montecarlo.spectrum.brem.mc_brem_spectrum`.
