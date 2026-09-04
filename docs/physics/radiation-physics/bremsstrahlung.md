# Bremsstrahlung

PyRITE models the smooth incoherent background emitted along transported
electron segments with the Livermore Evaluated Electron Data Library (EEDL)
{cite:p}`perkins1991eedl`. The default cross section is the evaluated total in
ENDF-6 MF=23/MT=527 multiplied by the normalized photon-energy probability
density in MF=26/MT=527. It shares the transport segments, the escape geometry,
and the per-electron normalization with the [line
kernel](coherent-radiation.md), and is evaluated on the same spectral grid, so
line and continuum can be added directly.

## EEDL cross section and spectrum

For element {math}`Z`, incident electron kinetic energy {math}`T`, and photon
energy {math}`k`, the default energy-differential cross section is

```{math}
\frac{d\sigma_Z}{dk}(T,k) = \sigma_Z^{23,527}(T)\,P_Z^{26,527}(k\mid T),
\qquad \int_0^T P_Z(k\mid T)\,dk=1.

```

`endf-parserpy` reads both sections from the checksum-pinned packaged
`EEDL.endf`. PyRITE accepts the tape only when the total and photon tables are
finite, ordered, non-negative, and declare the supported ENDF lin-lin laws. The
MF=23 values are converted from barns to cm². Each native MF=26 photon density
is normalized after parsing to remove only source-record rounding error.

The tape declares lin-lin interpolation for the total cross section, the
secondary photon-energy axis (`LEP=2`), and the incident-energy panels
(`INT=2`). PyRITE follows those laws: it interpolates adjacent photon spectra at
fixed absolute photon energy. That interpolation can retain a small tail above
an intermediate incident energy because adjacent panels have different upper
endpoints, so the implementation imposes {math}`0<k\le T` and analytically
renormalizes the surviving piecewise-linear density. Consequently, integrating
the differential cross section over the full physical photon range recovers the
interpolated MF=23 total.

The ENDF-6 electro-atomic format describes the bremsstrahlung photon subsection
as an isotropic, angle-independent tabulated spectrum (`LAW=1`, `LANG=1`,
`NA=0`) {cite:p}`trkov2018endf6`. This agrees with and therefore retains
PyRITE's existing angular model.

## Optional Bethe--Heitler backend

The former analytic model remains available with
`cross_section_model="bethe-heitler"` on `mc_brem_spectrum`. It is the
nonrelativistic, unscreened Born Bethe--Heitler form with the
{cite:t}`elwert1939` correction, using relativistic momenta as a
weakly-relativistic extension {cite:p}`kochmotz1959`:

```{math}
\frac{d\sigma}{dk}=\frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2}\ln\!\left(\frac{p_i+p_f}{p_i-p_f}\right)
\frac{\beta_i}{\beta_f}
\frac{1-e^{-2\pi\alpha Z/\beta_i}}{1-e^{-2\pi\alpha Z/\beta_f}}.

```

Momenta are carried in units of {math}`m_ec`, built from the exact relativistic
relation {math}`p=\sqrt{T(T+2m_ec^2)}/m_ec` and {math}`\beta=p/(1+T/m_ec^2)`, with
{math}`T_f=T_i-k`. This backend is retained for reproducibility and as the
fallback when EEDL lacks an element or incident-energy range. Every fallback
emits a `RuntimeWarning` that identifies the missing coverage. Malformed or
checksum-mismatched EEDL data are not treated as missing coverage and remain
hard errors.

## Per-segment yield

Emission is taken **isotropic**. For a segment of length {math}`L` traversed in an
element of number density {math}`n_Z`, the contribution to the observed spectrum is

```{math}
\frac{d^2N}{dE\,d\Omega}=\frac{1}{4\pi}\,n_Z\,L\,\frac{d\sigma}{dk}\,T_{\rm abs},

```

with {math}`T_{\rm abs}` the Beer--Lambert escape transmission from the segment
midpoint along the observation direction. Contributions are summed over segments
and electrons and divided by the electron count, so the returned quantity is
**photons per eV per steradian per incident electron** — the same units as the
line spectrum.

Isotropy is the standard assumption at weakly relativistic energies once
electron directions are scattering-randomized, and is the one adopted for the
comparison estimates this model was built against. The small coherent fraction
of the continuum — which is what forms the CBS lines — is **not** subtracted
here, so adding the line and continuum spectra slightly double counts that
fraction.

Compound targets add element contributions with their own {math}`Z^2` weighting at
their own number densities, while the self-absorption uses the summed
attenuation of the whole composition. In a layered stack each layer's segments
radiate with that layer's composition and every photon is attenuated across the
whole stack.

## Runtime implementation

The EEDL tables are prepared once per element and requested photon-energy grid.
Every native incident-energy panel is interpolated onto that grid once, while
the incident-panel bracket, interpolation fraction, total cross section, and
exact cutoff-normalization factor are stored as one-dimensional segment arrays.
Those staged values are reused for every reduction chunk; the tabulated physics
is not reparsed or recopied for each chunk.

On CUDA with float32 spectra, one fused kernel interpolates the two adjacent
panels, applies the {math}`k\le T` support, selects Bethe--Heitler only for
uncovered segments, applies the existing absorption model, and reduces directly
into the output bin. This avoids a dense segment-by-energy cross-section scratch
array. Other backends and float64 use the same equations through the portable
chunked implementation. Its memory admission uses the larger EEDL working-set
estimate, and delayed CUDA/ROCm allocation errors are recognized at the point
where they surface so the runner can retry the bremsstrahlung phase with a
smaller chunk. Non-memory runtime errors are not retried.

## Escape and geometry

The escape path reuses the line kernel's geometry: a flat slab uses the z-only
distance to the exit face, a finite rectangular footprint takes the nearest of
the prism's six faces along the fixed observation direction, a layered absorber
sums {math}`\mu_i\,\Delta z_i` across the stack, and a blazed groove replaces the
distance with the exact periodic working-facet path. Grooved escape is
single-slab only and requires the exact working-facet normal; other combinations
raise rather than silently using flat attenuation. Details and limits are in
[Photon escape and in-medium dispersion](photon-escape-and-dispersion.md).

Transport supplies material segments only, so vacuum legs advance the clock but
do not radiate.

## Energy range and cutoffs

- `k <= 0` and `k > T` are hard-zero bins for EEDL. The optional analytic
  backend also zeros its historical endpoint convention.
- The EEDL tables cover their declared incident and secondary-energy ranges.
  Requests outside the incident-energy range warn and use Bethe--Heitler for
  only the affected segments.
- The normalized EEDL spectrum extends to 0.1 eV in the packaged tape. A
  simulation grid with a higher lower bound intentionally records only the
  corresponding partial total cross section.
- Background runs want a **low transport cutoff** (~1 keV): electrons below the
  line-radiation cutoff still radiate into the soft X-ray window.
- One transport can be shared by radiation populations with different cutoffs, so
  each kernel re-applies the stopping rule to a population-specific energy floor.
  Rows starting below the floor are dropped; a terminal flight that crosses it is
  shortened — start time and energy unchanged, length and midpoint truncated —
  with the truncation distance solving {math}`E_{\rm end}=E_{\rm cut}` under the
  midpoint rule, so the shortened flight's state is reconstructed rather than
  lost. A higher-cutoff consumer therefore cannot recover radiation from the
  lower-cutoff tail.
- The Henke/Chantler attenuation tables span roughly 20 eV--30 keV. Outside that
  range {math}`\mu` is unavailable, and the wide background grid treats it as **zero**
  (fully transparent) rather than propagating NaN through the integrated count
  rate. Hard X-rays do escape essentially unattenuated, but the softest bins
  below the table floor are also silently unattenuated — read the extremes of a
  wide grid with that in mind.

## Path-integral order

Each row contributes one evaluation of the integrand rather than a quadrature
along the flight. Under `energy_model="midpoint"` that evaluation uses the
representative energy `E_repr_keV`, making it a midpoint rule (second order in
the flight length); frozen rows keep the left-endpoint one (first order). This is
what makes the yield insensitive to how many numerical substeps a flight was
integrated in. `brem_endpoint_quadrature_error` measures the residual difference,
opt-in and read-only.

Unlike the line kernel, the continuum needs no flight grouping: the contribution
is linear in segment length, so splitting a flight is exact by construction.

## External backgrounds

`load_external_brem` interpolates a two-column external background (energy [eV],
intensity) onto the spectral grid, for comparison against or subtraction from an
independently simulated continuum — e.g. a NIST DTSA-II simulation. The file is
treated as an **as-detected** spectrum: window efficiency and detector resolution
are not re-applied, and energies outside the file's range interpolate to zero.
The intensity must already be in the detected units of the plot it joins.

## Limits and assumptions

- emission is isotropic, appropriate only to the intended weakly relativistic
  regime, and carries no polarization;
- EEDL is elemental atomic data; molecular bonding, density-dependent emission
  effects, and interactions below the configured transport cutoff are outside
  this model;
- the optional/fallback Bethe--Heitler form is unscreened and approximate,
  especially for high {math}`Z`; its Elwert factor is not a substitute for an
  evaluated relativistic cross section;
- segment contributions and incident electrons are summed incoherently, with no
  phase and therefore no coherent-emission counterpart;
- the coherent (CBS) fraction of the continuum is not subtracted;
- self-absorption uses the same layered escape model as line radiation, with no
  feedback on emission; see [Multilayer
  materials](../materials/multilayer-materials.md).

## Validation

Ledger rows: `brem-spectrum` for EEDL parsing, interpolation, normalization,
fallback selection, and per-segment assembly,
`external-brem-subtraction` for the weighted scale-only sideband fit that
consumes a loaded external background, `substep-radiation-invariance` for the representative-energy evaluation,
`radiation-error-estimators` for the opt-in quadrature estimator, and
`finite-transverse-crystal` / `blazed-groove-geometry` for the escape variants.
The full derivation, dimensional analysis, limits, and numeric comparison are in
[Bremsstrahlung spectrum
validation](../../validation/radiation-physics/brem-spectrum.md), with the
literature comparison in [external bremsstrahlung
comparison](../../validation/literature-ref/external-bremsstrahlung-comparison.md).
Implementation owner: `pyrite.montecarlo.spectrum.brem.mc_brem_spectrum`.
