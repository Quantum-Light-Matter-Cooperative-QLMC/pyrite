# Bremsstrahlung

PyRITE models the smooth incoherent background emitted along transported
electron segments with a Born Bethe--Heitler cross-section and Elwert Coulomb
correction. It shares the transport segments, the escape geometry, and the
per-electron normalization with the [line
kernel](coherent-radiation.md), and is evaluated on the same spectral grid, so
line and continuum can be added directly.

## Cross section

For element atomic number {math}`Z`, electron kinetic energy {math}`T`, and photon
energy {math}`k`, the implemented energy-differential form is

```{math}
\frac{d\sigma}{dk} = \frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2} \ln \! \left( \frac{p_i+p_f}{p_i-p_f} \right)f_E,
\qquad 0<k<T,

```

with relativistic momenta used as a weakly relativistic extension and

```{math}
f_E = \frac{\beta_i}{\beta_f} \frac{1-e^{-2\pi\alpha Z/\beta_i}} {1-e^{-2\pi\alpha Z/\beta_f}}

```

Momenta are carried in units of {math}`m_ec`, built from the exact relativistic
relation {math}`p=\sqrt{T(T+2m_ec^2)}/m_ec` and {math}`\beta=p/(1+T/m_ec^2)`, with
 {math}`T_f=T_i-k`. The Born form itself is the nonrelativistic dipole result (cf. Koch
& Motz, *Rev. Mod. Phys.* **31**, 920 (1959)); using relativistic momenta inside
it is a weakly relativistic extension, not a relativistic derivation. The
combination is adequate for {math}`Z\lesssim30` and {math}`T\lesssim100` keV; Seltzer--Berger
tables would be the accuracy upgrade.

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

- `k <= 0` and `k >= T` are hard-zero bins.
- The infrared spectrum rises approximately as {math}`\ln(4T/k)/k` and therefore
  depends on the configured low-energy bound; the integrated background is not a
  cutoff-independent number.
- Born alone vanishes at the tip, while Born times Elwert approaches a finite
  value immediately below the hard cutoff.
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
- the unscreened Born form is approximate, especially for high {math}`Z` or outside
  the tens-of-keV regime — atomic screening is absent, so the soft end is
  overestimated where screening matters;
- the Elwert factor is a low-energy Coulomb correction and is not a substitute
  for a relativistic cross section;
- segment contributions and incident electrons are summed incoherently, with no
  phase and therefore no coherent-emission counterpart;
- the coherent (CBS) fraction of the continuum is not subtracted;
- self-absorption uses the same layered escape model as line radiation, with no
  feedback on emission; see [Multilayer
  materials](../materials/multilayer-materials.md).

## Validation

Ledger rows: `brem-spectrum` for the cross section and per-segment assembly,
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
