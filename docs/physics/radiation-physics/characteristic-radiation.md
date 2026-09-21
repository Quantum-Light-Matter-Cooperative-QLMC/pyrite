# Characteristic radiation

PyRITE models electron-impact characteristic X rays as direct atomic
vacancies, L-shell Coster--Kronig redistribution of those vacancies, and
radiative relaxation. Vacancy-production cross sections come from EEDL; line
energies, fluorescence yields, branching intensities, Coster--Kronig
probabilities, and natural level widths come from xraydb. The result is an incoherent,
self-absorbed track-length estimate evaluated on the fine line-energy grid.

## Atomic data and vacancy yield

For element {math}`a` and initially ionized subshell {math}`i`, PyRITE reads the
EEDL ENDF-6 File 23 sections `MT=534` through `MT=572`. Their TAB1 records give
the electron-impact ionization cross section {math}`\sigma_{ai}(T)` in barns as
a function of incident-electron energy. Declared lin-lin interpolation is used
inside each table's range and the cross section is zero outside it. The
packaged tape is checksum-pinned before it is parsed.

xraydb's Elam tables {cite:p}`elam2002` provide a fluorescence yield
{math}`\omega_{ai}` and
conditional intensity {math}`I_{ai\ell}` for each radiative line {math}`\ell`
from that vacancy.

A vacancy need not radiate from the subshell it was created in. PyRITE
redistributes L-shell vacancies by Coster--Kronig transfer before applying the
radiative yields, using the Elam/Krause probabilities
{math}`f_{ij}` that xraydb exposes:

```{math}
:label: eq-characteristic-l-shell-ck

n_{L_1} &= (1-f_{12}-f_{13})\,N_{L_1},\\
n_{L_2} &= (1-f_{23})\,N_{L_2} + f_{12}N_{L_1},\\
n_{L_3} &= N_{L_3} + f_{23}N_{L_2} + f_{13}N_{L_1},
```

where {math}`N_i` is the directly produced (primary) vacancy population and
{math}`n_i` the population that radiates. xraydb's {math}`f_{13}` is a *total*
probability and already contains the {math}`L_1\to L_2\to L_3` route, so it is
applied to the primary {math}`N_{L_1}` and {math}`f_{23}` only to the primary
{math}`N_{L_2}`; routing the transferred {math}`f_{12}N_{L_1}` through
{math}`f_{23}` as well would count that path twice. Written as a matrix,
{math}`n_{i'}=\sum_i T^a_{ii'}N_i` with {math}`T^a` the row-stochastic transfer
of `_l_shell_vacancy_transfer`: Coster--Kronig moves one L hole outward without
creating a second L hole, so every row of {math}`T^a` sums to one, and every
non-L row is the identity.

A material segment of length {math}`L_j` therefore produces the expected
integrated line yield

```{math}
Y_{jai\ell}
=\frac{n_a L_j\,\sigma_{ai}(T_j)
       \sum_{i'}T^a_{ii'}\,\omega_{ai'}I_{ai'\ell}}
       {4\pi N_e}
  \exp[-\tau_j(E_{ai\ell})].
```

Here {math}`n_a` is the elemental number density, {math}`N_e` is the number of
incident electrons, and {math}`\tau_j` is the Beer--Lambert optical depth from
the segment midpoint to the surface along the observation direction. The
{math}`1/(4\pi)` factor is the isotropic-emission approximation.

Index {math}`i` is now the subshell the *electron* ionized and {math}`i'` the
subshell the photon came from. With {math}`T^a=\mathbb{1}` — any element for
which xraydb tabulates no L Coster--Kronig, which is every {math}`Z\leq11` —
this reduces exactly to the earlier direct-vacancy product
{math}`\omega_{ai}I_{ai\ell}`.

## Natural Lorentzian line shape

Each xraydb transition is represented by a normalized Lorentzian,

```{math}
L_\ell(E)=\frac{1}{\pi}
\frac{\Gamma_\ell/2}
     {(E-E_\ell)^2+(\Gamma_\ell/2)^2},
\qquad
\int_{-\infty}^{\infty}L_\ell(E)\,dE=1.
```

The transition full width at half maximum is the sum of the initial- and
final-level natural widths,
{math}`\Gamma_\ell=\Gamma_{\rm initial}+\Gamma_{\rm final}`, as tabulated by
xraydb from atomic-level width compilations
{cite:p}`krauseoliver1979,keskirahkonenkrause1974`. For a combined final label
such as `M4,5`, PyRITE uses the mean of the available component widths. A
missing final-state width contributes zero; a missing initial width is an error
because it would leave the line's principal broadening undefined.

The code integrates the profile over each energy bin analytically rather than
sampling it at bin centres. For bin edges {math}`E_b^-` and {math}`E_b^+`,

```{math}
q_{\ell b}=\frac{1}{\pi}
\left[
\tan^{-1}\!\left(\frac{2(E_b^+-E_\ell)}{\Gamma_\ell}\right)
-\tan^{-1}\!\left(\frac{2(E_b^--E_\ell)}{\Gamma_\ell}\right)
\right].
```

The bin masses retain the normalization of the physical Lorentzian on the
whole energy axis. They are not renormalized over the requested window:

```{math}
0 < P_{\ell,W}=\sum_{b\in W}q_{\ell b}\leq 1.
```

Thus a finite window records the fraction {math}`P_{\ell,W}` of the integrated
line yield and does not redistribute omitted tails into retained bins. A line
centre outside the window still contributes its physical in-window tail.
Changing either window boundary leaves every bin with unchanged edges
unchanged. The spectral density contributed to bin {math}`b` is {math}`Y_{jai\ell}q_{\ell
b}/\Delta E_b`, in photons {math}`\mathrm{eV}^{-1}\,\mathrm{sr}^{-1}` per
incident electron.
`pyrite.montecarlo.spectrum.characteristic.characteristic_line_window_mass`
reports the captured and omitted probability per line. The spectrum function
warns when a line centre lies inside the requested grid but less than
`CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION` (50%) of its mass is captured.
Off-grid line centres contribute their tails without that warning.

### Bin-edge convention

{math}`E_b^-,E_b^+` come from
`_energy_bin_edges_and_widths`, reusing the interior-midpoint,
reflected-half-width convention shared with
`pyrite._grid_semantics.node_bin_edges_and_widths`: interior edges sit at
{math}`\tfrac12(E_i+E_{i+1})`, and each outer edge mirrors the adjacent spacing
outward. The low edge is clamped to 0 eV if the reflected edge would be
negative. This changes the first edge without adding a bin: there are still
`grid.size + 1` edges. The upper edge is not clamped.

Measured transition-metal emission features can require several Lorentzians to
describe unresolved satellites and asymmetric structure
{cite:p}`holzer1997`. PyRITE deliberately uses one natural-width Lorentzian per
xraydb transition because the available atomic tables do not provide a
portable multi-component fit for every element and line. Detector response is
still applied downstream and will dominate whenever its resolution is broader
than the natural width. Satellite structure, chemical shifts, and
multiple-vacancy broadening are therefore outside this model.

## Relaxation and transport scope

The relaxation model treats independent atoms. A primary L vacancy is
redistributed across the L subshells by {eq}`eq-characteristic-l-shell-ck`
before it radiates; every other vacancy radiates from the subshell it was
created in.

Still outside the model:

- **Auger-fed daughter vacancies.** A vacancy's nonradiative decay is counted
  only through its effect on the fluorescence yield, so the vacancies its Auger
  electrons leave behind never radiate. This includes the outer-shell spectator
  vacancy left by a Coster--Kronig electron. Whenever the K shell is open the
  K-fed L population is under 2% of the direct L population, because
  {math}`\sigma_L\gg\sigma_K`; the untracked M and N population fed by L Auger
  decay is larger and is not bounded here.
- **M-shell Coster--Kronig.** xraydb's M-shell values are not a probability
  distribution — the finals of Cr M1 sum to 3.82, and 134 (Z, initial) pairs
  exceed one — so they are excluded rather than renormalized. L-shell values sum
  to at most one for every {math}`3\leq Z\leq98`.
- **Radiative branching outside the Elam line list.** {math}`\omega_{ai}` is
  distributed over only the lines xraydb tabulates for that subshell, whose
  intensities sum to one by construction. That conserves the subshell's total
  radiative yield but over-assigns intensity to tabulated lines wherever the
  source table omits weak ones. An EEDL shell with nonzero fluorescence yield
  and no xraydb line list at all emits zero and raises a warning rather than
  being silently approximated — most notably the M and N shells of heavy
  elements, so PyRITE does not claim M-shell spectra.
- **Secondary fluorescence.** {math}`\exp(-\tau_j)` is a pure sink: a
  characteristic photon absorbed on its way out does not re-emit. The error is
  small for a line below its own element's absorption edge and is not bounded
  for alloys or multilayers.
- **Auger-electron transport**, multiple-vacancy shifts, and satellite
  structure.

Characteristic emission uses the bremsstrahlung electron population and its
default 1 keV transport cutoff. This retains more low-energy ionization path
than the default 5 keV PXR/CBS population. The present stopping and scattering
model is not validated below 1 keV, so characteristic scoring enforces
{math}`E_{\rm cut}\geq1\,\mathrm{keV}`: an omitted cutoff resolves to 1 keV and
an explicitly lower cutoff is rejected. Low-binding-energy vacancies that
could physically be produced below 1 keV are therefore omitted. PyRITE does
not claim precision characteristic yields for incident energies near this
floor. The 50 eV line-data cutoff is a photon-line data boundary and does not
lower the electron transport-validity floor.

Flat slabs, finite footprints, blazed grooves, and layered samples reuse the
same photon-escape geometry as the other radiation kernels. In a multilayer,
each layer emits from its own elements and the photon is attenuated by every
layer on its escape path; passive absorber layers do not become emitters for a
different layer.

## Composition, storage, and analysis

Atomic relaxation is incoherent, so one component adds to either the
incoherent or the optional coherent PXR/CBS spectrum. It is kept as its own
array, `spec_characteristic`; `spec` and `spec_coherent` exclude it, and
consumers add it once through `pyrite._spectral_components.line_spectrum`
(`Result.line_total()` for API results). Checkpoints store it in
`characteristic.h5`, separate from the line `line.h5` and continuum `brem.h5`
datasets. The analysis app shows characteristic radiation by default and
provides a **show characteristic radiation** checkbox that drops the component
from displayed records without changing stored results. A missing
characteristic file is valid and contributes no characteristic component.

## Limits and validation

- Zero density, path length, cross section, fluorescence yield, or branching
  intensity gives zero characteristic yield.
- An element with no tabulated L Coster--Kronig gives {math}`T^a=\mathbb{1}` and
  the pre-cascade direct-vacancy yield exactly; that covers every
  {math}`Z\leq11`, so the carbon anchors are unchanged by
  {eq}`eq-characteristic-l-shell-ck`.
- Each row of {math}`T^a` sums to one, so redistribution moves L emission
  between subshells without creating or destroying an L vacancy. Because
  {math}`\omega_{L_3}>\omega_{L_2}>\omega_{L_1}`, it always raises total L
  emission: the factor is 1.235 for Cu at 30 keV, 1.114 for Mo at 60 keV, 1.079
  for Au at 100 keV, and 1.033 for Ta at 30 keV. Per-line ratios move much
  further than the totals — Cu's L1-origin share of L photons falls from 0.031
  to 0.0005.
- With zero attenuation, a finite window integrates to
  {math}`P_{\ell,W}nL\sigma\omega I/(4\pi N_e)`; the infinite-window limit
  recovers the complete line yield.
- Narrowing a window only removes Lorentzian probability. It does not rescale
  bins retained by both windows; off-grid centres retain nonzero tails.
- Splitting a constant-energy segment preserves the total when its attenuation
  weight is also held fixed (in particular, with zero attenuation), because the
  estimator is then linear in path length. Subdividing a real absorbing track
  changes the segment-midpoint escape quadrature and therefore the total: for a
  uniform path of optical depth 2, one midpoint gives 0.3679, two equal
  subsegments give 0.4148, and exact path integration gives 0.4323.
- Increasing optical depth suppresses the line monotonically.
- Natural Lorentzian broadening is source physics; detector broadening remains
  a separate downstream operation.

This window and cutoff convention, and the L-shell Coster--Kronig
redistribution, are encoded in the `l-shell-ck-lorentzian-v5`
characteristic-model marker used by dataset identities and case-content keys.
Earlier model markers are cache-incompatible with `v5`: `v4` records hold
un-redistributed L line yields, not the same spectrum.

Implementation owner:
`pyrite.montecarlo.spectrum.characteristic.mc_characteristic_spectrum`.
The source-to-code mapping, dimensional analysis, assumptions, and regression
anchors are recorded in [Characteristic-radiation
validation](../../validation/radiation-physics/characteristic-radiation.md)
under ledger row `characteristic-radiation`.
