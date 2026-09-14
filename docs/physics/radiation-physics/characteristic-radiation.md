# Characteristic radiation

PyRITE models electron-impact characteristic X rays as direct atomic
vacancies followed by radiative relaxation. Vacancy-production cross sections
come from EEDL; line energies, fluorescence yields, branching intensities, and
natural level widths come from xraydb. The result is an incoherent,
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
from that vacancy. A material segment of length {math}`L_j` therefore produces
the expected integrated line yield

```{math}
Y_{jai\ell}
=\frac{n_a L_j\,\sigma_{ai}(T_j)\,\omega_{ai}I_{ai\ell}}
       {4\pi N_e}
  \exp[-\tau_j(E_{ai\ell})].
```

Here {math}`n_a` is the elemental number density, {math}`N_e` is the number of
incident electrons, and {math}`\tau_j` is the Beer--Lambert optical depth from
the segment midpoint to the surface along the observation direction. The
{math}`1/(4\pi)` factor is the isotropic-emission approximation.

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
unchanged. The
spectral density contributed to bin {math}`b` is {math}`Y_{jai\ell}q_{\ell
b}/\Delta E_b`, in photons {math}`\mathrm{eV}^{-1}\,\mathrm{sr}^{-1}` per
incident electron.
`pyrite.montecarlo.spectrum.characteristic.characteristic_line_window_mass`
reports {math}`P_{\ell,W}` and its complement {math}`1-P_{\ell,W}` per line so
window truncation is an explicit, queryable quantity rather than a number
folded silently into the returned density. `mc_characteristic_spectrum` also
warns when a line's centre lies inside the requested grid but the grid still
captures less than
`CHARACTERISTIC_SEVERE_TRUNCATION_FRACTION` (50%) of its mass -- the window
edge, not the line's off-grid centre, is then the reason for the missing mass.
A line whose centre lies entirely outside the grid does not warn: its small
in-window tail is the intended off-grid-line behaviour above, not a
misconfigured window.

### Bin-edge convention

{math}`E_b^-,E_b^+` come from
`_energy_bin_edges_and_widths`, reusing the interior-midpoint,
reflected-half-width convention shared with
`pyrite._grid_semantics.node_bin_edges_and_widths`: interior edges sit at
{math}`\tfrac12(E_i+E_{i+1})`, and each outer edge mirrors the adjacent spacing
outward. Photon energy is a one-sided physical coordinate, so the low edge is
additionally clamped at 0 eV instead of the raw mirror reflection, which can
go negative whenever the grid's first spacing exceeds its first node (for
example a log-floored grid whose first two nodes sit close together after a
wide gap to a lower floor). No coordinate is inserted for that floor -- the
edge array still holds exactly `grid.size + 1` entries -- only the value of
the existing first edge changes. The high edge is never clamped; photon energy
has no equivalent upper physical bound here.

Measured transition-metal emission features can require several Lorentzians to
describe unresolved satellites and asymmetric structure
{cite:p}`holzer1997`. PyRITE deliberately uses one natural-width Lorentzian per
xraydb transition because the available atomic tables do not provide a
portable multi-component fit for every element and line. Detector response is
still applied downstream and will dominate whenever its resolution is broader
than the natural width. Satellite structure, chemical shifts, and
multiple-vacancy broadening are therefore outside this model.

## Relaxation and transport scope

The relaxation model treats independent atoms and isolated, directly created
vacancies. It does not synthesize Auger-fed daughter vacancies,
Coster--Kronig redistribution, multiple-vacancy shifts, or Auger-electron
transport. An EEDL shell with nonzero fluorescence yield but no xraydb line
list emits zero and raises a warning rather than being silently approximated.

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

Atomic relaxation is incoherent. The computed component is added once to both
the incoherent and optional coherent PXR/CBS totals, while remaining available
as `spec_characteristic`. Checkpoints store it in `characteristic.h5`, separate
from the line-only `line.h5` and continuum `brem.h5` datasets. The analysis app
shows characteristic radiation by default and provides a **show characteristic
radiation** checkbox that can subtract it from both displayed totals without
changing stored results. A missing characteristic file is valid and behaves as
a line-only legacy or intentionally excluded dataset.

## Limits and validation

- Zero density, path length, cross section, fluorescence yield, or branching
  intensity gives zero characteristic yield.
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

This window and cutoff convention is encoded in the `lorentzian-v4`
characteristic-model marker used by dataset identities and case-content keys.
Older `lorentzian-v2` checkpoints used conditional window renormalization;
`lorentzian-v3` could emit an unphysical negative low bin edge for a grid
whose first spacing exceeds its first node. Both are deliberately
cache-incompatible with `v4`.

Implementation owner:
`pyrite.montecarlo.spectrum.characteristic.mc_characteristic_spectrum`.
The source-to-code mapping, dimensional analysis, assumptions, and regression
anchors are recorded in [Characteristic-radiation
validation](../../validation/radiation-physics/characteristic-radiation.md)
under ledger row `characteristic-radiation`.
