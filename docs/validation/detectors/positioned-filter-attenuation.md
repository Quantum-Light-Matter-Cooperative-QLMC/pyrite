# Positioned-filter attenuation verification packet

**Validation ID:** `positioned-filter-attenuation`

**Status:** author-prepared derivation and implementation checks; fresh-context
independent verification pending.

**Code:**
`src/pyrite/materials/attenuation.py::linear_attenuation_inv_mm`,
`src/pyrite/instrument/geometry.py::ray_box_path_lengths`, and
`src/pyrite/instrument/attenuation.py::primary_transmission`.

## Claim and source

For a point source, let the unit ray to detector pixel centre (p) be
(hat{u}_p). Its exact intersection length with finite filter box (j) is
(ell_{pj}) in millimetres. The primary-photon transmission and pixel flux
are

\[
T_p(E) = \exp\!\left[-\sum_j \mu_j(E)\ell_{pj}\right],
\qquad
F_p(E) = I_{q(p)}(E)\,\Delta\Omega_p\,T_p(E).
\]

Here (I_{q(p)}) is the intrinsic tile spectrum in photons per electron per
electronvolt per steradian and (Delta\Omega_p) is the pixel solid angle.
The exponential is the Bouguer--Beer law. Elemental attenuation coefficients
come from the Henke/Chantler imaginary scattering factors through the already
ledgered `absorption-length` implementation. The primary tabulation source is
Henke et al., *Atomic Data and Nuclear Data Tables* **54**, 181--342 (1993),
DOI [10.1006/adnd.1993.1013](https://doi.org/10.1006/adnd.1993.1013).

## Implementation derivation

For composition number densities (n_i), independent elemental absorption
rates add:

\[
\mu_j(E) = \sum_i \frac{1}{L_{\mathrm{abs},i}(E)}.
\]

The existing absorption-length routine returns (L_{\mathrm{abs}}) in
angstroms, hence the public helper multiplies the summed inverse length by
(10^7\ \mathrm{\AA/mm}) to return (mathrm{mm}^{-1}).

In plate-local coordinates, each ray is clipped against all three box slabs.
The largest entry parameter and smallest exit parameter delimit the in-box
interval; clipping that interval to the finite source--pixel segment gives
(ell_{pj}). This same interval handles normal incidence, oblique thickness,
side escape, and misses without a separate projected-mask approximation.

Because exponentials multiply, serial passive plates give

\[
\prod_j e^{-\mu_j\ell_{pj}}
= e^{-\sum_j\mu_j\ell_{pj}},
\]

so plate order cannot change transmission. Declared order remains part of
provenance.

## Units, assumptions, and limits

- (mu_j) has units (mathrm{mm}^{-1}), (ell_{pj}) has units mm, and
  the exponential argument is dimensionless.
- Zero filters, zero path length, or an uncovered pixel gives (T_p(E)=1)
  exactly. For a normal ray through a full plate, (ell=t). For finite
  positive (muell), (0<T\leq1).
- A homogeneous passive filter and independent primary-photon attenuation are
  assumed. The model excludes scattering, fluorescence, diffraction,
  secondary production, surface reflection, and detector charge transport.
- Pixels are sampled by centre rays from a point source. The model does not
  integrate the finite pixel sensitive area or an extended emission volume.

## Implementation-side anchors

The focused tests pin direct composition summation against elemental
absorption lengths, the angstrom-to-millimetre conversion, exact zero-filter
identity, a closed-form normal-incidence case, compound additivity, and plate
order invariance. Geometry tests separately pin misses, partial coverage,
rotation, side escape, movement, and finite source--detector clipping.

These checks are implementation evidence, not independent validation.

## Independent verification contract

A fresh-context verifier must derive the equations and units without using the
new public attenuation helper as its oracle, compare at least one element and
one compound `MediumSpec` against an external or source-level calculation,
check normal, grazing/parallel, miss, zero-filter, and plate-order limits, and
audit the source-to-code sign and unit conversion. The verifier records any
discrepancy here and updates the ledger status. Human sign-off remains separate.
