# Positioned-filter attenuation verification packet

**Validation ID:** `positioned-filter-attenuation`

**Status:** fresh-context independent re-derivation completed 2026-08-14;
human ledger transition pending.

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

## Independent re-derivation (2026-08-14)

The verifier first recorded the cited Bouguer--Beer/Henke source, the three
ledgered signatures, their units and assumptions, and the limiting cases. The
following derivation was completed before any implementation body or this
author-prepared packet was read.

Let the source be the origin and write the finite source--pixel-centre ray as

\[
\boldsymbol{x}_p(q)=q\,\hat{\boldsymbol{u}}_p,
\qquad 0\leq q\leq d_p,
\]

where \(d_p\) is the source--pixel distance in millimetres. For plate \(j\),
let \(\boldsymbol{c}_j\) be its centre, let
\(\{\boldsymbol{e}_{jk}\}_{k=1}^3\) be its orthonormal local axes, and let
\(h_{jk}\) be its three half-extents. Along local axis \(k\), points inside
the box satisfy

\[
-h_{jk}\leq a_{jk}+q b_{pjk}\leq h_{jk},\qquad
a_{jk}=-\boldsymbol{c}_j\mathbin{\cdot}\boldsymbol{e}_{jk},\quad
b_{pjk}=\hat{\boldsymbol{u}}_p\mathbin{\cdot}\boldsymbol{e}_{jk}.
\]

For \(b_{pjk}\ne0\), sorting
\((-h_{jk}-a_{jk})/b_{pjk}\) and
\((h_{jk}-a_{jk})/b_{pjk}\) gives that slab's entry and exit distances. For
\(b_{pjk}=0\), the interval is all distances when \(|a_{jk}|\leq h_{jk}\),
and empty otherwise. Intersecting all three slab intervals with \([0,d_p]\)
gives \([q_{\rm in},q_{\rm out}]\), hence

\[
\ell_{pj}=\max(0,q_{\rm out}-q_{\rm in}).
\]

This derivation includes finite-segment clipping, misses, side escape, and
parallel rays. A centred normal ray gives \(\ell=t\); an infinite lateral
plate at incidence angle \(\theta\) from its normal gives
\(\ell=t/|\cos\theta|\).

For the \(\exp(-i\omega t)\) phasor convention, write the passive Henke index
as \(\tilde n=1-\delta+i\beta\). Intensity propagation then gives

\[
I(\ell)=I(0)e^{-2k\beta\ell},\qquad
\beta(E)=\frac{r_e\lambda^2}{2\pi}\sum_i n_i f_{2,i}(E).
\]

With \(k=2\pi/\lambda\), the independently derived linear attenuation is

\[
\mu(E)=2r_e\lambda\sum_i n_i f_{2,i}(E)
      =\sum_i L_{\mathrm{abs},i}^{-1}(E).
\]

Thus each independent passive plate obeys
\(dI/d\ell=-\mu I\), serial factors multiply, and

\[
T_p(E)=\exp\!\left[-\sum_j\mu_j(E)\ell_{pj}\right],\qquad
F_p(E)=I_{q(p)}(E)\,\Delta\Omega_p\,T_p(E).
\]

The units are \(r_e,\lambda\,[\mathrm{\AA}]\),
\(n_i\,[\mathrm{\AA}^{-3}]\), and therefore
\(\mu\,[\mathrm{\AA}^{-1}]\). Multiplication by
\(10^7\ \mathrm{\AA/mm}\) produces \(\mathrm{mm}^{-1}\), so every
\(\mu_j\ell_{pj}\) is dimensionless.

### Independent comparison

- `ray_box_path_lengths` uses the equivalent distance parameter \(q\), starts
  with the finite interval \([0,d_p]\), intersects the same three sorted slab
  intervals, and returns \(\max(0,q_{\rm out}-q_{\rm in})\). No factor of
  \(d_p\), cosine, or two is missing.
- `linear_attenuation_inv_mm` sums
  \(L_{\mathrm{abs},i}^{-1}\) and multiplies by \(10^7\), exactly matching the
  derived angstrom-to-millimetre conversion.
- `primary_transmission` contracts the filter index as
  \(\sum_j\ell_{pj}\mu_j(E)\) and applies one negative exponential. The
  materialized pixel result multiplies the intrinsic tile spectrum, pixel
  solid angle, and this transmission once each.
- A source-level numeric calculation used SciPy physical constants and
  `xraydb.f2_chantler` directly, without either attenuation implementation
  helper. At 8 and 12 keV it gave respectively
  \([14.55358057,4.31004892]\) mm\(^{-1}\) for Si at
  \(n=0.04994\ \mathrm{\AA}^{-3}\), versus
  \([14.55358041,4.31004887]\) mm\(^{-1}\) from the public helper. For an
  Al--O medium with \(n_{\rm Al}=0.02345\) and
  \(n_{\rm O}=0.03517\ \mathrm{\AA}^{-3}\), the source calculation gave
  \([6.06953199,1.77018238]\) mm\(^{-1}\), versus
  \([6.06953192,1.77018236]\) mm\(^{-1}\). The maximum relative difference
  was \(1.08\times10^{-8}\), consistent with the physical-constant precision
  used by the two paths.
- The focused material, transmission, and finite-box geometry tests all pass
  (36 tests). They cover the zero-filter and uncovered-ray identities, normal
  and oblique incidence, grazing side escape, finite-segment misses, compound
  additivity, and plate-order invariance.

Cheap filters therefore pass: \(\mu\ell\) is dimensionless; passive
coefficients give \(0<T\leq1\); no filters, \(\mu\to0\), or \(\ell=0\) gives
\(T=1\); positive \(\mu\) with \(\ell\to\infty\) gives \(T\to0\); and the
sum is invariant under a joint plate permutation. The implementation matches
the independent derivation with no factor, sign, exponent, unit, or convention
divergence.

Traceability finding: `ray_box_path_lengths` is named in the ledgered claim but
its docstring lacks `Validation: positioned-filter-attenuation`. This is not a
physics-expression discrepancy, but the marker should be added by the owning
implementation context before human sign-off.

**Verifier verdict:** `rederived`. Suggested human-applied ledger change:
`unverified` to `rederived`, record the completed fresh-context/source-level
comparison, and retain the missing-marker finding until repaired. This is not
human `signed-off` status.

## Independent verification contract

A fresh-context verifier must derive the equations and units without using the
new public attenuation helper as its oracle, compare at least one element and
one compound `MediumSpec` against an external or source-level calculation,
check normal, grazing/parallel, miss, zero-filter, and plate-order limits, and
audit the source-to-code sign and unit conversion. The verifier records any
discrepancy here and updates the ledger status. Human sign-off remains separate.
