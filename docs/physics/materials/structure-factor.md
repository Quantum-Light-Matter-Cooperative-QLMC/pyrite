# Structure factor and Debye–Waller

The structure factor is the bridge between the crystal's atomic arrangement and
its X-ray couplings. It is the one place where the per-element
[atomic form factors](../atomic-physics/atomic-form-factors.md) and the
[unit-cell geometry](crystal-structure.md) are combined, and both PXR and CBS
couplings are built from its output.

## Definition

For a reflection $hkl$ with reciprocal vector $\mathbf g$ and photon energy $E$,

```{math}
:label: eq-structure-factor-definition

S(\mathbf g, E) = \sum_j F_j(\mathbf g, E)\,
e^{\,i 2\pi\,(hkl)\cdot\mathbf R_j}\,
e^{-W_j},
```

summed over the basis sites $j$ of the unit cell, where $\mathbf R_j$ is the
**fractional** position of site $j$, $F_j$ is that element's complex atomic form
factor, and $e^{-W_j}$ is its amplitude Debye–Waller factor. The phase is
written with fractional coordinates and $2\pi$ explicit, which is identical to
$e^{i\mathbf g\cdot\mathbf r_j}$ in Cartesian form and avoids a second place
where the $|\mathbf g| = 2\pi/d$ convention could be misapplied.

$S$ is complex and dimensionless (electron units). It is returned together with
$|\mathbf g|$, because every consumer needs both and recomputing $|\mathbf g|$ is
the kind of duplication that invites a convention mismatch.

## Debye–Waller factor

Thermal motion smears each atom over a distribution of positions, which reduces
the coherent amplitude at momentum transfer $\mathbf g$:

```{math}
:label: eq-structure-factor-debye-waller

e^{-W},
\qquad
W = B\left(\frac{\sin\theta}{\lambda}\right)^{2}
  = B\,s^2
  = \frac{B\,g^2}{16\pi^2},
```

with the tabulated isotropic $B$ factor in $\AA^2$, related to the mean-square
displacement by $B = 8\pi^2\langle u_x^2\rangle$.

Two normalization traps live in {eq}`eq-structure-factor-debye-waller` and are
worth naming, because both are single factors of 4 that would survive most
sanity checks:

* The exponent is $g^2/16\pi^2$, **not** $g^2/4\pi^2$. This follows directly from
  $s = g/(4\pi)$ under the $g = 2\pi/d$ convention.
* {eq}`eq-structure-factor-debye-waller` is the **amplitude** factor. Intensities
  carry its square, $e^{-2W}$. The implementation applies the amplitude form
  inside $S$, so any $|S|^2$ automatically carries $e^{-2W}$ exactly once.

One scalar $B$ is applied to **every** site in the cell. There is no per-site
$B$, no anisotropic displacement tensor, and no anharmonic correction.

## Per-site form factor policy

Which variant of $F_j$ enters {eq}`eq-structure-factor-definition` is a policy,
described in full in
[Atomic form factors](../atomic-physics/atomic-form-factors.md). In short: the
production line-spectrum path uses the full complex $f_0 + f' + i f''$; a caller
that explicitly asks for the non-resonant treatment still gets the complex factor
for the hard-coded edge-prone element set, so a crystal with an in-band
absorption edge cannot quietly be evaluated without its anomalous terms.
Form factors are computed once per **unique element**, not once per site, since
$F$ depends only on the element and $|\mathbf g|$.

## What the structure factor feeds

```{list-table} Downstream users of $S(\mathbf g, E)$.
:name: tbl-structure-factor-consumers
:header-rows: 1

* - Quantity
  - Relation
  - Documented in
* - PXR susceptibility $\chi_{\mathbf g}$
  - $-r_e\lambda^2 S/(\pi V_{\rm cell})$
  - [Coherent PXR and CBS radiation](../radiation-physics/coherent-radiation.md)
* - CBS potential $U_{\mathbf g}$
  - same sum with $Z_j - f_j$ and a $1/g^2$ denominator
  - [Coherent PXR and CBS radiation](../radiation-physics/coherent-radiation.md)
* - Forward susceptibility $\chi_0$
  - {eq}`eq-structure-factor-definition` at $\mathbf g = 0$
  - [Photon escape and in-medium dispersion](../radiation-physics/photon-escape-and-dispersion.md)
* - Reflection ranking
  - $|S|/g^2$ as a proxy for $|\chi_{\mathbf g}|$
  - [Reflection selection](reflection-selection.md)
```

The $\mathbf g \to 0$ case is instructive: every phase factor and every
Debye–Waller factor reduces to 1, so $S(0)$ collapses to the plain forward sum
over the cell. The implementation of $\chi_0$ takes that limit analytically
rather than calling {eq}`eq-structure-factor-definition` with $\mathbf g = 0$,
so that it can use $Z + f'$ and share one normalization exactly with the
absorption coefficient instead of inheriting the $f_0(0) \approx Z$ fit residual.

## Limiting cases

```{list-table} Limiting behavior of {eq}`eq-structure-factor-definition`.
:name: tbl-structure-factor-limits
:header-rows: 1

* - Limit
  - Result
* - $B \to 0$
  - $e^{-W}\to 1$; the sum becomes the purely static structure factor
* - $\mathbf g \to 0$
  - $S \to \sum_j F_j(0)$, the forward-scattering sum over the cell
* - destructive basis phases
  - $S \to 0$; the reflection is **extinct** and contributes nothing
* - $f', f'' \to 0$
  - $S$ becomes real up to the basis phases; no anomalous contribution
* - single-atom basis at the origin
  - $S = F(g)e^{-W}$
```

Extinction is a real, checkable prediction of the phase convention rather than a
formality: for diamond, {eq}`eq-structure-factor-definition` must give exactly
zero for $(200)$ and $(222)$ and nonzero for $(111)$, $(220)$, $(311)$, $(400)$.
Those are pinned by regression test, which is what makes the phase sign and the
fractional-coordinate convention falsifiable.

## Assumptions and limits

* **Kinematic** (single-scattering) theory. No dynamical diffraction, no
  extinction corrections, no multiple-beam effects. This is the appropriate
  regime for the thin films and mosaic crystals the project targets, but it is a
  real ceiling for a thick, high-quality perfect crystal at a strong reflection.
* Independent, spherical, neutral atoms — the form-factor approximation carried
  in from the atomic layer.
* A single isotropic scalar $B$ per crystal, applied to every site.
* No thermal diffuse scattering: the intensity removed from the Bragg peak by
  {eq}`eq-structure-factor-debye-waller` is not re-emitted anywhere in the model.
* No temperature knob. $B$ is a static catalog value; a sample at a different
  temperature from the source refinement is not represented.

## Validation

`structure-factor` (`anchored`) covers {eq}`eq-structure-factor-definition` and
{eq}`eq-structure-factor-debye-waller`: units, the forward, $g\to0$, and $B\to0$
limits, the phase sign, the $16\pi^2$ exponent normalization (the factor-4 trap
above is explicitly ruled out), and the pinned diamond extinct/allowed set. Three
concurring fresh-context derivations agree.

`debye-waller-catalog-provenance` is a recorded **`discrepancy`**, and anyone
using absolute line intensities should read it: of 48 catalog entries, 37
chemically diverse crystals reuse the same $0.6$ $\AA^2$ value, four legacy values
have no value-level provenance, and the crystal-wide scalar is known to be
insufficient where sites differ or where nonparallel reflections probe an
anisotropic displacement tensor. The scalar approximation is a stated scope
limit, not an oversight, and the schema is deliberately not extended until source
coverage justifies a site- or tensor-level representation.

An optional independent oracle compares $|F_{hkl}|^2$ against a second
implementation under `dans-diffraction-oracle` (`unverified`).

See the [validation ledger](../../validation/physics-validation-ledger.md) and
the [write-up](../../validation/atomic-physics/structure-factor.md). No row here
is human `signed-off`.

Implementation owners: `pyrite.materials.crystal.structure_factor` and
`pyrite.materials.crystal.debye_waller`.
