# Structure factor and Debye–Waller

The structure factor is the bridge between the crystal's atomic arrangement and its X-ray couplings. It is the one place where the per-element [atomic form factors](../atomic-physics/atomic-form-factors.md) and the [unit-cell geometry](crystal-structure.md) are combined, and both PXR and CBS couplings are built from its output.

## Definition

For a reflection $hkl$ with reciprocal vector $\mathbf g$ and photon energy $E$,

```{math}
:label: eq-structure-factor-definition

S(\mathbf g, E) = \sum_j F_j(\mathbf g, E)\,
e^{\,i 2\pi\,(hkl)\cdot\mathbf R_j}\,
e^{-W_j},
```

summed over the basis sites $j$ of the unit cell, where $\mathbf R_j$ is the **fractional** position of site $j$, $F_j$ is that element's complex atomic form factor, and $e^{-W_j}$ is its amplitude Debye–Waller factor. The phase is written with fractional coordinates and $2\pi$ explicit, which is identical to $e^{i\mathbf g\cdot\mathbf r_j}$ in Cartesian form and avoids a second place where the $|\mathbf g| = 2\pi/d$ convention could be misapplied.

$S$ is complex and dimensionless (electron units). It is returned together with $|\mathbf g|$, because every consumer needs both and recomputing $|\mathbf g|$ is the kind of duplication that invites a convention mismatch.

## Debye–Waller factor

Thermal motion smears each atom over a distribution of positions, which reduces the coherent amplitude at momentum transfer $\mathbf g$:

```{math}
:label: eq-structure-factor-debye-waller

e^{-W},
\qquad
W = B\left(\frac{\sin\theta}{\lambda}\right)^{2}
  = B\,s^2
  = \frac{B\,g^2}{16\pi^2},
```

with the tabulated isotropic $B$ factor in $\AA^2$, related to the mean-square displacement by $B = 8\pi^2\langle u_x^2\rangle$.

Two normalization points in {eq}`eq-structure-factor-debye-waller`:

* The exponent is $g^2/16\pi^2$, **not** $g^2/4\pi^2$. This follows directly from $s = g/(4\pi)$ under the $g = 2\pi/d$ convention.
* {eq}`eq-structure-factor-debye-waller` is the **amplitude** factor. Intensities carry its square, $e^{-2W}$. The implementation applies the amplitude form inside $S$, so any $|S|^2$ automatically carries $e^{-2W}$ exactly once.

One scalar $B$ is applied to **every** site in the cell. There is no per-site $B$, no anisotropic displacement tensor, and no anharmonic correction.

## Per-site form factor policy

Which variant of $F_j$ enters {eq}`eq-structure-factor-definition` is a policy, described in full in [Atomic form factors](../atomic-physics/atomic-form-factors.md). In short: the production line-spectrum path uses the full complex $f_0 + f' + i f''$; a caller that explicitly asks for the non-resonant treatment still gets the complex factor for the hard-coded edge-prone element set, so a crystal with an in-band absorption edge cannot quietly be evaluated without its anomalous terms. Form factors are computed once per **unique element**, not once per site, since $F$ depends only on the element and $|\mathbf g|$.

## Measured low-order structure factors (bonding charge)

The independent-atom sum misses the valence charge that covalent bonding piles up between neighbours. In silicon and diamond this changes the lowest reflections by up to a few percent and gives the independent-atom-extinct $(222)$ reflection a small nonzero value. A catalog crystal may therefore carry a `[measured_structure_factors]` table of measured dynamic structure factors, and every structure-factor sum then adds the static correction

```{math}
:label: eq-structure-factor-bonding-charge

\Delta F(\mathbf g) = F_{\rm meas}(\mathbf g)\,e^{+B_{\rm src}s^2} - F_{\rm IAM}^{\rm static}(\mathbf g),
\qquad
S \to S + \Delta F(\mathbf g)\,e^{-W}.
```

Here $B_{\rm src}$ is the Debye–Waller factor the source's dynamic values carry, and $e^{-W}$ is the caller's own factor: the catalog $B$ for $\chi_{\mathbf g}$, and the phonon-DOS $B(T)$ for channeling. Values are tabulated per atom (`dynamic-per-atom`), as in the sources. An allowed reflection keeps the independent-atom phase, $F_{\rm meas,cell} = F\,P(\mathbf g)$ with $P = \sum_j e^{i\mathbf g\cdot\mathbf R_j}$. An extinct one, with $P = 0$, takes the phase of the source's centrosymmetric origin $\mathbf o$, $F_{\rm meas,cell} = N F e^{i\mathbf g\cdot\mathbf o}$. For silicon and diamond $\mathbf o$ is a bond centre, so $(222)$ has the phase of positive charge in the bonds. Each tabulated reflection stands for its whole $m\bar 3m$ family, so the table is accepted only for cubic, single-element crystals.

The same $\Delta F$ enters the Mott–Bethe potentials as $Z - f_0 \to Z - f_0 - \Delta F$: in $U_{\mathbf g}$, in the planar channeling coefficients $U_n$ and in the axial string potential. The anomalous $f'$ and $f''$ stay in the per-atom factors, because the sources correct their values for anomalous dispersion. Every structure-factor function takes `bonding_charge=False` to return the pure independent-atom sum.

| Crystal | Source | Effect |
|---|---|---|
| Si | Cummings & Hart (1988) Pendellösung set; $F(222)$ from Alkire, Yelon & Schneider (1982); tabulated by Lu, Zunger & Deutsch (1993), Table III | $F(111)$ +1.8%; $\lvert F_{\rm cell}(222)\rvert$ = 1.456 e; $U_{111}$, $U_{222}$ match CBED (Zuo 1999) to 0.1% and 2% |
| diamond | Takama et al. (1990) Pendellösung; $F(222)$ from Weiss & Middleton; LZD Table V | $F(111)$ +8%; (111) planar well 6.5% shallower; Klein et al. (1985) bonding-corrected (111) channeling lines reproduced |

Identity: the crystal-conditional `structure_factor_model` marker joins dataset and case digests only for crystals that carry the table. Validation: `bonding-charge-structure-factors`.

## What the structure factor feeds

| Quantity | Relation | Documented in |
|---|---|---|
| PXR susceptibility $\chi_{\mathbf g}$ | $-r_e\lambda^2 S/(\pi V_{\rm cell})$ | [Coherent PXR and CBS radiation](../radiation-physics/coherent-radiation.md) |
| CBS potential $U_{\mathbf g}$ | same sum with $Z_j - f_j$ and a $1/g^2$ denominator | [Coherent PXR and CBS radiation](../radiation-physics/coherent-radiation.md) |
| Forward susceptibility $\chi_0$ | {eq}`eq-structure-factor-definition` at $\mathbf g = 0$ | [Photon escape and in-medium dispersion](../radiation-physics/photon-escape-and-dispersion.md) |
| Reflection ranking | $\lvert S\rvert/g^2$ as a proxy for $\lvert\chi_{\mathbf g}\rvert$ | [Reflection selection](reflection-selection.md) |

In the $\mathbf g \to 0$ limit every phase factor and every Debye–Waller factor reduces to 1, so $S(0)$ collapses to the plain forward sum over the cell. The implementation of $\chi_0$ takes that limit analytically rather than calling {eq}`eq-structure-factor-definition` with $\mathbf g = 0$, so that it can use $Z + f'$ and share one normalization exactly with the absorption coefficient instead of inheriting the $f_0(0) \approx Z$ fit residual.

## Limiting cases

| Limit | Result |
|---|---|
| $B \to 0$ | $e^{-W}\to 1$; the sum becomes the purely static structure factor |
| $\mathbf g \to 0$ | $S \to \sum_j F_j(0)$, the forward-scattering sum over the cell |
| destructive basis phases | $S \to 0$; the reflection is extinct and contributes nothing |
| $f', f'' \to 0$ | $S$ becomes real up to the basis phases; no anomalous contribution |
| single-atom basis at the origin | $S = F(g)e^{-W}$ |

Extinction follows from the phase convention: for diamond, {eq}`eq-structure-factor-definition` must give exactly zero for $(200)$ and $(222)$ and nonzero for $(111)$, $(220)$, $(311)$, $(400)$. This fixes the phase sign and the fractional-coordinate convention.

## Assumptions and limits

* **Kinematic** (single-scattering) theory. No dynamical diffraction, no extinction corrections, no multiple-beam effects. This suits thin films and mosaic crystals; it is a limit for thick, high-quality perfect crystals at strong reflections.
* Independent, spherical, neutral atoms — the form-factor approximation carried in from the atomic layer — except for the reflections a crystal's measured structure-factor table lists (Si and diamond; see above). Reflections beyond that table use the independent-atom sum.
* A single isotropic scalar $B$ per crystal, applied to every site.
* No thermal diffuse scattering: the intensity removed from the Bragg peak by {eq}`eq-structure-factor-debye-waller` is not re-emitted anywhere in the model.
* No temperature knob. $B$ is a static catalog value; a sample at a different temperature from the source refinement is not represented.

## Implementation

`pyrite.materials.crystal.structure_factor` evaluates {eq}`eq-structure-factor-definition` and `pyrite.materials.crystal.debye_waller` evaluates {eq}`eq-structure-factor-debye-waller`. Both are checked against the units, the $\mathbf g\to0$ and $B\to0$ limits, the phase sign, the $16\pi^2$ exponent, and the diamond extinct/allowed set above. The validation status of each claim is tracked in the [validation ledger](../../validation/physics-validation-ledger.md) and the [structure-factor write-up](../../validation/atomic-physics/structure-factor.md).
