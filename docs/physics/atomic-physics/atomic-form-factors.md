# Atomic form factors

Every X-ray coupling in PyRITE — the structure factor, the PXR susceptibility $\chi_{\mathbf g}$, the CBS crystal potential $U_{\mathbf g}$, the forward susceptibility $\chi_0$, the refractive index, and the absorption coefficient — is built from one per-element complex quantity, the atomic form factor. This page defines it, fixes its conventions, and states the domain on which it is valid. Where the numbers come from and why that source was chosen is a separate question, answered in [Atomic data sources](atomic-data-sources.md).

## The quantity

For an isolated neutral atom of a given element, the scattering amplitude for a photon of energy $E$ transferring momentum $\hbar\mathbf g$ is written

```{math}
:label: eq-form-factor-total

F(\mathbf g, E) = f_0(g) + f'(E) + i\,f''(E),
```

in units of the Thomson amplitude $-r_e$ (i.e. "electron units", so a free electron scatters with $F = 1$). The three terms separate cleanly by what they depend on:

* $f_0(g)$ — the **non-resonant** term, the Fourier transform of the atomic electron density. It depends on momentum transfer only, is real, and is energy-independent.
* $f'(E)$ — the real **dispersion** correction, from the finite binding of the atomic electrons. It depends on photon energy only, not on $g$.
* $f''(E)$ — the imaginary **absorption** term, proportional to the photoabsorption cross section through the optical theorem. Also $g$-independent.

The $g$/$E$ factorization in {eq}`eq-form-factor-total` is itself an approximation: the anomalous terms are evaluated in the dipole (forward- scattering) limit and then reused at every $g$. It is standard, and it is excellent for the soft reflections this project works with, where the anomalous terms are small corrections to a slowly varying $f_0$.

## Conventions

These are the repository-wide conventions; getting any of them wrong is a factor-of-$4\pi$ or factor-of-4 trap.

```{list-table} Form-factor conventions used throughout PyRITE.
:name: tbl-form-factor-conventions
:header-rows: 1

* - Symbol
  - Definition
  - Units
* - $g$
  - $|\mathbf g| = 2\pi/d_{hkl}$
  - $\AA^{-1}$
* - $s$
  - $\sin\theta/\lambda = g/(4\pi)$, the crystallographic argument of $f_0$
  - $\AA^{-1}$
* - $E$
  - photon energy
  - eV
* - $f_1$
  - $Z + f'$, the Henke-convention forward-scattering factor
  - electron units
* - $f_2$
  - $f''$, the Henke-convention absorption factor
  - electron units
```

The imaginary part enters {eq}`eq-form-factor-total` with a **positive** sign, $+i f''$, which fixes the passive-medium sign of every downstream quantity: a real material must give $\operatorname{Im}\chi_0 < 0$ and a positive absorption coefficient. The two soft-X-ray tabulation conventions differ only in whether $Z$ is folded into the real part — $f_1 = Z + f'$ versus $f'$ alone — and the implementation is explicit about which one each consumer receives.

## Non-resonant term

$f_0(g)$ is the Waasmaier–Kirfel parameterization,{cite:p}`waasmaier1995` an eleven-parameter Gaussian fit

```{math}
:label: eq-form-factor-f0

f_0(s) = c + \sum_{i=1}^{5} a_i\,e^{-b_i s^2},
\qquad s = \frac{g}{4\pi},
```

fitted to relativistic Hartree–Fock free-atom densities. Two properties of {eq}`eq-form-factor-f0` matter for interpretation:

* **Forward limit.** $f_0(0) = c + \sum_i a_i \approx Z$, exact in principle and reproduced by the fit to about 0.05 % (C: 5.9972 against $Z=6$; W: 73.964 against $Z=74$). This residual is the reason $\chi_0$ deliberately builds its forward sum from $Z + f'$ rather than from $f_0(0) + f'$ — see [Photon escape and in-medium dispersion](../radiation-physics/photon-escape-and-dispersion.md).
* **Fit range.** The Waasmaier–Kirfel coefficients are fitted over $0 \le s \le 6$ $\AA^{-1}$, a far wider window than the Cromer–Mann fit they replaced ($s \le 2$ $\AA^{-1}$). PyRITE's reflection search caps momentum transfer at $g \le 8$ $\AA^{-1}$, i.e. $s \le 0.637$ $\AA^{-1}$, so every reflection the code can select sits deep inside the fitted region.

$f_0$ describes a **spherical, neutral, isolated** atom. Bonding charge redistribution, ionic charge states, and aspherical valence density are not represented. For the low-order reflections that dominate coherent emission this is the standard independent-atom approximation and is accurate at the percent level; it degrades for the very lowest-order reflections of strongly ionic crystals.

## Anomalous terms

$f'(E)$ and $f''(E)$ come from the Chantler/FFAST self-consistent Dirac–Hartree–Fock calculation.{cite:p}`chantler1995,chantler2000,ffast,xraydb` They are returned as the anomalous parts **directly**, not in the Henke $f_1 = Z + f'$ form; the adapter adds $Z$ back where a consumer wants $f_1$.

Both terms vary rapidly across absorption edges: $f''$ jumps discontinuously upward when a new shell opens, and $f'$ shows the associated Kramers–Kronig dispersion dip just below the edge. Because catalog line grids run from roughly 350 eV to 5 keV, they cross the $L$ edges of the first-row transition metals and the $M$ edges of the heavy metals, so edge structure is not a corner case here — it is squarely inside the working band.

### Tabulated domain

Each element's Chantler table spans roughly 1.01 eV to 966 keV. PyRITE evaluates only on the **strict interior** of that range, for two reasons: the spline is unstable at the exact endpoints, and the bremsstrahlung energy grid starts at $E = 0$, which must read as out-of-domain anyway.

The out-of-range contract is deliberate: energies outside the interior return `NaN` with the array shape and index alignment preserved, rather than raising or silently clamping. Downstream kernels apply an explicit `nan_to_num` policy at the point where a physical zero is meant. A strict mode that raises instead is available for callers that want to assert their grid is in range.

### Where the anomalous terms are used

Not every consumer wants the full complex $F$. The policy is:

```{list-table} Form-factor variant by consumer.
:name: tbl-form-factor-policy
:header-rows: 1

* - Consumer
  - Variant
  - Reason
* - Line-spectrum couplings $\chi_{\mathbf g}$, $U_{\mathbf g}$
  - $f_0 + f' + i f''$
  - resonant corrections are switched on by default in the production spectrum path
* - $\chi_0$, refractive index
  - $Z + f' + i f''$
  - $f'$ *is* the real refractive correction; dropping it leaves only the Thomson term
* - Absorption length, optical constants
  - $f_2 = f''$
  - $\mu = 2 r_e \lambda\, n\, f_2$
* - Non-resonant structure-factor calls
  - $f_0$ only, plus $f_0+f'+if''$ for edge-prone elements
  - fallback policy, see below
```

When a caller explicitly asks for the non-resonant treatment, an element-level override still forces the full complex factor for a hard-coded **edge-prone** set — P, Si, Fe, Ge, Mo, Nb, Se, Te, Ta, Re, Bi — whose edges are known to fall inside the 350–3500 eV catalog line grids. This is a policy, not data: it exists so that a non-resonant call cannot quietly produce a badly wrong coupling for a crystal whose constituent has an edge in band.

## Tabulation and interpolation

The anomalous terms depend only on `(element, energy grid)`, never on $g$ or $hkl$, but the per-reflection spectrum loop re-requests the same pair once per reflection. They are therefore memoized on the exact energy-array bytes and returned frozen read-only, collapsing the repeats to one spline pass per element.

The spectrum kernels do not evaluate the tables per energy bin either. They build a shared tabulation grid — a 1 eV uniform mesh unioned with the **native Chantler nodes** of every basis and absorber element in range, plus the EPDL2025 knots and edge node pairs of every absorber (the escape $\mu$ comes from EPDL; see [Photon escape](../radiation-physics/photon-escape-and-dispersion.md)) — evaluate the couplings and $\mu$ once on it, and interpolate at each segment's own resonance energy. Including the native nodes is what keeps the edge jumps resolved: a plain 1 eV mesh would alias a discontinuity that the tables place exactly.

## Assumptions and limits

* Isolated neutral atoms, spherical charge density, no bonding or ionic corrections.
* Anomalous terms evaluated in the dipole limit and applied at all $g$; no $g$-dependence of $f'$, $f''$.
* No nuclear Thomson term, no Delbrück scattering, no magnetic scattering.
* Elastic (Rayleigh) amplitude only; incoherent Compton scattering is not part of $F$ and is not carried by the X-ray couplings.
* Free-atom tables at nominal conditions: no explicit temperature dependence (thermal motion enters separately through the [Debye–Waller factor](../materials/structure-factor.md)).
* Values outside the tabulated energy interior are `NaN`, not extrapolated.

## Validation

`Validation: atomic-form-factor` (`rederived`) covers {eq}`eq-form-factor-total` — units, the $s = g/(4\pi)$ argument, the $+if''$ convention, the forward limit, and exact agreement with three direct source points. The consumers each carry their own rows: `structure-factor`, `absorption-length`, `grazing-optical-constants`, `xray-chi-zero`, and `xray-refractive-index`. An optional independent `Dans_Diffraction` oracle compares $|F_{hkl}|^2$ against a second implementation under `dans-diffraction-oracle`; its cross-database dispersive spread is bounded at 10 %, which is the honest scale of "how well do two anomalous-scattering databases agree".

See the [validation ledger](../../validation/physics-validation-ledger.md) and the [write-up](../../validation/atomic-physics/atomic-form-factor.md). No row here is human `signed-off`.

Implementation owner: `pyrite.materials.atomic` — `atomic_form_factor`, `cromer_mann_f0`, `henke_dispersion`, `load_henke`, `Z_TABLE`. The names `cromer_mann_f0` and `henke_dispersion` are retained API spellings from the pre-migration implementation; the data behind them is Waasmaier–Kirfel and Chantler.
