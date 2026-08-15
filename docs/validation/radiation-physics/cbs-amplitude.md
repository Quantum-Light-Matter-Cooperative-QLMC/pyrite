# Validation: `cbs-amplitude`

## Claim and source

- Claim: `materials/crystal.py::U_g` returns the Fourier component of the
  periodic crystal potential (the coherent-bremsstrahlung / CBS coupling
  amplitude), and `montecarlo/spectrum/lines.py` assembles the coherent amplitude by
  combining the PXR term (`chi_g`) and the CBS term (`U_g`) with a relativistic
  factor carrying a `1/gamma` (or `1/gamma^2`) high-energy suppression.
- Source: Feranchuk, Ulyanenkov, Harada, Spence, "Parametric x-ray radiation
  and coherent bremsstrahlung from nonrelativistic electrons in crystals,"
  *Phys. Rev. E* **62**, 4225 (2000) ("Feranchuk--Spence 2000"). Ledger cites
  Eq. (3) for `chi_g`, Eq. (10),(12) for the line spectrum. The implementation's
  braced CBS assembly additionally cites Feranchuk Eq. (13)/(14) and Zhai (2025)
  SI Eq. (6).
- Intended quantity: the electron-optical potential Fourier component
  `U_g` (an energy) from the crystal's periodic electrostatic potential, and
  the braced factor that weights PXR against CBS with the `1/gamma` suppression
  that becomes numerically important above ~100 keV.

This derivation was written before reading the `U_g` / amplitude-assembly
implementation bodies.

## Independent derivation of `U_g`

### Electrostatic potential of the crystal

Gaussian units. The crystal charge density is nuclei plus bound-electron
clouds, periodic on the lattice:

\[
\rho(\mathbf r)=\sum_{\text{cells}}\sum_j
  \Big[Z_j e\,\delta(\mathbf r-\mathbf R-\mathbf r_j)
       -e\,n_j(\mathbf r-\mathbf R-\mathbf r_j)\Big],
\]

with \(\int n_j\,d^3r = Z_j\). Its reciprocal-lattice Fourier component (per
unit cell volume \(V\)) is

\[
\rho_{\mathbf g}=\frac{e}{V}\sum_j\big[Z_j-f_j(\mathbf g)\big]
  e^{i\mathbf g\cdot\mathbf r_j}\,e^{-W_j},
\qquad
f_j(\mathbf g)=\int n_j(\mathbf r)e^{i\mathbf g\cdot\mathbf r}\,d^3r,
\]

where \(f_j(\mathbf g)\) is the ordinary (non-dispersive) X-ray atomic form
factor and \(\exp(-W_j)\) is the Debye–Waller factor.

### Poisson to potential

\[
\nabla^2\varphi=-4\pi\rho \;\Rightarrow\; -g^2\varphi_{\mathbf g}=-4\pi\rho_{\mathbf g}
\;\Rightarrow\;
\varphi_{\mathbf g}=\frac{4\pi}{g^2}\rho_{\mathbf g}
=\frac{4\pi e}{V g^2}\sum_j\big[Z_j-f_j(\mathbf g)\big]
  e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}.
\]

The beam electron (charge \(-e\)) has potential energy \(U(\mathbf
r)=-e\varphi\), so its Fourier magnitude is

\[
\boxed{\;\big|U_{\mathbf g}\big|=\frac{4\pi e^2}{V g^2}
  \sum_j\big[Z_j-f_j(\mathbf g)\big]e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}\;}
\]

(the physical potential energy carries a leading \(-\); see sign note below).
Using the classical electron radius \(r_e=e^2/(mc^2)\) or, equivalently,
\(e^2=\alpha\hbar c=14.3996\) eV·Å (Gaussian),

\[
U_{\mathbf g}=\pm\frac{4\pi e^2}{V g^2}
  \sum_j\big[Z_j-f_j(\mathbf g)\big]e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}.
\]

This is the Mott--Bethe electron-scattering combination `(Z - f_x)`: nucleus
minus electron cloud. It is the key structural distinction from PXR:

- PXR susceptibility (`chi_g`, verified `pxr-amplitude` row):
  \(\chi_{\mathbf g}\propto\sum_j f_j\) — the **electron** density only, with
  a \(1/k^2\) (photon) denominator.
- CBS potential (`U_g`, this row): \(U_{\mathbf g}\propto\sum_j(Z_j-f_j)\) —
  the **net screened-nucleus** potential, with a \(1/g^2\)
  (momentum-transfer) denominator.

### Anomalous corrections

The electrostatic potential is a static Coulomb quantity, so only the real,
energy-independent charge distribution should enter: \(f_j =
f_{0,j}(\mathbf g)\). The resonant dispersion corrections \(f'(E)\),
\(f''(E)\) are photon-frequency responses and do **not** belong in a static
`U_g`. A physically clean `U_g` uses \(Z_j - f_{0,j}(\mathbf g)\), not
\(Z_j - f_0 - f' - if''\). Flagged as a code check.

## Cheap filters

- **Units.** \(e^2/(Vg^2)\) carries units eV·Å / (Å³·Å⁻²) = eV. `U_g` is an
  energy (eV); \((Z-f)\) is dimensionless. Pass.
- **Forward / small-\(g\) limit.** As \(g\to0\), \(Z-f_0(g)\to0\) like
  \(g^2\) (\(f_0(0)=Z\), neutral atom), cancelling \(1/g^2\) and keeping
  \(U_0\) finite. For a reflection \(g\neq0\), \(f_0(g)<Z\), so
  \(Z-f_0>0\) and `U_g` is finite. Pass.
- **Extinction.** Extinct reflection → phased site sum → `U_g` \(\to0\).
  Pass.
- **Sign / convention.** The electron sits in the attractive field of the net
  positive cores; only `|U_g|` and the PXR/CBS *relative* phase enter
  `|A_PXR + A_CBS|^2`. The absolute sign is a convention that must be
  consistent between the two braced terms. Recorded as a relative-phase check,
  not an absolute-sign check.

### Relativistic braced factor: expected behaviour

Required limiting behaviour, pinned without committing to the paper's exact
algebra:

- **Non-relativistic recovery (\(\gamma\to1\)).** For keV beams \(\gamma
  \approx1\) (\(\gamma=1+T/mc^2\); 30 keV \(\to\gamma=1.059\)); the braced
  factor must reduce to the plain PXR+CBS combination with no
  vanishing/blow-up.
- **High-energy suppression.** The CBS weight (or shared formation factor)
  carries \(1/\gamma\) or \(1/\gamma^2\) and must *decrease* as
  \(\gamma\to\infty\). At 100 keV \(\gamma=1.196\), \(1/\gamma=0.836\),
  \(1/\gamma^2=0.699\) — a 16–30% effect, consistent with "\(1/\gamma\)
  matters ≳100 keV."
- **Physical origin of a \(\gamma\) power.** The photon formation region is
  set by \(\theta_{\rm ph}^2=\gamma^{-2}+|\chi_0|\); a \(\gamma\) factor in
  the CBS amplitude/weight is the standard signature of this
  formation-length physics.

**Ambiguity flagged up front.** The exact algebraic combination of `chi_g` and
`U_g` inside the braces, and whether the suppression is \(1/\gamma\) or
\(1/\gamma^2\) and multiplies CBS alone, PXR, or a shared prefactor, is
specific to Feranchuk--Spence 2000 Eq. (14) / Zhai SI Eq. (6). Without those
equation texts reproduced in-repo I cannot re-derive that exact mapping from
first principles; I verify the limiting behaviour, the \((Z-f)\) vs \(f\)
split, and dimensional consistency, and compare the remaining algebra against
the code.

## Implementation comparison

### `crystal.py::U_g`

Production computes

```text
U_g = 4 pi * E2_EV_ANG * sum_i exp(i 2pi hkl.R_i) * (Z_i - F_i.real) / g^2 * dwf / V_cell
```

with `E2_EV_ANG = alpha*hbar*c = 14.3996 eV·Å`, `dwf = exp(-W)` applied once,
and `F_i` from `_atom_F` (Cromer--Mann `f0` for non-edge atoms, Henke
`f0+f'+i f''` for edge-prone atoms / `use_henke`).

Term-by-term this **matches** the independent expression
`(4 pi e^2 / V g^2) sum (Z - f) exp(i g.r) exp(-W)`:

- prefactor `4 pi e^2 / V`, denominator `1/g^2`, phase `exp(i g.r_j)` (= `exp(i
  2pi hkl.R_j)`), single Debye--Waller `exp(-W)` — all present and correct;
- the Mott--Bethe `(Z - f)` combination is used, not `chi_g`'s `f`.

Divergences from the clean derivation, all minor / convention-level:

1. **Sign.** Production returns `+4 pi e^2 (Z-f)/g^2/V`, i.e. `+e*phi_g`
   (`= -U_electron`), the *magnitude* branch. The physical electron potential
   energy is negative. Because only `|A_PXR + A_CBS|^2` is observable, what
   matters is the relative sign against `chi_g` (which is negative). The braced
   assembly fixes this relative phase explicitly (see below); the absolute
   `U_g` sign is a convention, not an error.
2. **`F.real` for edge-prone atoms.** For those atoms `U_g` uses
   `Z - (f0 + f')`, folding the anomalous *real* dispersion `f'` into a static
   Coulomb potential and (correctly) dropping the imaginary `f''`. This makes
   `U_g` spuriously photon-energy dependent near edges; for non-edge atoms `F =
   f0` and `U_g` is correctly energy-independent (numerically confirmed). Minor
   physical inconsistency limited to edge-prone constituents.

Numeric sanity (10 keV, no independent helper reused beyond `Z`, `f0`):

```text
silicon (111): U_g = 3.68 - 3.70 i eV,  |U_g| = 5.2 eV
lif     (200): U_g = 5.11 + 0 i    eV
diamond (111): U_g = 5.09 - 5.09 i eV,  |U_g| = 7.2 eV
```

These are physically sane crystal-potential Fourier magnitudes (order a few
eV/V); the Si/diamond `(111)` real≈imag split is exactly the two-atom
`(0,0,0)+(1/4,1/4,1/4)` basis phase `exp(i pi (h+k+l)/2) = -i`, and the
centrosymmetric LiF `(200)` is purely real — both as expected.

### `spectrum.py` braced CBS amplitude

```text
eUg_over_m = U_g / M_E_EV                       # dimensionless U_g/(m c^2)
gamma      = 1/sqrt(1-beta^2)
{a;b}      = a.b - (a.v)(b.v)                    # transverse-to-velocity product
braced_ge  = g.e - (g.v)(e.v)                    = {g;e}
braced_kg  = k.g - (k.v)(g.v)                    = {k;g}
A_CBS = -eUg_over_m/(gamma * (v.g)) * ( {g;e} + (v.e){k;g}/(v.g) )
```

paired with `A_PXR = chi/detuning * ((v.(k+g))(g.e) - omega^2 (v.e))` and
`A2 += |A_PXR + A_CBS|^2` summed over polarizations.

Filter results on the assembly:

- **Dimensions.** `U_g/(m c^2)` dimensionless; `{g;e} ~ 1/Å`, `{k;g}/(v.g) ~
  1/Å`, divided by `(v.g) ~ 1/Å` → `A_CBS` dimensionless. `A_PXR` is likewise
  dimensionless (`chi` dimensionless, numerator `1/Å^2` over `detuning ~
  1/Å^2`). The two amplitudes share units and are addable. Pass.
- **High-energy suppression.** The prefactor is \(1/\gamma\) (single power);
  combined with `U_g/(m c^2)` this is \(U_g/(\gamma mc^2)=U_g/E_{\rm
  total}\), the natural CBS deflection scale. `A_CBS` \(\propto1/\gamma\)
  decreases as \(\gamma\to\infty\). Pass; the ledger's "\(1/\gamma^2\) (or
  \(1/\gamma\))" is resolved by the code to **\(1/\gamma\)** (cited to Zhai
  SI Eq. 6 / Feranchuk Eq. 14).
- **Non-relativistic recovery.** \(\gamma\to1\) leaves the braced CBS
  amplitude finite and reduces the prefactor to unity. Pass.
- **Relative PXR/CBS sign.** `A_PXR` carries `chi_g < 0`; `A_CBS` carries an
  explicit leading `-` on `+U_g > 0`. The interference sign is therefore an
  explicit, definite choice in the assembly (consistent, not dangling).

**Not independently certified.** The exact braced form of `A_CBS`
(`{g;e} + (v.e){k;g}/(v.g)` all over `gamma (v.g)`) is Feranchuk Eq. (14) /
Zhai SI Eq. (6). I did not have those equation texts and did not re-derive that
specific tensor structure from the radiated-current integral in this context.
Its units, both limiting cases, the `1/gamma` power, and the transverse
projection `{a;b}=a.b-(a.v)(b.v)` (standard relativistic-current form) all pass,
but the exact prefactor and the `(v.e){k;g}/(v.g)` cross term are **filtered,
not rederived**.

## Verdict

`filtered`.

- **`U_g` (crystal.py):** independently **rederived** from Poisson's equation.
  Prefactor `4 pi e^2/V`, `1/g^2`, `(Z - f)` Mott--Bethe combination, phase,
  single Debye--Waller, and eV units all match; numeric magnitudes are
  physically sane. Two convention-level notes: the returned sign is the
  `+|U_g|` (`-U_electron`) branch (relative phase fixed downstream), and
  edge-prone atoms fold `f'` into the static potential (spurious near-edge
  energy dependence; `f''` correctly dropped).
- **Relativistic braced factor (spectrum.py):** passes all cheap filters —
  dimensionally consistent and addable to `A_PXR`, `1/gamma` high-energy
  suppression correct, `gamma → 1` recovery correct, definite relative sign —
  but its exact algebraic form (Feranchuk Eq. 14 / Zhai SI Eq. 6) is **not
  independently rederived** here because the source equation text was not
  available. Ambiguity in the ledger's "`1/gamma^2` (or `1/gamma`)" is resolved
  by the implementation to a single power `1/gamma`.

Overall the composite claim advances from `unverified` to `filtered`: `U_g`
alone would reach `rederived`, but the braced relativistic assembly cannot be
certified past `filtered` until Feranchuk Eq. (14) / Zhai SI Eq. (6) are
checked against the coded `A_CBS` term-for-term. Not `signed-off` (human only).

## Suggested ledger change

`status: unverified → filtered`; `checks: — → units+limits+signs (U_g rederived
from Poisson; braced 1/gamma factor filtered, exact Eq. 14/Zhai SI Eq. 6 form
not yet independently rederived)`; `anchor: — → docs/validation/radiation-physics/cbs-amplitude.md`.
Consider tightening the note "1/gamma^2 (or 1/gamma)" to state the code uses a
single `1/gamma` power. Human applies.
