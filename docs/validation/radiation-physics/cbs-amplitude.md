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

$$
\rho(\mathbf r)=\sum_{\text{cells}}\sum_j
  \Big[Z_j e\,\delta(\mathbf r-\mathbf R-\mathbf r_j)
       -e\,n_j(\mathbf r-\mathbf R-\mathbf r_j)\Big],
$$

with $\int n_j\,d^3r = Z_j$. Its reciprocal-lattice Fourier component (per
unit cell volume $V$) is

$$
\rho_{\mathbf g}=\frac{e}{V}\sum_j\big[Z_j-f_j(\mathbf g)\big]
  e^{i\mathbf g\cdot\mathbf r_j}\,e^{-W_j},
\qquad
f_j(\mathbf g)=\int n_j(\mathbf r)e^{i\mathbf g\cdot\mathbf r}\,d^3r,
$$

where $f_j(\mathbf g)$ is the ordinary (non-dispersive) X-ray atomic form
factor and $\exp(-W_j)$ is the Debye–Waller factor.

### Poisson to potential

$$
\nabla^2\varphi=-4\pi\rho \;\Rightarrow\; -g^2\varphi_{\mathbf g}=-4\pi\rho_{\mathbf g}
\;\Rightarrow\;
\varphi_{\mathbf g}=\frac{4\pi}{g^2}\rho_{\mathbf g}
=\frac{4\pi e}{V g^2}\sum_j\big[Z_j-f_j(\mathbf g)\big]
  e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}.
$$

The beam electron (charge $-e$) has potential energy $U(\mathbf
r)=-e\varphi$, so its Fourier magnitude is

$$
\boxed{\;\big|U_{\mathbf g}\big|=\frac{4\pi e^2}{V g^2}
  \sum_j\big[Z_j-f_j(\mathbf g)\big]e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}\;}
$$

(the physical potential energy carries a leading $-$; see sign note below).
Using the classical electron radius $r_e=e^2/(mc^2)$ or, equivalently,
$e^2=\alpha\hbar c=14.3996$ eV·Å (Gaussian),

$$
U_{\mathbf g}=\pm\frac{4\pi e^2}{V g^2}
  \sum_j\big[Z_j-f_j(\mathbf g)\big]e^{i\mathbf g\cdot\mathbf r_j}e^{-W_j}.
$$

This is the Mott--Bethe electron-scattering combination `(Z - f_x)`: nucleus
minus electron cloud. It is the key structural distinction from PXR:

- PXR susceptibility (`chi_g`, verified `pxr-amplitude` row):
  $\chi_{\mathbf g}\propto\sum_j f_j$ — the **electron** density only, with
  a $1/k^2$ (photon) denominator.
- CBS potential (`U_g`, this row): $U_{\mathbf g}\propto\sum_j(Z_j-f_j)$ —
  the **net screened-nucleus** potential, with a $1/g^2$
  (momentum-transfer) denominator.

### Anomalous corrections

The electrostatic potential is a static Coulomb quantity, so only the real,
energy-independent charge distribution should enter: $f_j =
f_{0,j}(\mathbf g)$. The resonant dispersion corrections $f'(E)$,
$f''(E)$ are photon-frequency responses and do **not** belong in a static
`U_g`. A physically clean `U_g` uses $Z_j - f_{0,j}(\mathbf g)$, not
$Z_j - f_0 - f' - if''$. Flagged as a code check.

## Cheap filters

- **Units.** $e^2/(Vg^2)$ carries units eV·Å / (Å³·Å⁻²) = eV. `U_g` is an
  energy (eV); $(Z-f)$ is dimensionless. Pass.
- **Forward / small-$g$ limit.** As $g\to0$, $Z-f_0(g)\to0$ like
  $g^2$ ($f_0(0)=Z$, neutral atom), cancelling $1/g^2$ and keeping
  $U_0$ finite. For a reflection $g\neq0$, $f_0(g)<Z$, so
  $Z-f_0>0$ and `U_g` is finite. Pass.
- **Extinction.** Extinct reflection → phased site sum → `U_g` $\to0$.
  Pass.
- **Sign / convention.** The electron sits in the attractive field of the net
  positive cores; only `|U_g|` and the PXR/CBS *relative* phase enter
  `|A_PXR + A_CBS|^2`. The absolute sign is a convention that must be
  consistent between the two braced terms. Recorded as a relative-phase check,
  not an absolute-sign check.

### Relativistic braced factor: expected behaviour

Required limiting behaviour, pinned without committing to the paper's exact
algebra:

- **Non-relativistic recovery ($\gamma\to1$).** For keV beams $\gamma
  \approx1$ ($\gamma=1+T/mc^2$; 30 keV $\to\gamma=1.059$); the braced
  factor must reduce to the plain PXR+CBS combination with no
  vanishing/blow-up.
- **High-energy suppression.** The CBS weight (or shared formation factor)
  carries $1/\gamma$ or $1/\gamma^2$ and must *decrease* as
  $\gamma\to\infty$. At 100 keV $\gamma=1.196$, $1/\gamma=0.836$,
  $1/\gamma^2=0.699$ — a 16–30% effect, consistent with "$1/\gamma$
  matters ≳100 keV."
- **Physical origin of a $\gamma$ power.** The photon formation region is
  set by $\theta_{\rm ph}^2=\gamma^{-2}+|\chi_0|$; a $\gamma$ factor in
  the CBS amplitude/weight is the standard signature of this
  formation-length physics.

**Ambiguity flagged up front.** The exact algebraic combination of `chi_g` and
`U_g` inside the braces, and whether the suppression is $1/\gamma$ or
$1/\gamma^2$ and multiplies CBS alone, PXR, or a shared prefactor, is
specific to Feranchuk--Spence 2000 Eq. (14) / Zhai SI Eq. (6). Without those
equation texts reproduced in-repo I cannot re-derive that exact mapping from
first principles; I verify the limiting behaviour, the $(Z-f)$ vs $f$
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
- **High-energy suppression.** The prefactor is $1/\gamma$ (single power);
  combined with `U_g/(m c^2)` this is $U_g/(\gamma mc^2)=U_g/E_{\rm
  total}$, the natural CBS deflection scale. `A_CBS` $\propto1/\gamma$
  decreases as $\gamma\to\infty$. Pass; the ledger's "$1/\gamma^2$ (or
  $1/\gamma$)" is resolved by the code to **$1/\gamma$** (cited to Zhai
  SI Eq. 6 / Feranchuk Eq. 14).
- **Non-relativistic recovery.** $\gamma\to1$ leaves the braced CBS
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

---

## 2026-08-15 — second pass: the braced `A_CBS` prefactor and the PXR+CBS cross term

Everything above is the first pass. It is unchanged. This section addresses only
the item that pass left open: the exact vector/tensor structure multiplying
$U_{\mathbf g}$ in `A_CBS`, the power of $1/\gamma$, the placement of the
$(\mathbf v\cdot\mathbf g)$ denominators, and the relative sign between
`A_PXR` and `A_CBS` when the two are summed before squaring. The already
adjudicated `U_g` result (independently rederived from Poisson's equation) is
taken as given and is not revisited.

### Source availability — stated plainly

Neither Feranchuk–Spence 2000 Eq. (14) nor Zhai 2025 SI Eq. (6) is reproduced
anywhere in this repository: there is no PDF, no quoted equation text, and no
transcription of either equation in `docs/`, `checks/`, or the docstrings. The
only in-repo statement of the source equation is the `U_g` docstring's
"`e U_g / (m V)` of Eq. (14)", which fixes the dimensionless coupling but not
the tensor prefactor.

Rather than guess at the paper's algebra, this pass derives the CBS amplitude
from first principles — the exact relativistic equation of motion in a periodic
potential plus the Liénard–Wiechert spectral radiation integral — and derives
`A_PXR` in the *same* normalization so that the relative sign is fixed by the
derivation and not by convention. What this can and cannot certify is stated in
[What remains blocked](#what-remains-blocked).

### Independence statement

The derivation below was carried out from the source-level physics, the coupling
signature, and the resonance condition. The first pass' pseudocode sketch of the
assembly was visible in this file before the derivation was written, so this is
not a blind re-derivation of the *shape* of the expression; it is an independent
derivation of every factor, exponent, denominator, and sign in it, followed by a
numeric certification (below) that does not use any PyRITE code.

### Conventions fixed before deriving

Gaussian units, $c=1$ (so $\omega$, $k$, and $g$ are all inverse lengths and
$m$ means $mc^2$ in eV). Both couplings are computed in `crystal.py` with the
same site phase $e^{+i\mathbf g\cdot\mathbf r_j}$, which is the coefficient
convention for the expansions

$$
U(\mathbf r)=\sum_{\mathbf g}U_{\mathbf g}\,e^{-i\mathbf g\cdot\mathbf r},
\qquad
\chi(\mathbf r)=\chi_0+\sum_{\mathbf g\neq0}\chi_{\mathbf g}\,e^{-i\mathbf g\cdot\mathbf r}.
$$

With this convention the emission resonance comes out as
$\omega=\mathbf v\cdot\mathbf g/(1-\mathbf v\cdot\hat{\mathbf n})$, which is the
resonance the code uses, so the convention is self-consistent and is the one
adopted throughout. Field/time Fourier convention:
$f(\mathbf q,\omega)=\int dt\,d^3r\,f\,e^{i\omega t-i\mathbf q\cdot\mathbf r}$.

$U_{\mathbf g}^{\rm phys}$ denotes the Fourier coefficient of the beam
electron's potential *energy*; the code's `U_g` returns $+e\varphi_{\mathbf g}$,
i.e. $U_{\mathbf g}^{\rm code}=-U_{\mathbf g}^{\rm phys}$ (first pass, note 1;
re-confirmed at `crystal.py:344-364`, which returns
$+4\pi e^2\sum_j(Z_j-F_j)e^{i\mathbf g\cdot\mathbf r_j}e^{-W}/(Vg^2)$).

### Perturbed trajectory: where the single $1/\gamma$ and the projector come from

The force on the beam electron along the unperturbed straight line
$\mathbf r_0=\mathbf v t$ is

$$
\mathbf F=-\nabla U=\sum_{\mathbf g}i\,\mathbf g\,U_{\mathbf g}\,
e^{-i(\mathbf g\cdot\mathbf v)t}.
$$

Relativistic dynamics with $\mathbf p=\gamma m\mathbf v$ gives, exactly,

$$
\frac{d\mathbf p}{dt}=\mathbf F
\;\Longrightarrow\;
\mathbf a=\frac{1}{\gamma m}\Big[\mathbf F-\boldsymbol\beta\,(\boldsymbol\beta\cdot\mathbf F)\Big]
=\frac{1}{\gamma m}\big(\mathbb 1-\boldsymbol\beta\boldsymbol\beta\big)\cdot\mathbf F .
$$

This is the whole $\gamma$ story. The prefactor carries **one** power of
$1/\gamma$; the familiar extra $1/\gamma^2$ for a longitudinal force is
*inside* the projector, not a separate factor — check the two extremes:
$\mathbf F\parallel\mathbf v$ gives $a=F(1-\beta^2)/(\gamma m)=F/(\gamma^3m)$,
and $\mathbf F\perp\mathbf v$ gives $a=F/(\gamma m)$. So writing the amplitude
with $1/\gamma^2$ *and* the projector would double-count the longitudinal
suppression; $1/\gamma$ with the projector is the correct pairing.

Define the projected reciprocal vector and the code's brace product

$$
\mathbf G\equiv\big(\mathbb 1-\boldsymbol\beta\boldsymbol\beta\big)\cdot\mathbf g
=\mathbf g-\boldsymbol\beta(\boldsymbol\beta\cdot\mathbf g),
\qquad
\{\mathbf a;\mathbf b\}\equiv\mathbf a\cdot\mathbf b-(\mathbf a\cdot\mathbf v)(\mathbf b\cdot\mathbf v),
$$

so that $\mathbf G\cdot\boldsymbol\epsilon=\{\mathbf g;\boldsymbol\epsilon\}$ and
$\mathbf k\cdot\mathbf G=\{\mathbf k;\mathbf g\}$ — the two braces are not two
independent structures, they are one projector contracted twice.

Integrating once and twice in time (the steady-state particular solution, no
free drift):

$$
\delta\mathbf v_{\mathbf g}(t)=-\frac{U_{\mathbf g}\,\mathbf G}
{\gamma m\,(\mathbf g\cdot\mathbf v)}e^{-i(\mathbf g\cdot\mathbf v)t},
\qquad
\delta\mathbf r_{\mathbf g}(t)=-\,i\,\frac{U_{\mathbf g}\,\mathbf G}
{\gamma m\,(\mathbf g\cdot\mathbf v)^2}e^{-i(\mathbf g\cdot\mathbf v)t}.
$$

**The two $(\mathbf v\cdot\mathbf g)$ denominators are counted here**: one power
for the velocity response, two for the displacement response. Nothing else in
the problem produces a $(\mathbf v\cdot\mathbf g)$.

### Radiation integral

For polarization $\boldsymbol\epsilon\perp\hat{\mathbf n}$, Jackson Eq. (14.65)
with $\boldsymbol\epsilon^*\!\cdot(\hat{\mathbf n}\times(\hat{\mathbf n}\times\boldsymbol\beta))=-\boldsymbol\epsilon^*\!\cdot\boldsymbol\beta$ gives the
spectral amplitude

$$
I_{\boldsymbol\epsilon}=\int dt\;\big(-\boldsymbol\epsilon^*\!\cdot\boldsymbol\beta(t)\big)\,
e^{i\omega(t-\hat{\mathbf n}\cdot\mathbf r(t))}.
$$

Expanding to first order in the perturbation, with
$\boldsymbol\beta=\boldsymbol\beta_0+\delta\mathbf v$ and
$\mathbf r=\mathbf v t+\delta\mathbf r$,

$$
I_{\boldsymbol\epsilon}=-\int dt\,
\Big[\boldsymbol\epsilon^*\!\cdot\delta\mathbf v(t)
-i\omega(\boldsymbol\epsilon^*\!\cdot\boldsymbol\beta_0)\,
\hat{\mathbf n}\cdot\delta\mathbf r(t)\Big]e^{i\omega(1-\hat{\mathbf n}\cdot\boldsymbol\beta)t}
+(\text{non-radiating zeroth order}).
$$

The $\mathbf g$ harmonic makes the total phase
$\exp\!\big[i\big(\omega(1-\hat{\mathbf n}\cdot\boldsymbol\beta)-\mathbf g\cdot\mathbf v\big)t\big]$,
whose stationarity is exactly the coded resonance
$\omega=\mathbf v\cdot\mathbf g/(1-\mathbf v\cdot\hat{\mathbf n})$, equivalently
$\omega=\mathbf v\cdot(\mathbf k+\mathbf g)$. Substituting $\delta\mathbf v$ and
$\delta\mathbf r$, and using $\omega\hat{\mathbf n}=\mathbf k$:

$$
\boxed{\;
A_{\rm CBS}\;\propto\;
\frac{U^{\rm phys}_{\mathbf g}}{\gamma\,mc^2\,(\mathbf v\cdot\mathbf g)}
\left[\{\mathbf g;\boldsymbol\epsilon\}
+(\mathbf v\cdot\boldsymbol\epsilon)\,
\frac{\{\mathbf k;\mathbf g\}}{\mathbf v\cdot\mathbf g}\right]\;}
$$

with the proportionality constant $4\pi i\,q_e\,\omega/c^2$ shared with the PXR
term below. Every disputed feature is now fixed by derivation: the coupling is
$U_{\mathbf g}/(\gamma mc^2)$ with a **single** $1/\gamma$; the tensor structure
is the transverse projector contracted with $\boldsymbol\epsilon$ and with
$\mathbf k$; there is **one** overall $1/(\mathbf v\cdot\mathbf g)$ and a
**second** one only on the cross term; and the two braces enter with a **plus**
sign relative to each other. The second term is the retardation/phase-modulation
term $-i\omega\,\hat{\mathbf n}\cdot\delta\mathbf r$ beating against the
*unperturbed* velocity, which is why it carries $(\mathbf v\cdot\boldsymbol\epsilon)$
rather than a second projector.

### PXR in the same normalization: the relative sign

The relative sign cannot be read off `A_CBS` alone, so `A_PXR` is derived here
from the same wave equation. Collecting the perturbed electron current and the
$\delta\chi$ polarization current on one right-hand side,

$$
\big(k^2-\varepsilon_0\omega^2/c^2\big)\,\boldsymbol\epsilon^*\!\cdot\mathbf E
=\frac{4\pi i\omega}{c^2}\,\boldsymbol\epsilon^*\!\cdot\mathbf J_e^{(1)}
+\frac{\omega^2}{c^2}\,\chi_{\mathbf g}\;
\boldsymbol\epsilon^*\!\cdot\mathbf E^{(0)}(\mathbf k+\mathbf g,\omega),
$$

so both mechanisms are sources for the *same* outgoing wave and their relative
sign is unambiguous. The Coulomb field of the uniformly moving charge in the
mean medium is

$$
\mathbf E^{(0)}(\mathbf q,\omega)=4\pi i\,\rho_{\mathbf q}\,
\frac{\omega\mathbf v/c^2-\mathbf q/\varepsilon_0}{q^2-\varepsilon_0\omega^2/c^2},
\qquad
\rho_{\mathbf q}=q_e\,2\pi\delta(\omega-\mathbf q\cdot\mathbf v).
$$

At $\mathbf q=\mathbf k+\mathbf g$ transversality gives
$\boldsymbol\epsilon^*\!\cdot\mathbf q=\boldsymbol\epsilon^*\!\cdot\mathbf g$,
the delta reproduces the same resonance, and the denominator is
$|\mathbf k+\mathbf g|^2-\varepsilon_0\omega^2/c^2=g^2+2\,\mathbf k\cdot\mathbf g$
(using $k^2=\varepsilon_0\omega^2/c^2$ in the medium) — the code's `detuning`.
Dividing both sources by the common $4\pi i\,q_e\,2\pi\delta\,(\omega/c^2)$ and
setting $\varepsilon_0\to1$:

$$
S_{\rm PXR}=\omega\,\chi_{\mathbf g}\,
\frac{\omega(\mathbf v\cdot\boldsymbol\epsilon)-(\mathbf g\cdot\boldsymbol\epsilon)}
{g^2+2\mathbf k\cdot\mathbf g},
\qquad
S_{\rm CBS}=-\frac{U^{\rm phys}_{\mathbf g}}{\gamma m(\mathbf v\cdot\mathbf g)}
\left[\{\mathbf g;\boldsymbol\epsilon\}+(\mathbf v\cdot\boldsymbol\epsilon)
\frac{\{\mathbf k;\mathbf g\}}{\mathbf v\cdot\mathbf g}\right].
$$

Using $\omega=\mathbf v\cdot(\mathbf k+\mathbf g)$ on resonance,

$$
-S_{\rm PXR}=\frac{\chi_{\mathbf g}}{g^2+2\mathbf k\cdot\mathbf g}
\Big[\big(\mathbf v\cdot(\mathbf k+\mathbf g)\big)(\mathbf g\cdot\boldsymbol\epsilon)
-\omega^2(\mathbf v\cdot\boldsymbol\epsilon)\Big],
\qquad
-S_{\rm CBS}=-\frac{U^{\rm code}_{\mathbf g}}{\gamma mc^2\,(\mathbf v\cdot\mathbf g)}
\left[\cdots\right],
$$

where the second identity used $U^{\rm code}_{\mathbf g}=-U^{\rm phys}_{\mathbf g}$.
Both physical sources therefore map onto the coded amplitudes with **the same**
overall factor $-1$:

$$
A^{\rm code}_{\rm PXR}+A^{\rm code}_{\rm CBS}=-\big(S_{\rm PXR}+S_{\rm CBS}\big),
$$

and $\lvert A^{\rm code}_{\rm PXR}+A^{\rm code}_{\rm CBS}\rvert^2$ is the correct
interference. The sign chain that makes this work is worth tabulating, because
two sign flips cancel:

| step | effect |
| --- | --- |
| `crystal.py::U_g` returns $+e\varphi_{\mathbf g}$ | $U^{\rm code}_{\mathbf g}=-U^{\rm phys}_{\mathbf g}$ (flip 1) |
| `lines.py` writes `A_CBS = -eUg_over_m/(gamma*vdg)*(...)` | explicit leading $-$ (flip 2) |
| net | $A^{\rm code}_{\rm CBS}=-S_{\rm CBS}$ |
| `lines.py` writes `A_PXR = chi/detuning*((v.(k+g))(g.e) - k^2 (v.e))` | $=-S_{\rm PXR}$ |
| relative sign | **correct**; the common $-1$ is annihilated by $\lvert\cdot\rvert^2$ |

Numerically the two couplings are close to antiphase at a real reflection —
$\arg(U_{\mathbf g}/\chi_{\mathbf g})=2.995$ rad for silicon $(111)$ and
$2.930$ rad for HOPG $(002)$ (deviation from $\pi$ is the absorptive $f''$ in
$\chi_{\mathbf g}$) — which is the expected consequence of
$U_{\mathbf g}\propto+(Z-f)$ and $\chi_{\mathbf g}\propto-f$ sharing one site
phase. The interference is therefore a definite, physically meaningful,
non-cancelling choice, not a dangling convention.

### Cheap filters on the braced expression

| filter | result |
| --- | --- |
| Units | `eUg_over_m` dimensionless; $\{\mathbf g;\boldsymbol\epsilon\}\sim$ $\AA^{-1}$; $\{\mathbf k;\mathbf g\}/(\mathbf v\cdot\mathbf g)\sim$ $\AA^{-1}$; overall $1/(\mathbf v\cdot\mathbf g)\sim$ Å. `A_CBS` dimensionless, same as `A_PXR`. Pass |
| $\gamma\to1$ | $A_{\rm CBS}\to-(U^{\rm code}_{\mathbf g}/mc^2)(\mathbf g\cdot\boldsymbol\epsilon)/(\mathbf v\cdot\mathbf g)+O(\beta)$ — finite and non-vanishing. CBS survives non-relativistically, which is the source paper's entire premise. Pass |
| $\gamma\to\infty$ | $\chi_{\mathbf g}$, $\mathbf v\cdot\mathbf g$ and $\omega$ all saturate as $\beta\to1$, so $A_{\rm CBS}/A_{\rm PXR}\propto1/\gamma\to0$: PXR wins at high energy, CBS is the low-energy mechanism. Pass, and consistent with the cited paper's scope |
| $\mathbf v\cdot\mathbf g\to0$ | $A_{\rm CBS}$ diverges as $(\mathbf v\cdot\mathbf g)^{-2}$. This is physical within first-order perturbation theory (an adiabatically slow modulation drives an unboundedly large excursion) and is exactly where the expansion fails: validity needs $\lvert\mathbf g\cdot\delta\mathbf r\rvert\ll1$, i.e. $\lvert U_{\mathbf g}\rvert g^2/(\gamma mc^2(\mathbf v\cdot\mathbf g)^2)\ll1$. The same limit sends $\omega\to0$, and `lines.py:1084-1089` drops every segment with $E_{\rm res}<10$ eV (and outside the padded grid), which bounds $\mathbf v\cdot\mathbf g=\omega(1-\mathbf v\cdot\hat{\mathbf n})$ away from zero. Pass, with the cutoff noted as load-bearing |
| $U_{\mathbf g}\to0$ | extinct reflection: `A_CBS` $\to0$, sum reduces to pure PXR. Pass |
| $\chi_{\mathbf g}\to0$ | `A_PXR` $\to0$, sum reduces to pure CBS. Pass |
| Sign/convention | relative PXR/CBS sign derived, not assumed; see table above. Pass |

### $1/\gamma$ versus $1/\gamma^2$: where it is distinguishable, and by how much

With $\gamma=1+T/m_ec^2$ and $m_ec^2=510.999$ keV, replacing $1/\gamma$ by
$1/\gamma^2$ scales the CBS amplitude by $1/\gamma$ and the CBS-only intensity
by $1/\gamma^2$:

| $T$ | $\gamma$ | CBS amplitude deficit | CBS-only intensity deficit |
| --- | --- | --- | --- |
| 30 keV | 1.058709 | 5.55 % | 10.78 % |
| 100 keV | 1.195695 | 16.37 % | 30.05 % |
| 300 keV | 1.587085 | 36.99 % | 60.30 % |

Because CBS is summed coherently with PXR, the observable change in
$\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$ is diluted by the CBS/PXR amplitude
ratio. Rebuilding the assembly at $\theta_{\rm obs}=120^\circ$, $45^\circ$ tilt
(CBS/PXR amplitude ratio $\approx0.43$ for Si, $\approx0.25$ for HOPG):

| case | 30 keV | 100 keV | 300 keV |
| --- | --- | --- | --- |
| silicon $(111)$, $\lvert A\rvert^2$ change | $-3.3$ % | $-9.9$ % | $-21.1$ % |
| HOPG $(002)$, $\lvert A\rvert^2$ change | $-2.3$ % | $-6.4$ % | $-12.1$ % |

Thresholds: the CBS-only intensity difference reaches 1 % at $T=2.6$ keV, 5 % at
13.3 keV and 10 % at 27.6 keV; the *total* silicon $(111)$ line intensity
difference reaches 1 % at $\approx9.5$ keV, 2 % at $\approx18.5$ keV, 5 % at
$\approx46.5$ keV and 10 % at $\approx101.5$ keV. So the choice is already
numerically distinguishable across PyRITE's whole operating range — at 30 keV it
is a 3 % effect on a real line and an 11 % effect on the CBS channel, both far
above statistical noise in a converged run. The ledger's "1/$\gamma$ matters
$\gtrsim100$ keV" understates this by roughly a decade in beam energy and should
be restated.

### Numeric certification against a directly integrated trajectory

The analytic result was checked against a brute-force calculation that shares no
algebra with it and imports no PyRITE code: integrate the exact relativistic
equation of motion $d\mathbf p/dt=-\nabla U$ (RK4) for an electron crossing the
real single-harmonic potential $U(\mathbf r)=2U\cos(\mathbf g\cdot\mathbf r)$,
then evaluate the radiation integral
$I=\int dt\,(-\boldsymbol\epsilon\cdot\boldsymbol\beta)\,
e^{i\omega(t-\hat{\mathbf n}\cdot\mathbf r)}$ numerically and extract the
coefficient that grows linearly in $T$. Geometry:
$\mathbf g=(1.5,-0.7,2.3)$ $\AA^{-1}$, $\hat{\mathbf n}\parallel(0.6,0.3,-0.5)$,
$\hat{\mathbf v}\parallel(0.1,0.2,1.0)$, $U=0.5$ eV, 400 driving periods at 400
steps per period, both polarizations. Initial conditions are placed on the
steady-state solution — otherwise the free $\delta\mathbf v=\text{const}$ mode
adds a spurious linear drift in $\mathbf r$ that contaminates the resonant
coefficient at the same order (this alone moved the ratio by 2–11 %).

Ratio of the numerically extracted coefficient to the analytic prediction:

| $T$ | line energy | with $1/\gamma$ | with $1/\gamma^2$ |
| --- | --- | --- | --- |
| 30 keV | 1275.2 eV | 1.000000, 1.000000 | 1.058709, 1.058709 |
| 100 keV | 1962.0 eV | 1.000000, 1.000001 | 1.195695, 1.195696 |
| 300 keV | 2569.8 eV | 1.000000, 0.999998 | 1.587086, 1.587082 |

The derived braced expression is reproduced to $<3\times10^{-6}$ relative at all
three energies and both polarizations, and the $1/\gamma^2$ variant is rejected
by exactly a factor $\gamma$. Residual imaginary parts are $\sim10^{-7}$ of the
real part, as required for a real $U_{\mathbf g}$. This certifies, numerically
and independently, the transverse projector, the single $1/\gamma$, the
$1/(\mathbf v\cdot\mathbf g)$ and $1/(\mathbf v\cdot\mathbf g)^2$ placements,
and the relative $+$ between the two braced terms.

### Implementation comparison, term for term

Production (`lines.py:1142-1146`, and the fused real kernel at
`lines.py:118-121`):

```text
A_PXR     = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
braced_ge = g_dot_e - vdg * v_dot_e
braced_kg = k_dot_g - k_dot_v * vdg
A_CBS     = -eUg_over_m / (gamma * vdg) * (braced_ge + v_dot_e * braced_kg / vdg)
pol_A.append(A_PXR + A_CBS)
```

with `v_all = beta_all[:,None] * seg_v` (velocity in units of $c$,
`lines.py:948`), `gamma_all = 1/sqrt(1-beta^2)` (`lines.py:994`),
`eUg_over_m` = `U_g`$/m_ec^2$ (`lines.py:1402`), and
`detuning = g2 + 2*k_dot_g` (`lines.py:1126`).

| derived | coded | verdict |
| --- | --- | --- |
| $U_{\mathbf g}/(\gamma m c^2)$ | `eUg_over_m / gamma` | match, single power |
| $(\mathbb 1-\boldsymbol\beta\boldsymbol\beta)$ contracted with $\boldsymbol\epsilon$ | `braced_ge = g_dot_e - vdg*v_dot_e` | match |
| same projector contracted with $\mathbf k$ | `braced_kg = k_dot_g - k_dot_v*vdg` | match |
| overall $1/(\mathbf v\cdot\mathbf g)$ | `/(gamma * vdg)` | match |
| extra $1/(\mathbf v\cdot\mathbf g)$ on the cross term only | `braced_kg / vdg` inside the parenthesis | match |
| $+$ between the two braced terms | `braced_ge + v_dot_e * braced_kg / vdg` | match |
| $(\mathbf v\cdot\boldsymbol\epsilon)$ (unperturbed velocity) on the cross term | `v_dot_e *` | match |
| $-S_{\rm CBS}$ with $U^{\rm code}=-U^{\rm phys}$ | leading `-` on `eUg_over_m` | match |
| $\lvert A_{\rm PXR}+A_{\rm CBS}\rvert^2$ with a shared overall $-1$ | `pol_A.append(A_PXR + A_CBS)` then `abs()**2` | match |
| $\lvert\mathbf k+\mathbf g\rvert^2-\varepsilon_0\omega^2/c^2$ | `g2 + 2*k_dot_g` | match |
| $\omega=\mathbf v\cdot(\mathbf k+\mathbf g)$ on resonance | `v_dot_kg = vdg + k_dot_v` | match |

No divergent term, factor, exponent, denominator, or sign was found in the
braced assembly.

### What remains blocked

Two things stay open, and neither is a defect found in the braced expression:

1. **Source transcription.** Because Feranchuk–Spence Eq. (14) and Zhai SI
   Eq. (6) are not obtainable from this repository, this pass certifies that the
   coded `A_CBS` is *the correct classical/first-Born CBS amplitude with the
   correct relative normalization against the coded `A_PXR`* — it does not
   certify that it is a faithful character-for-character transcription of those
   two equations. The residual transcription risk is bounded to a factor common
   to both amplitudes, which is absorbed by the separately ledgered
   $d^2N/dE\,d\Omega$ prefactor. A human with the papers should still confirm
   the citation text.
2. **Two $\varepsilon_0$-level details in `A_PXR`, out of scope here.** The
   derivation gives $\big[\omega(\mathbf v\cdot\boldsymbol\epsilon)
   -(\mathbf g\cdot\boldsymbol\epsilon)/\varepsilon_0\big]$, i.e. one of the two
   PXR numerator terms carries $1/\varepsilon_0$ that the code does not have
   (relative error $\sim\lvert\chi_0\rvert\sim10^{-4}$), and the derived
   $\omega^2$ appears in the code as `k_mag**2` $=\varepsilon_0\omega^2$. Both
   belong to `pxr-amplitude` / `xray-in-medium-resonance`, are already discussed
   at `docs/validation/radiation-physics/xray-in-medium-resonance.md`, and do
   not affect the CBS structure or the relative sign adjudicated here.

No regression test pins the braced structure, so this claim cannot advance past
`rederived`. A useful anchor would assert `A_CBS` against a closed-form
evaluation at one fixed geometry, plus a guard that the CBS prefactor scales as
$1/\gamma$ and not $1/\gamma^2$ across two beam energies.

### 2026-08-15 verdict

`rederived` for the whole row.

- **`U_g`** — unchanged from the first pass: independently rederived from
  Poisson's equation, with the two convention-level notes recorded there
  (magnitude branch, `f'` folded into the static potential for edge-prone
  atoms).
- **Braced `A_CBS` prefactor** — now independently **rederived** and numerically
  certified: single $1/\gamma$ (the $1/\gamma^2$ variant double-counts the
  longitudinal suppression already carried by the projector), transverse
  projector contracted twice, one overall $1/(\mathbf v\cdot\mathbf g)$ plus a
  second on the cross term only.
- **PXR+CBS cross term** — the relative sign is **correct**. Two sign flips
  (the `+e phi_g` branch of `U_g`, and the explicit leading `-` in `A_CBS`)
  cancel, and the coded pair equals $-(S_{\rm PXR}+S_{\rm CBS})$ with a shared
  overall $-1$ that the modulus squared removes.
- The first pass' "not independently certified / filtered, not rederived" caveat
  on the braced assembly is **closed** by this section. Not `signed-off` —
  that is a human transition.

### 2026-08-15 suggested ledger change

- `Status:` `filtered` → `rederived`.
- `Checks:` append "braced `A_CBS` tensor prefactor, single `1/γ`, `(v·g)` /
  `(v·g)²` denominator placement and the PXR+CBS relative sign independently
  rederived from the relativistic equation of motion plus the
  Liénard–Wiechert radiation integral, with `A_PXR` rederived in the same
  normalization; certified numerically against direct RK4 trajectory
  integration (rel. dev. `<3e-6` at 30/100/300 keV, both polarizations; the
  `1/γ²` variant is rejected by exactly a factor `γ`)".
- `Notes:` replace "1/γ matters ≳100 keV" with "1/γ vs 1/γ² changes the CBS
  amplitude by 5.5 / 16.4 / 37.0 % and a real Si(111) line intensity by 3.3 /
  9.9 / 21.1 % at 30 / 100 / 300 keV — distinguishable from ~10 keV up, not
  only ≳100 keV"; drop "exact `A_CBS` tensor prefactor + cross term unverified"
  and record instead that the source text for Feranchuk Eq. (14) / Zhai SI
  Eq. (6) is still not in-repo, so the transcription (as opposed to the
  physics) is uncertified.
- `Anchor:` still `—`; add a regression test pinning `A_CBS` at one fixed
  geometry and its `1/γ` scaling before this can reach `anchored`.

A human applies these.
