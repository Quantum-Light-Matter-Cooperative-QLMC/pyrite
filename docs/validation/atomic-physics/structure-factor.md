# Validation: `structure-factor`

**Claim.** Kinematical (geometric) structure factor of a crystal unit cell,
together with the isotropic Debye–Waller (thermal) attenuation factor:

```{math}
S(g) = \sum_j f_j(g)\exp(i g . r_j)\exp(-W_j)
```
```{math}
W = B \left(\frac{\sin\theta}{\lambda}\right)^2 = B\left(\frac{g}{4\pi}\right)^2
```

with the crystallographic B-factor `B = 8π² <u_x²>` (`<u_x²>` = mean-square
displacement along the scattering vector).

**Code.** `src/pyrite/materials/crystal.py::structure_factor`,
`src/pyrite/materials/crystal.py::debye_waller`
**Source (as handed).** Standard kinematical structure factor
$F(\mathbf g) = \sum_j f_j\exp(i\mathbf g\cdot\mathbf r_j)
\exp\left(-\dfrac{Bg^2}{16\pi^2}\right)$. The flagged trap is the
Debye–Waller exponent convention: $B$ vs $\langle u^2\rangle$, and the
factor $16\pi^2$ vs $4$.
**Anchor.** none in the ledger row.
**Verifier context.** Fresh, independent session. Derivation below was written
from the standard kinematical-diffraction result and the module's stated
reciprocal-space convention *before* reading the `debye_waller` body; the code
was only diffed afterward.

---

## 1. Independent derivation

### 1.1 Reciprocal-space convention used by this module

The module fixes the crystallographer's 2π convention throughout:
`_reciprocal_basis` builds `b_i` with the $2\pi$ prefactor so that
$\mathbf b_i\cdot\mathbf a_j=2\pi\delta_{ij}$, and
`reciprocal_g_vector`/`g_mag` document $|\mathbf g|=2\pi/d_{hkl}$. With
Bragg $\lambda=2d\sin\theta$,

$$
g=\frac{2\pi}{d}=\frac{4\pi\sin\theta}{\lambda}
$$

(so $g$ here is the physics momentum transfer $q$), hence

$$
s\equiv\frac{\sin\theta}{\lambda}=\frac{g}{4\pi}.
$$

This is the pivot for the entire "$16\pi^2$ vs $4$" trap: it is *only*
correct to write $W=Bg^2/16\pi^2$ when $g=2\pi/d$. Had the module used the
no-2π crystallographic convention $g=1/d=2\sin\theta/\lambda$, the correct
exponent would instead be $W=B(g/2)^2=Bg^2/4$. So the two candidate
denominators are not both "styles"; each belongs to exactly one $g$
convention. I must therefore check that the same `g` fed to `debye_waller` is
the $2\pi/d$ one.

### 1.2 Phase term

Write atom $j$ at fractional coordinates $\mathbf R_j=(x_j,y_j,z_j)$, i.e.
$\mathbf r_j=x_j\mathbf a_1+y_j\mathbf a_2+z_j\mathbf a_3$, and
$\mathbf g=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3$. Then

$$
\mathbf g\cdot\mathbf r_j
=2\pi(hx_j+ky_j+lz_j)
=2\pi(hkl\cdot\mathbf R_j).
$$

So $\exp(i\mathbf g\cdot\mathbf r_j)=\exp(2\pi i(hkl\cdot\mathbf R_j))$. The
global sign ($+i$ vs $-i$) is a pure convention choice; it conjugates `S`
and leaves the only physical observable $|S|^2$ invariant, provided the
same sign is used for every atom.

### 1.3 Debye–Waller factor (the trap)

For a Gaussian-distributed thermal displacement $\mathbf u$ (zero mean),
the coherent amplitude from a vibrating atom is

$$
f_{\rm thermal}=f_0\langle\exp(i\mathbf g\cdot\mathbf u)\rangle
=f_0\exp\!\left(-\tfrac12\langle(\mathbf g\cdot\mathbf u)^2\rangle\right).
$$

Hence the *amplitude* attenuation is $\exp(-W)$ with

$$
W=\tfrac12\langle(\mathbf g\cdot\mathbf u)^2\rangle
=\tfrac12 g^2\langle u_g^2\rangle,
$$

where $\langle u_g^2\rangle$ is the mean-square displacement resolved
along $\mathbf g$. The crystallographic isotropic displacement parameter
is *defined* as

$$
B=8\pi^2\langle u_g^2\rangle
\quad\Rightarrow\quad
\langle u_g^2\rangle=\frac{B}{8\pi^2}.
$$

Substituting, and using $g=4\pi s$:

$$
W=\tfrac12 g^2\cdot\frac{B}{8\pi^2}
=\tfrac12(4\pi s)^2\frac{B}{8\pi^2}
=\tfrac12\cdot16\pi^2 s^2\cdot\frac{B}{8\pi^2}
=Bs^2.
$$

Therefore the amplitude Debye–Waller factor is

$$
\boxed{\exp(-W)=\exp(-Bs^2)=\exp\!\left(-B\left(\frac{g}{4\pi}\right)^2\right)
=\exp\!\left(-\frac{Bg^2}{16\pi^2}\right)}.
\qquad(\star)
$$

Cross-check of the two traps:
- **$B$ vs $\langle u^2\rangle$:** $B=8\pi^2\langle u_g^2\rangle$ (one
  Cartesian/along-$\mathbf g$ component, *not* the 3-D total
  $\langle u^2\rangle=3\langle u_x^2\rangle$). Using the 3-D total here
  would introduce a spurious factor of 3.
- **$16\pi^2$ vs $4$:** with $g=2\pi/d$ (this module) the correct
  denominator is $16\pi^2$. The $4$ would only be correct for
  $g=1/d$.

The **intensity** carries $\exp(-2W)=\exp(-2Bs^2)$; the amplitude carries
one factor of $\exp(-W)$. `structure_factor` returns an *amplitude* `S`,
so it must apply $\exp(-W)$ exactly once per atom.

### 1.4 Limiting cases I expect

- $B\to0$ or $g\to0$: $\exp(-W)\to1$ (no thermal suppression).
- $g\to\infty$ / large $B$: $\exp(-W)\to0^+$ (high-angle reflections
  damped — attenuation, so the exponent sign **must be negative**).
- $g\to0$: $S(0)=\sum_j f_j(0)\approx\sum_j Z_j$ (forward scattering =
  total electrons).
- Diamond-structure selection rule: FCC lattice with 2-atom basis at
  $(0,0,0)$ and $(\tfrac14,\tfrac14,\tfrac14)$ gives
  $S\propto(1+i^{h+k+l})$ on top of the all-even/all-odd FCC rule. So
  `(111)` and `(400)` are allowed, `(222)` and `(200)` are extinct. This is a
  strong, sign-of-phase-sensitive test.

---

## 2. Diff against the implementation

### 2.1 `debye_waller`

```python
def debye_waller(g_invang, B_ang2):
    """... W = B (sin(theta)/lambda)^2 = B (g/4pi)^2, ... (B = 8 pi^2 <u_x^2>).
    Intensities carry exp(-2W) = the square of this."""
    s = g_invang / (4.0 * np.pi)
    return np.exp(-B_ang2 * s**2)
```

$s=g/(4\pi)$, return $\exp(-Bs^2)=\exp(-Bg^2/16\pi^2)$. This is
**exactly** $(\star)$. The docstring states $B=8\pi^2\langle
u_x^2\rangle$ and that intensities take $\exp(-2W)$ — both match my
derivation. Sign is negative (attenuation). ✓

### 2.2 `structure_factor`

```python
_, g = reciprocal_g_vector(hkl, info["lattice"])  # g = 2π/d  (2π convention)
dwf = debye_waller(g, B_ang2)  # single scalar B for all atoms
S = 0.0 + 0.0j
for (_el, R), F in zip(info["basis"], _basis_F(...)):
    phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))  # exp(2π i hkl·R)
    S += F * phase * dwf
return S, g
```

- `g` is the `2π/d` vector (confirmed via `reciprocal_g_vector`/`_reciprocal_basis`),
  so it is the correct partner for the `16π²` denominator. ✓
- `phase = exp(2π i (hkl·R))` = my `exp(i g·r_j)` with the `+i` convention. ✓
- `dwf` applied **once** per atom → amplitude convention `exp(-W)`. ✓
- `F_j` from `_basis_F`: `f0(g)` (Waasmaier–Kirfel) for non-resonant elements,
  or complex `f0+f'+i f''` for edge-prone elements / `use_henke`. Consistent with
  `f_j(g)` in the cited formula. ✓

**Caveat (not a discrepancy).** The code applies a *single* scalar `B_ang2` to
all atoms in the basis, whereas the general formula allows a per-atom `B_j`
(`exp(-W_j)`). For crystals whose species have different B-factors (e.g. Al vs O
in sapphire), this is an isotropic single-B approximation. It matches the
handed-in formula (which also carries a single `B`) and the ledger's per-crystal
scalar `B_ang2` usage, so it is a documented modelling choice, not an error in the
cited equation.

---

## 3. Cheap-filter results

| filter | result |
|--------|--------|
| **Units** | $B\ [\text{Å}^2]\cdot g^2\ [\text{Å}^{-2}]$ → dimensionless exponent ✓; `S` dimensionless (electron units): $f_j$ (electrons) × phase × DW ✓ |
| **Limit $B\to0$ / $g\to0$** | `debye_waller` $\to1$ ✓ (numeric below) |
| **Limit $g\to\infty$ / large $B$** | $\exp(-W)\to0^+$, high-angle damping ✓; sign of exponent negative ✓ |
| **Forward scattering** | $S(000)\approx\sum Z_j$ ✓ (numeric below) |
| **Selection rules (phase)** | diamond `(222)`,`(200)` extinct; `(111)`,`(400)` allowed ✓ (numeric below) |
| **$16\pi^2$ vs $4$** | code uses $16\pi^2$, consistent with its own $g=2\pi/d$; $4$ would be wrong here ✓ |
| **$B$ vs $\langle u^2\rangle$** | $B=8\pi^2\langle u_g^2\rangle$ (1-component), standard crystallographic ✓ |

---

## 4. Numeric probes (real output)

### 4.1 Debye–Waller vs the two candidate denominators

`CUDA_VISIBLE_DEVICES="" uv run --no-sync python` with `B=0.5`:

```
g= 0.0000  code=1.00000000  16pi2=1.00000000  factor4=1.00000000
g= 1.0000  code=0.99683872  16pi2=0.99683872  factor4=0.88249690
g= 3.0000  code=0.97190562  16pi2=0.97190562  factor4=0.32465247
g= 6.2832  code=0.88249709  16pi2=0.88249709  factor4=0.00719194

W_code = 0.028496583   B*s^2 = 0.028496583   u2 = B/(8pi^2) = 0.0063326
```

Code reproduces $\exp(-Bg^2/16\pi^2)$ to 8 decimals and is *nowhere near*
the `factor4` trap. $W_{\rm code}=Bs^2$ confirms $(\star)$ and the
$B=8\pi^2\langle u^2\rangle$ mapping.

### 4.2 Structure-factor selection rules (silicon, diamond structure)

```
hkl=(1, 1, 1)  |S|=   61.0415  S=44.477-41.808j  g=2.0039
hkl=(2, 2, 2)  |S|=    0.0000  S=0.000+0.000j    g=4.0077
hkl=(4, 0, 0)  |S|=   62.1595  S=62.102+2.669j   g=4.6277
hkl=(2, 0, 0)  |S|=    0.0000  S=0.000+0.000j    g=2.3139

S(000)= 113.98 + 2.67j   (expect 8·Z_Si = 8·14 = 112; f0(0)=Z)
```

`(222)` and `(200)` extinct to machine zero while `(111)`/`(400)` are strong —
this only happens if the $\exp(2\pi i\,hkl\cdot\mathbf R)$ phases (and
their relative signs) are implemented correctly, so the phase convention is
verified. $S(000)\approx114$ matches $\sum Z_j=112$ (the small excess +
imaginary part are Si's anomalous $f'+if''$ at 8 keV, since Si is
edge-prone → complex form factor). ✓

---

## 5. Finding: missing in-code `Validation:` marker

Per the [validation methodology](../methodology.md), every ledgered physics function must carry a
one-line `Validation: <id>` marker in its docstring. `materials/crystal.py` contains
that marker only on `optical_constants` (`Validation: grazing-optical-constants`).
`structure_factor` and `debye_waller` — the code for this claim — carry **no**
`Validation: structure-factor` marker (nor do the sibling ledgered functions
`chi_g`, `U_g`, `absorption_length_ang`). The physics is correct; the back-reference
is absent. This is a documentation/traceability gap for the human to close, not a
numerical defect.

---

## 6. Adjudication

All cheap filters pass. The independent derivation reproduces the code
term-for-term: the phase $\exp(2\pi i\,hkl\cdot\mathbf R)$, the single
amplitude Debye–Waller factor $\exp(-Bg^2/16\pi^2)$ with
$B=8\pi^2\langle u_g^2\rangle$, and the $g=2\pi/d$ convention that
makes $16\pi^2$ (not $4$) the correct denominator. Both flagged traps are handled
correctly. Numerics confirm the DW value to 8 digits and the phase convention via
diamond selection rules. The only caveats are (a) a single scalar `B` shared across
all basis atoms (a documented isotropic approximation consistent with the cited
formula) and (b) the missing in-code `Validation:` marker.

**Recommended status: `rederived`** (independent derivation matches the code).
Final `signed-off` remains a human decision.

> **Not yet promoted.** This is a *single-pass* validation. The house rule for
> promotion to `rederived` requires that two independent adversarial refuters
> (algebraic + numeric lens) also fail to break the claim; that stage did not run
> (the overnight batch was cut short by a usage limit). The ledger row for
> `structure-factor` is therefore **unchanged**. Treat the verdict above as one
> verifier's opinion, not as a validated claim. To reach `anchored`, add a regression
test pinning the diamond `(222)`-extinct / `(111)`-allowed selection rules and the
`exp(-B g²/16π²)` DW value at one `(g,B)` point.

---

## 7. Second independent verifier pass (2026-07-22)

A second fresh, independent verifier — with no access to the first verifier's
numerics and re-deriving from standard kinematical crystallography — reproduced
every result above. This is the algebraic+numeric second lens the section-6
caveat said had not yet run.

**Independent re-derivation (matches).** Starting only from the cited standard
form `F(g) = Σ_j f_j(g,E) exp(±i g·r_j) exp(−M_j)` and the module's documented
`|g| = 2π/d` convention (verified via `_reciprocal_basis`, which carries the `2π`
prefactor so `b_i·a_j = 2π δ_ij`):

- Phase: `g·r_j = 2π (hkl·R_j)` for fractional coords → `exp(2π i hkl·R_j)`,
  matching the code's `np.exp(1j*2π*np.dot(hkl,R))`. The global `+i` is a
  convention that conjugates `S` and leaves `|S|²` invariant.
- Bragg `λ = 2d sinθ` with `g = 2π/d` gives `sinθ/λ = g/(4π)`, so
  `M = B (sinθ/λ)² = B (g/4π)² = B g²/(16π²)`. This is the *amplitude* exponent;
  the `4` denominator would require the no-2π `g=1/d` convention, which this
  module does not use. Code's `s = g/(4π); exp(−B s²)` is exactly this.
- `B = 8π²⟨u_g²⟩` (single along-`g` component, not 3-D total): matches docstring.

**Independent numerics (this pass, own probes).**
- `debye_waller` at `B=0.5`, `g∈{0,1,3,2π}` reproduces `exp(−B g²/16π²)` to 10
  digits and diverges sharply from the `factor4` candidate (e.g. `g=2π`:
  code/16π² = 0.88250 vs factor4 = 0.00719).
- Silicon (diamond) at 8 keV: `(222)` and `(200)` extinct to machine zero;
  `(111)`, `(220)`, `(311)`, `(400)` allowed — the phase-sign-sensitive diamond
  selection rule. `S(000) = 113.98 + 2.67j ≈ 8·Z_Si = 112` (excess/imag from Si
  anomalous `f'+if''` at 8 keV).
- LiF (rock-salt) at 8 keV: `F` real (centrosymmetric, imag ≈ 0);
  `|S(111)| = 18.97` (difference `∝ |f_F − f_Li|`) `< |S(200)| = 30.04`
  (sum `∝ |f_F + f_Li|`) — the correct NaCl-type sum/difference rule.

**Concurrence.** Both traps (`B` vs `⟨u²⟩`; `16π²` vs `4`) handled correctly;
phase convention verified via selection rules. No divergent term found. Two
documented non-defects stand: single scalar `B_ang2` shared across basis atoms
(isotropic single-B approximation, consistent with the handed-in single-`B`
formula) and the absent in-code `Validation: structure-factor` marker on
`structure_factor`/`debye_waller` (traceability gap for a human to close).

**Verdict of this pass:** `rederived` (independent derivation matches the code).
Promotion of the ledger row and any `signed-off` remain human decisions.

---

## 8. Third independent verifier pass (2026-07-25)

A third fresh-context verifier re-derived from standard kinematical
crystallography and the module's documented `|g| = 2π/d` convention, applied
cheap filters, then read the live `crystal.py` bodies and ran an own numeric
probe (independent of the section-4 and section-7 numbers).

**Live code re-read.** `_reciprocal_basis` builds `B = 2π·[cross]/V`
(`b_i·a_j = 2π δ_ij`, verified 2π convention); `reciprocal_g_vector`/`g_mag`
document `|g| = 2π/d`. `debye_waller`: `s = g/(4π); exp(−B s²)`. `structure_factor`:
`phase = exp(1j·2π·hkl·R)`, accumulates `F·phase·dwf` with one scalar `dwf` per
atom. Bodies are unchanged from the snippets diffed in sections 2 and 7.

**Own numerics (this pass).**
- `debye_waller(B=0.5)` at `g∈{0,1,3,2π}` reproduces `exp(−B g²/16π²)` to 10
  digits; the `factor4` candidate diverges (`g=2π`: 0.88250 vs 0.00719).
- Silicon (diamond) at 8 keV: `(222)`, `(200)` extinct to machine zero;
  `(111)`, `(220)`, `(311)`, `(400)` allowed. An independent hand-built 8-atom
  diamond basis reproduces the identical geometric-factor extinction pattern
  (`|geom|` = 5.66, 0, 8.0, 0 for `(111)`,`(222)`,`(400)`,`(200)`), confirming the
  phase sign/convention without any implementation helper.
- `S(000) = 113.98 + 2.67j ≈ 8·Z_Si = 112` (excess/imag = Si anomalous `f'+if''`).

**Concurrence.** Both flagged traps handled correctly (`B = 8π²⟨u_g²⟩`, not the
3-D total; `16π²`, not `4`, given `g=2π/d`); single amplitude DW factor per atom;
`+i` phase convention verified via selection rules. No divergent term. The two
standing non-defects are unchanged: single scalar `B_ang2` shared across basis
atoms (documented isotropic single-B approximation, consistent with the handed-in
single-`B` formula) and the absent in-code `Validation: structure-factor` marker
on `structure_factor`/`debye_waller`.

**Verdict of this pass:** `rederived` (independent derivation matches the code).
This is now a third concurring independent pass. Promotion of the ledger row and
`signed-off` remain human decisions.
