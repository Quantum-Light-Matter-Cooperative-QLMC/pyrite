# Validation: `structure-factor`

**Claim.** Kinematical (geometric) structure factor of a crystal unit cell,
together with the isotropic Debye–Waller (thermal) attenuation factor:

```
S(g) = sum_j f_j(g) * exp(i g . r_j) * exp(-W_j),   W = B (sin θ / λ)^2 = B (g/4π)^2
```

with the crystallographic B-factor `B = 8π² <u_x²>` (`<u_x²>` = mean-square
displacement along the scattering vector).

**Code.** `src/cxr_mc/crystallography.py::structure_factor`,
`src/cxr_mc/crystallography.py::debye_waller`
**Source (as handed).** Standard kinematical structure factor
`F(g) = Σ_j f_j exp(i g·r_j) exp(−B g²/16π²)`. The flagged trap is the
Debye–Waller exponent convention: `B` vs `<u²>`, and the factor `16π²` vs `4`.
**Anchor.** none in the ledger row.
**Verifier context.** Fresh, independent session. Derivation below was written
from the standard kinematical-diffraction result and the module's stated
reciprocal-space convention *before* reading the `debye_waller` body; the code
was only diffed afterward.

---

## 1. Independent derivation

### 1.1 Reciprocal-space convention used by this module

The module fixes the crystallographer's 2π convention throughout:
`_reciprocal_basis` builds `b_i` with the `2π` prefactor so that
`b_i · a_j = 2π δ_ij`, and `reciprocal_g_vector`/`g_mag` document
`|g| = 2π/d_hkl`. With Bragg `λ = 2 d sinθ`,

```
g = 2π/d = 4π sinθ / λ            (so g here is the physics momentum transfer q)
⇒ s ≡ sinθ/λ = g / (4π).
```

This is the pivot for the entire "16π² vs 4" trap: it is *only* correct to write
`W = B g²/16π²` when `g = 2π/d`. Had the module used the no-2π crystallographic
convention `g = 1/d = 2 sinθ/λ`, the correct exponent would instead be
`W = B (g/2)² = B g²/4`. So the two candidate denominators are not both "styles";
each belongs to exactly one `g` convention. I must therefore check that the same
`g` fed to `debye_waller` is the `2π/d` one.

### 1.2 Phase term

Write atom `j` at fractional coordinates `R_j = (x_j, y_j, z_j)`, i.e.
`r_j = x_j a_1 + y_j a_2 + z_j a_3`, and `g = h b_1 + k b_2 + l b_3`. Then

```
g · r_j = 2π (h x_j + k y_j + l z_j) = 2π (hkl · R_j).
```

So `exp(i g·r_j) = exp(2π i (hkl·R_j))`. The global sign (+i vs −i) is a pure
convention choice; it conjugates `S` and leaves the only physical observable
`|S|²` invariant, provided the same sign is used for every atom.

### 1.3 Debye–Waller factor (the trap)

For a Gaussian-distributed thermal displacement `u` (zero mean), the
coherent amplitude from a vibrating atom is

```
f_thermal = f0 <exp(i g·u)> = f0 exp(-½ <(g·u)²>).
```

Hence the *amplitude* attenuation is `exp(-W)` with

```
W = ½ <(g·u)²> = ½ g² <u_g²>,
```

where `<u_g²>` is the mean-square displacement resolved along `g`. The
crystallographic isotropic displacement parameter is *defined* as

```
B = 8π² <u_g²>   ⇒   <u_g²> = B / (8π²).
```

Substituting, and using `g = 4π s`:

```
W = ½ g² · B/(8π²) = ½ (4π s)² B/(8π²) = ½ · 16π² s² · B/(8π²) = B s².
```

Therefore the amplitude Debye–Waller factor is

```
exp(-W) = exp(-B s²) = exp(-B (g/4π)²) = exp(-B g² / 16π²).            (★)
```

Cross-check of the two traps:
- **B vs <u²>:** `B = 8π² <u_g²>` (one Cartesian/`along-g` component, *not* the
  3-D total `<u²> = 3<u_x²>`). Using the 3-D total here would introduce a
  spurious factor of 3.
- **16π² vs 4:** with `g = 2π/d` (this module) the correct denominator is
  `16π²`. The `4` would only be correct for `g = 1/d`.

The **intensity** carries `exp(-2W) = exp(-2B s²)`; the amplitude carries one
factor of `exp(-W)`. `structure_factor` returns an *amplitude* `S`, so it must
apply `exp(-W)` exactly once per atom.

### 1.4 Limiting cases I expect

- `B → 0` or `g → 0`: `exp(-W) → 1` (no thermal suppression).
- `g → ∞` / large `B`: `exp(-W) → 0⁺` (high-angle reflections damped —
  attenuation, so the exponent sign **must be negative**).
- `g → 0`: `S(0) = Σ_j f_j(0) ≈ Σ_j Z_j` (forward scattering = total electrons).
- Diamond-structure selection rule: FCC lattice with 2-atom basis at `(0,0,0)`
  and `(¼,¼,¼)` gives `S ∝ (1 + i^{h+k+l})` on top of the all-even/all-odd FCC
  rule. So `(111)` and `(400)` are allowed, `(222)` and `(200)` are extinct.
  This is a strong, sign-of-phase-sensitive test.

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

`s = g/(4π)`, return `exp(-B s²) = exp(-B g²/16π²)`. This is **exactly** (★).
The docstring states `B = 8π² <u_x²>` and that intensities take `exp(-2W)` — both
match my derivation. Sign is negative (attenuation). ✓

### 2.2 `structure_factor`

```python
_, g = reciprocal_g_vector(hkl, info["lattice"])   # g = 2π/d  (2π convention)
dwf = debye_waller(g, B_ang2)                       # single scalar B for all atoms
S = 0.0 + 0.0j
for (_el, R), F in zip(info["basis"], _basis_F(...)):
    phase = np.exp(1j * 2.0 * np.pi * np.dot(hkl, R))   # exp(2π i hkl·R)
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
| **Units** | `B [Å²]·g² [Å⁻²]` → dimensionless exponent ✓; `S` dimensionless (electron units): `f_j` (electrons) × phase × DW ✓ |
| **Limit B→0 / g→0** | `debye_waller → 1` ✓ (numeric below) |
| **Limit g→∞ / large B** | `exp(-W) → 0⁺`, high-angle damping ✓; sign of exponent negative ✓ |
| **Forward scattering** | `S(000) ≈ Σ Z_j` ✓ (numeric below) |
| **Selection rules (phase)** | diamond `(222)`,`(200)` extinct; `(111)`,`(400)` allowed ✓ (numeric below) |
| **16π² vs 4** | code uses `16π²`, consistent with its own `g=2π/d`; `4` would be wrong here ✓ |
| **B vs <u²>** | `B = 8π² <u_g²>` (1-component), standard crystallographic ✓ |

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

Code reproduces `exp(-B g²/16π²)` to 8 decimals and is *nowhere near* the `factor4`
trap. `W_code = B s²` confirms (★) and the `B = 8π²<u²>` mapping.

### 4.2 Structure-factor selection rules (silicon, diamond structure)

```
hkl=(1, 1, 1)  |S|=   61.0415  S=44.477-41.808j  g=2.0039
hkl=(2, 2, 2)  |S|=    0.0000  S=0.000+0.000j    g=4.0077
hkl=(4, 0, 0)  |S|=   62.1595  S=62.102+2.669j   g=4.6277
hkl=(2, 0, 0)  |S|=    0.0000  S=0.000+0.000j    g=2.3139

S(000)= 113.98 + 2.67j   (expect 8·Z_Si = 8·14 = 112; f0(0)=Z)
```

`(222)` and `(200)` extinct to machine zero while `(111)`/`(400)` are strong —
this only happens if the `exp(2π i hkl·R)` phases (and their relative signs) are
implemented correctly, so the phase convention is verified. `S(000) ≈ 114` matches
`Σ Z_j = 112` (the small excess + imaginary part are Si's anomalous `f'+i f''` at
8 keV, since Si is edge-prone → complex form factor). ✓

---

## 5. Finding: missing in-code `Validation:` marker

Per `docs/validation/README.md`, every ledgered physics function must carry a
one-line `Validation: <id>` marker in its docstring. `crystallography.py` contains
that marker only on `optical_constants` (`Validation: grazing-optical-constants`).
`structure_factor` and `debye_waller` — the code for this claim — carry **no**
`Validation: structure-factor` marker (nor do the sibling ledgered functions
`chi_g`, `U_g`, `absorption_length_ang`). The physics is correct; the back-reference
is absent. This is a documentation/traceability gap for the human to close, not a
numerical defect.

---

## 6. Adjudication

All cheap filters pass. The independent derivation reproduces the code
term-for-term: the phase `exp(2π i hkl·R)`, the single amplitude Debye–Waller
factor `exp(-B g²/16π²)` with `B = 8π² <u_g²>`, and the `g = 2π/d` convention that
makes `16π²` (not `4`) the correct denominator. Both flagged traps are handled
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
