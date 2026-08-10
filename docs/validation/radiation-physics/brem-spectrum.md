# Validation: `brem-spectrum`

Independent verification of the bremsstrahlung background spectrum.

- **Claim id:** `brem-spectrum`
- **Code:** `src/cxr_mc/montecarlo/spectrum/brem.py::mc_brem_spectrum`
  (cross-section in `::_brem_dsigma_dk`)
- **Ledger source:** "bremsstrahlung background, Born + Elwert" — i.e. the
  Born-approximation Bethe–Heitler bremsstrahlung cross-section, energy
  differential, multiplied by the Elwert Coulomb-correction factor.
- **Ledger note:** "benign 0-eV divide-by-zero clamped".
- **Verifier independence:** the derivation in §1–§3 was written from the cited
  source (Born + Elwert) and the function signature/docstring ONLY, before
  reading the cross-section implementation body (`_brem_dsigma_dk`). The
  absorption/`mu` setup (lines 688–710) was seen incidentally and is out of
  scope for this claim (it belongs to `self-absorption` /
  `finite-transverse-crystal`).

---

## 1. Intended quantity, signature, units

From the docstring: `mc_brem_spectrum` returns the incoherent bremsstrahlung
background

```
d2N / (dE dOmega)   [photons / eV / sr / electron]
```

evaluated on `E_grid_eV` (photon energies, eV), summed over the same Monte
Carlo electron slowing-down `segments` used by `mc_spectrum`. Per the
docstring, each segment radiates

```
dN_seg/dk = n_atoms * (dsigma/dk) * L_seg          [photons / eV]
```

at the segment's (start) kinetic energy `T`, taken isotropic (`1/4pi`
per steradian), then multiplied by a Beer–Lambert escape factor from the
segment midpoint along the observation direction `n_hat`. For a compound the
emission is additive over elements, each weighted by its own `Z^2`.

Signature (physics-relevant args):
`mc_brem_spectrum(segments, E_grid_eV, element=None, n_atoms_per_ang3=None,
theta_obs_rad=deg2rad(119), n_hat=None, composition=None, layers=None, ...)`.

### Dimensional check of the assembly

- `n_atoms_per_ang3` = `n`  [1/Å³]
- `dsigma/dk`               [Å² / eV]   (cross-section area per unit photon energy)
- `L_seg`                   [Å]
- product `n * dsigma/dk * L_seg` = [1/Å³][Å²/eV][Å] = **[1/eV]** ✓ (photons/eV)
- `× 1/(4π)` → photons/eV/sr ✓
- `× exp(−∫μ dl)` (dimensionless) → photons/eV/sr ✓

So the outer assembly is dimensionally the intended `d2N/dE dOmega` per
electron. The remaining content is the cross-section `dsigma/dk`.

---

## 2. Independent derivation of the Born + Elwert cross-section

### 2a. Born (Bethe–Heitler) non-relativistic energy-differential cross-section

PyRITE operates at weakly relativistic beam energies (tens of keV), and the
docstring states emission is taken isotropic at "weakly relativistic
energies". The canonical Born-approximation bremsstrahlung cross-section,
differential in photon energy `k` and integrated over photon and electron
emission angles, in its non-relativistic (unscreened) limit is the
Bethe–Heitler / Koch–Motz 3BN low-energy form (Koch & Motz, *Rev. Mod. Phys.*
31, 920 (1959); Heitler, *Quantum Theory of Radiation*, §25):

```
dsigma/dk = (16/3) * (Z^2 * r_e^2 * alpha) / (k * beta_i^2)
                   * ln[ (p_i + p_f) / (p_i − p_f) ]
```

with, in non-relativistic kinematics,

```
T   = initial electron kinetic energy
k   = emitted photon energy,   0 < k <= T
T_f = T − k                    (final kinetic energy)
p_i / p_f = sqrt(T / T_f) = 1 / sqrt(1 − k/T)
=> (p_i + p_f)/(p_i − p_f) = (1 + sqrt(1 − k/T)) / (1 − sqrt(1 − k/T))
```

Constants: `alpha = e^2/(hbar c) ≈ 1/137` (fine structure),
`r_e = e^2/(m c^2)` (classical electron radius). `beta_i = v_i/c` is the
initial electron velocity.

Dimensions: `Z^2 r_e^2 alpha / k` = [dimensionless][Å²][dimensionless]/[eV]
= **[Å²/eV]** ✓. The `1/beta_i^2` factor and the log are dimensionless.

**Positivity:** for `0 < k < T`, `0 < sqrt(1−k/T) < 1`, so the log argument
`(1+s)/(1−s) > 1` and the log is strictly positive; `1/(k beta_i^2) > 0`.
Hence `dsigma/dk > 0` on the physical interval. ✓

**Infrared / low-energy limit (`k → 0`):** `sqrt(1−k/T) ≈ 1 − k/(2T)`, so the
log argument ≈ `4T/k` and

```
dsigma/dk  ~  (16/3)(Z^2 r_e^2 alpha)/(k beta_i^2) * ln(4T/k) .
```

Leading behavior is the **`1/k` Bethe–Heitler infrared rise** (with a slow
`ln(4T/k)` enhancement). This is the "low-energy 1/E behavior" limiting case.
It is a genuine `1/k` divergence as `k → 0`: any `E_grid` point at exactly
`E = 0` yields `1/0`, which is the **divide-by-zero the ledger says is
"benign … clamped"**.

**Tip / short-wavelength limit (`k → T`):** `p_f → 0`,
`sqrt(1−k/T) → 0`, so `ln[(1+s)/(1−s)] → ln(1) = 0`. The **bare Born
cross-section vanishes at the tip**, `dsigma/dk ∝ 2 sqrt(1−k/T) → 0`.

**Kinematic hard cutoff:** for `k >= T` there is no phase space
(`T_f < 0`); `dsigma/dk ≡ 0`. Any implementation must zero photon-energy bins
above the emitting segment's kinetic energy. This is the "tip cutoff".

### 2b. Elwert Coulomb-correction factor

```
f_E = [ beta_i * (1 − exp(−2*pi*alpha*Z / beta_i)) ]
      / [ beta_f * (1 − exp(−2*pi*alpha*Z / beta_f)) ]
    = (beta_i/beta_f) * (1 − exp(−2παZ/beta_i)) / (1 − exp(−2παZ/beta_f))
```

with `beta_i`, `beta_f` the initial and final electron velocities/c.
Positivity: each `(1 − e^{−x})` with `x > 0` is in `(0,1)`, `beta > 0`, so
`f_E > 0`. ✓

**High-energy / Born limit** (`2*pi*alpha*Z/beta ≪ 1`): expand
`1 − e^{−x} ≈ x`, numerator `≈ beta_i·(2παZ/beta_i) = 2παZ`, denominator
`≈ 2παZ`, so `f_E → 1`. ✓ (Elwert reduces to unity where Born is valid.)

**Low-energy limit** (`2παZ/beta ≫ 1`): `1 − e^{−x} → 1`, so `f_E → beta_i/beta_f`.

### 2c. Combined

```
dsigma/dk = (16/3)(Z^2 r_e^2 alpha)/(k beta_i^2)
              * ln[(1+sqrt(1−k/T))/(1−sqrt(1−k/T))] * f_E ,   0 < k < T ;   = 0, k >= T.
```

### 2d. Tip caveat (verified against code in §4)

The task lists "spectrum → 0 at the tip" as a limiting case. That is exact for
**bare Born**. It is **NOT** true once Elwert is applied, because near the tip
`ln[(1+s)/(1−s)] ≈ 2 beta_f/beta_i → 0` while `f_E ≈ beta_i(1−e^{−2παZ/beta_i})/beta_f → ∞`
like `1/beta_f`; the product tends to `2(1 − e^{−2παZ/beta_i})`, **finite and
non-zero**. This is the intended physics of the Elwert/Sommerfeld correction:
it removes the spurious Born vanishing at the tip and restores the finite
(Kramers-like) short-wavelength intensity. Correct behavior is therefore:
finite as `k → T⁻`, then an exact hard drop to `0` for `k ≥ T`.

---

## 3. Cheap filters

| filter | expectation | status |
|--------|-------------|--------|
| units of `dsigma/dk` | area/energy; assembled → photons/eV/sr/e⁻ | PASS |
| positivity | `dσ/dk > 0` on `0<k<T`; `f_E > 0` | PASS |
| Elwert high-E limit | `f_E → 1` when `2παZ/β ≪ 1` | PASS |
| IR limit | `dσ/dk ~ (1/k) ln(4T/k)`; `E=0` bin clamped | PASS |
| tip cutoff | `dσ/dk = 0` for `k ≥ T` | PASS |
| tip approach | Born×Elwert → finite (not 0) | PASS (see §4) |
| `Z^2` weighting | additive over compound elements, each ∝ Z² | PASS |

---

## 4. Comparison against the implementation

Read: `_brem_dsigma_dk(Z, T_keV, k_eV)` (lines 581–628) and the assembly loop
in `mc_brem_spectrum` (lines 714–757).

Implemented cross-section:

```
dsigma/dk = (16/3) alpha r_e^2 Z^2 (1/k) (1/p_i^2) ln[(p_i+p_f)/(p_i-p_f)] * f_Elwert
f_Elwert  = (beta_i/beta_f) (1-exp(-2 pi Z alpha/beta_i)) / (1-exp(-2 pi Z alpha/beta_f))
p = sqrt(T (T + 2 mc^2)) / mc^2         (relativistic momentum, units m_e c)
beta = p / (1 + T/mc^2) = p / gamma
```

### Term-by-term diff vs the independent derivation (§2)

| term | derivation (§2) | code | match |
|------|-----------------|------|-------|
| prefactor | `(16/3) α r_e² Z²` | `16/3 * ALPHA_FS * R_E_CM2 * Z**2` | ✓ |
| photon-energy factor | `1/k` | `1/max(k·1e3, 1e-30)` (k in eV) | ✓ |
| velocity factor | `1/β_i²` (NR) | `1/p_i²` (rel. momentum) | ✓ NR-equivalent* |
| Born log | `ln[(p_i+p_f)/(p_i−p_f)]` | `log((p_i+p_f)/max(p_i−p_f,1e-30))` | ✓ |
| Elwert | `β_i(1−e^{−2παZ/β_i}) / [β_f(1−e^{−2παZ/β_f})]` | `(β_i/β_f)(1−e^{…/β_i})/(1−e^{…/β_f})` | ✓ identical |
| tip cutoff | `= 0` for `k ≥ T` | `where((T_f>1e-6)&(k>0), dsig, 0)` | ✓ |
| 0-eV clamp | drop `E=0` bin | `k>0` mask + `max(k·1e3,1e-30)` + `max(p_i−p_f,1e-30)`; outer `nan_to_num(mu)` | ✓ |

*`1/β_i²` vs `1/p_i²`: since `p = γβ`, `1/p_i² = 1/(γ_i² β_i²)`. In the
non-relativistic limit `γ_i → 1` the two coincide, so the code reduces to the
standard Koch–Motz NR 3BN. The code deliberately uses relativistic momenta
`p`/`β` as a mildly-relativistic extension of an intrinsically NR Born formula;
the docstring flags this as an approximation ("Adequate for Z≲30 and T≲100 keV;
swap in Seltzer–Berger tables for better accuracy"). This is a documented
modeling choice, not an error, and is self-consistent (`p_i`, `β_i`, `p_f`,
`β_f` all use the same relativistic kinematics).

### Assembly (lines 714–757)

`spec += (n_i * 1e24 * path_cm) @ (dsig * T_abs)`, then
`return spec / (4π) / Ne`. Unit chain: `n_i·1e24` (Å⁻³→cm⁻³) · `path_cm`
(Å·1e-8→cm) · `dsig` (cm²/eV) = photons/eV; `/(4π)` → /sr; `/Ne` → per
incident electron; `T_abs = exp(−L_esc·μ)` Beer–Lambert escape (dimensionless).
Matches the intended `d2N/dE dOmega` [photons/eV/sr/electron]. Compound loop
sums elements each with its own `Z**2` via `TRANSPORT_ELEMENTS[el]["Z"]` — the
additive-`Z²` weighting of §1. ✓

### Numeric spot check (independent reimplementation, not the code helper)

Z=14 (Si), T=50 keV; my numpy reimplementation of §2c vs `_brem_dsigma_dk`:

| k | code [cm²/eV] | independent [cm²/eV] | rel diff |
|---|---------------|----------------------|----------|
| 10 keV  | 8.91244e-28 | 8.91244e-28 | −2e-16 |
| 100 eV  | 2.23037e-25 | 2.23037e-25 | 0 |
| 49.9 keV| 8.50614e-29 | 8.50614e-29 | 0 |
| 50 keV  | 0           | 0           | (hard cutoff) |
| 0 eV    | 0           | 0           | (clamp)  |

Machine-precision agreement. Constants confirmed: `ALPHA_FS = 1/137.035999`
(materials/crystal.py:42), `R_E_CM2 = 7.9407877e-26 cm²` = `r_e²`
(spectrum.py:578).

**Tip approach** (k = 45 → 49.999 keV at T=50 keV): dsig = 9.94e-29, 9.60e-29,
9.28e-29, 8.99e-29, 8.73e-29, 8.48e-29 — **finite and slowly varying**, then
exact `0` for `k ≥ T`. This confirms §2d: the Elwert-corrected code does NOT go
to 0 as `k → T⁻`; it stays finite (correct Kramers-like tip) and only the hard
kinematic cutoff at `k ≥ T` gives exactly 0.

### Adjudication of the "spectrum → 0 at tip" limiting case

The naive limiting-case phrasing "spectrum → 0 at the tip" describes **bare
Born**, not the ledgered **Born + Elwert** claim. The implementation correctly
reproduces Born + Elwert: finite as `k → T⁻`, zero for `k ≥ T`. This is a
correction to the *expectation phrasing*, not a code discrepancy — the code
matches the cited source. Ledger note "benign 0-eV divide-by-zero clamped" is
accurate and implemented (`k>0` mask + `max()` guards + `nan_to_num`).

---

## 5. Verdict

- **Filters:** units PASS · limits PASS (Elwert→1, IR 1/k, tip cutoff,
  finite-tip) · signs/conventions PASS.
- **Re-derivation:** `matches`. Every term (prefactor 16/3·α·r_e²·Z², 1/k, Born
  log, Elwert factor, tip cutoff, 0-eV clamp) matches symbolically,
  dimensionally, and numerically to machine precision. The only nuances are (i)
  `1/p_i²` vs `1/β_i²` — a documented relativistic-momentum extension of the NR
  formula, NR-equivalent and internally consistent, and (ii) the "spectrum → 0
  at tip" expectation applies to bare Born, whereas the ledgered Born+Elwert
  correctly gives a finite tip below `k=T` and hard zero above.
- **Status suggestion:** `rederived`.
- **Not signed-off** (human-only).
