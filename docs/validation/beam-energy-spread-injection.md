# `beam-energy-spread-injection` — independent re-derivation

- **Ledger row**: per-electron relative energy spread `δ ~ N(0, σ_δ)`,
  `E_i = E₀(1 + δ_i)`, drawn independently of the longitudinal arrival
  offsets; fractional line shift `S δ` with
  `S = (γ−1)/(γ³β²(1 − β cos θ_obs))`.
- **Anchors**: `montecarlo/transport.py::simulate_trajectories` (sampling),
  `energy_grid/bounds.py::line_shift_fraction` (closed form).
- **Source cited**: PXR resonance condition used by the kernel,
  `ω = v·g / (1 − n·v)` (`montecarlo/spectrum.py::_line_kin_core`), plus the
  standard relativistic relation between kinetic energy and speed.
- **Assumptions stated**: standard Gaussian relative energy/momentum spread,
  no energy-position chirp; first order in `δ`; fixed emission direction
  (spread moves the line, does not redistribute it over angle); `S` is
  differentiated from the resonance formula at fixed geometry (fixed `g`,
  fixed beam direction, fixed `θ_obs`).
- **Limiting cases claimed**: `δ → 0` ⇒ no shift; `γ → 1` (non-relativistic)
  ⇒ `S → 1/2`; monotone falling `S` over 30–300 keV; `θ_obs = 90°` used as
  the catalog's worst-case geometry.

## 1. Cheap filters

**Units.** `δ = dT/T` and `S δ = dω/ω` are both dimensionless ratios, so `S`
itself must be dimensionless. In the claimed closed form, `γ`, `β`, and
`cos θ_obs` are all dimensionless, so `S` is dimensionless. **Pass.**

**Sign/convention.** The Doppler-type denominator `1 − β cos θ_obs` is
strictly positive for any physical `β < 1` and `|cos θ_obs| ≤ 1`, so `S > 0`
everywhere in the physical domain — no spurious pole or sign flip. This
matches the code's `E_i = E₀(1+δ_i)` construction, which is symmetric under
`δ → −δ` (broadening in both directions with the same `|S|`), consistent
with `S` itself carrying no explicit sign dependence on `δ`. **Pass.**

**Limiting cases (to be confirmed below).** Deferred to §3.

## 2. Independent re-derivation

### 2.1 From the resonance condition to `dω/ω`

Take the cited PXR resonance, with reciprocal-lattice vector `g` and
observation direction `n` both held fixed (fixed geometry), and electron
velocity `v = β v̂` along the fixed beam direction `v̂`:

```
ω(β) = v·g / (1 − n·v) = β (v̂·g) / (1 − β cos θ_obs)
```

where `cos θ_obs ≡ n·v̂` is fixed by geometry and `G∥ ≡ v̂·g` is a fixed
constant (both held while only `β` varies with energy spread).

Differentiate with respect to `β`:

```
dω/dβ = G∥ · [(1 − β cos θ_obs) − β(−cos θ_obs)] / (1 − β cos θ_obs)²
       = G∥ / (1 − β cos θ_obs)²
```

(the `β cos θ_obs` terms in the numerator cancel exactly). Divide by
`ω = β G∥ / (1 − β cos θ_obs)`:

```
dω/ω = (dβ/dω)⁻¹ ... → dω/ω = (dβ/β) / (1 − β cos θ_obs)          (★)
```

This reproduces the intermediate result stated in the docstring exactly.

### 2.2 From kinetic-energy spread to `dβ/β`

Use `β² = 1 − 1/γ²` with `γ = 1 + T/(m_e c²)`. Differentiating:

```
2β dβ = 2 dγ/γ³  ⟹  dβ/β = dγ/(γ³β²)
```

`γ = 1 + T/(m_e c²)` gives `dγ = dT/(m_e c²)`. Writing the relative energy
deviation as `δ = dT/T` and using `T = (γ−1) m_e c²`:

```
dT = T δ = (γ−1) m_e c² δ
dγ = dT/(m_e c²) = (γ−1) δ
dβ/β = (γ−1) δ / (γ³ β²)                                           (★★)
```

This reproduces the docstring's second intermediate result exactly.

### 2.3 Combine

Substitute (★★) into (★):

```
dω/ω = [(γ−1) δ / (γ³β²)] / (1 − β cos θ_obs)
      = δ · (γ−1) / [γ³ β² (1 − β cos θ_obs)]
```

so

```
S = (γ−1) / [γ³ β² (1 − β cos θ_obs)]
```

**This is exactly the claimed closed form**, independently re-derived from
the cited resonance condition and the standard `γ(T)`, `β(γ)` relations,
with no free parameters or fitted constants.

Equivalently, `S = (E₀/ω) dω/dE` at fixed geometry (chain rule on `δ = dE/E₀`
applied to `dω/ω = S δ`), which matches the alternate definition quoted in
the task.

## 3. Limiting-case and monotonicity checks (independent numerics)

Using `m_e c² = 510.99895` keV and the closed form above, computed from
scratch (no import of `cxr_mc`):

```
γ→1 (T = 1 meV):        S = 0.499998...   → 0.5 to 5+ significant figures
S(30 keV,  cosθ=0):     S = 0.45881
S(60 keV,  cosθ=0):     S = 0.42265
S(100 keV, cosθ=0):     S = 0.38090
S(150 keV, cosθ=0):     S = 0.33706
S(200 keV, cosθ=0):     S = 0.30054
S(250 keV, cosθ=0):     S = 0.26976
S(300 keV, cosθ=0):     S = 0.24355
```

- **Non-relativistic limit.** Taylor-expanding `γ = 1+ε` for small `ε`:
  `β² = 1 − 1/(1+ε)² ≈ 2ε` to leading order, and `β cos θ_obs → 0` as
  `β → 0`, so `S ≈ ε/(2ε) = 1/2`. Numerically confirmed to 5 significant
  figures at `T = 1 meV`. **Matches claim.**
- **Monotone falloff 30–300 keV.** The tabulated values above strictly
  decrease with energy at `cos θ_obs = 0`; `S(30 keV) ≈ 0.459` is the largest
  in that band, consistent with the ledger's stated worst case
  (`S ≈ 0.46` at 30 keV, matching the `test_catalog_margin_...` docstring
  comment verbatim). **Matches claim.**
- **Forward vs. side observation.** At 300 keV, `δ = 0.01`: side
  (`cos θ_obs = 0`) gives `S δ ≈ 0.00244`; forward (`cos θ_obs = 1`) gives
  `S δ ≈ 0.01090`, i.e. forward observation amplifies the shift by roughly
  4.5×, consistent with relativistic Doppler beaming making the shift purely
  geometric via the `(1 − β cos θ_obs)` denominator. **Matches claim** (and
  matches the qualitative direction of `test_forward_observation_raises_the_sensitivity`).
- **`δ → 0`.** `S·0 = 0` trivially — no shift. **Matches claim.**

All limiting-case and monotonicity filters pass; no discrepancy found in the
`S` formula.

## 4. Sampling / independence claim (`E_i = E₀(1+δ_i)`, `⟨t δ⟩ = 0`)

`simulate_trajectories` draws `δ_i` as i.i.d. standard normals scaled by
`energy_spread_frac`, from a dedicated RNG child stream
(`SeedSequence(seed).spawn(6)[5]`), applied as
`E_i = E₀ · (1 + f · u_i)`. The longitudinal arrival-time offsets are drawn
from a *separate* child stream (`spawn(4)[3]`, per the docstring). Because
the two quantities are constructed from statistically independent RNG
streams (disjoint `SeedSequence` spawns, no shared draws or shared
transformation), `Cov(t, δ) = 0` in expectation by construction — this is a
structural independence argument, not a numerical coincidence, and requires
no separate derivation beyond confirming the two draws do not share any
random input. `σ_δ → 0`/`energy_spread_frac` unset reproduces the
monoenergetic beam bit-for-bit (guarded by `if energy_spread_frac:`), and a
non-positive drawn energy is a hard `ValueError` rather than a silent clamp,
consistent with the docstring's stated Gaussian-model validity range
(`σ_δ ≪ 1`).

## 5. Comparison against the implementation

- `energy_grid/bounds.py::line_shift_fraction`: computes
  `gamma = 1 + E/m_ec²`; `beta_sq = 1 − 1/gamma**2`;
  `sensitivity = (gamma−1)/(gamma**3 * beta_sq * (1 − sqrt(beta_sq)*cos_theta_obs))`;
  returns `sensitivity * energy_spread_frac`. This is symbolically identical,
  term for term, to the independently re-derived `S` and `S·δ` above — same
  numerator `(γ−1)`, same `γ³β²` factor, same `(1 − β cos θ_obs)` Doppler
  denominator, no extra factors of 2, no sign inversion.
- `montecarlo/spectrum.py::_line_kin_core` / its caller: uses
  `denom = 1 − v·n` (i.e. `1 − n·v`) and `omega_res = v_dot_g/denom`,
  confirming the cited resonance convention `ω = v·g/(1 − n·v)` is in fact
  the one the derivation started from (not a mismatched sign or a `1+n·v`
  variant).
- `montecarlo/transport.py::simulate_trajectories`: `energy_spread_frac`
  branch matches the claimed sampling law
  `E_keV = E_keV * (1 + f * standard_normal(Ne))` on its own RNG substream,
  with the same hard-error-on-non-positive-energy behavior described above.

No divergent term, sign, exponent, or convention found between the
independent derivation and the implementation.

## Verdict

- **Claim**: `beam-energy-spread-injection` —
  `montecarlo/transport.py::simulate_trajectories`,
  `energy_grid/bounds.py::line_shift_fraction` — PXR resonance
  `ω = v·g/(1−n·v)` (`spectrum.py::_line_kin_core`), differentiated at fixed
  geometry to give `S = (γ−1)/(γ³β²(1−β cos θ_obs))`.
- **Filters**: units `pass`; limits `pass` (`γ→1 ⇒ S→1/2`, `δ→0 ⇒` no shift,
  monotone fall 30–300 keV at `cos θ_obs=0`); signs/conventions `pass`
  (`S>0` everywhere physical, resonance-denominator sign matches
  `_line_kin_core`'s `1 − n·v`).
- **Re-derivation**: `matches` — the closed form for `S`, the sampling law
  `E_i = E₀(1+δ_i)`, and the RNG-stream independence argument for
  `⟨t δ⟩ = 0` all reproduce the implementation exactly; no divergent term
  found.
- **Verdict**: `filtered` (re-derivation corroborates; ledger status
  unchanged pending anchoring/human sign-off — this write-up does not
  itself flip the ledger).
- **Write-up**: `docs/validation/beam-energy-spread-injection.md`.
- **Suggested ledger change**: none required by this re-derivation; existing
  `filtered` status and cited anchor tests (`tests/test_transverse.py`,
  `tests/energy-grid/test_bounds.py`, `tests/montecarlo/test_beam_energy_spread_grid.py`,
  all green locally) are consistent with promotion to `anchored` if a human
  wants to formalize that the cited tests already pin the reference values
  used above (`S(30 keV, cos θ_obs=0) ≈ 0.4588`, `γ→1 ⇒ S→0.5`). Sign-off
  remains a human action.
