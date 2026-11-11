# `beam-energy-spread-injection` — independent re-derivation

> Ownership refresh (2026-10-20, #599): current entry points replace retired
> monolithic module paths below. Dated source excerpts, line numbers, numerical
> evidence and verdicts describe their original revision; this reference repair
> does not independently revalidate those equations or extend their evidence.

- **Ledger row**: per-electron relative energy spread `δ ~ N(0, σ_δ)`, `E_i = E₀(1 + δ_i)`, drawn independently of the longitudinal arrival offsets; fractional line shift `S δ` with `S = (γ−1)/(γ³β²(1 − β cos θ_obs))`.
- **Anchors**: `montecarlo/transport/api.py::_simulate_trajectories` (sampling), `energy_grid/bounds.py::line_shift_fraction` (closed form).
- **Source cited**: PXR resonance condition used by the kernel, `ω = v·g / (1 − n·v)` (`montecarlo/spectrum/cxr/_kernels.py::_cxr_kin_core`), plus the standard relativistic relation between kinetic energy and speed.
- **Assumptions stated**: standard Gaussian relative energy/momentum spread, no energy-position chirp; first order in `δ`; fixed emission direction (spread moves the line, does not redistribute it over angle); `S` is differentiated from the resonance formula at fixed geometry (fixed `g`, fixed beam direction, fixed `θ_obs`).
- **Limiting cases claimed**: `δ → 0` ⇒ no shift; `γ → 1` (non-relativistic) ⇒ `S → 1/2`; monotone falling `S` over 30–300 keV; `θ_obs = 90°` used as the catalog's worst-case geometry.

## 1. Cheap filters

**Units.** `δ = dT/T` and `S δ = dω/ω` are both dimensionless ratios, so `S` itself must be dimensionless. In the claimed closed form, `γ`, `β`, and `cos θ_obs` are all dimensionless, so `S` is dimensionless. **Pass.**

**Sign/convention.** The Doppler-type denominator `1 − β cos θ_obs` is strictly positive for any physical `β < 1` and `|cos θ_obs| ≤ 1`, so `S > 0` everywhere in the physical domain — no spurious pole or sign flip. This matches the code's `E_i = E₀(1+δ_i)` construction, which is symmetric under `δ → −δ` (broadening in both directions with the same `|S|`), consistent with `S` itself carrying no explicit sign dependence on `δ`. **Pass.**

**Limiting cases (to be confirmed below).** Deferred to §3.

## 2. Independent re-derivation

### 2.1 From the resonance condition to `dω/ω`

Take the cited PXR resonance, with reciprocal-lattice vector `g` and observation direction `n` both held fixed (fixed geometry), and electron velocity `v = β v̂` along the fixed beam direction `v̂`:

```
ω(β) = v·g / (1 − n·v) = β (v̂·g) / (1 − β cos θ_obs)
```

where `cos θ_obs ≡ n·v̂` is fixed by geometry and `G∥ ≡ v̂·g` is a fixed constant (both held while only `β` varies with energy spread).

Differentiate with respect to `β`:

```
dω/dβ = G∥ · [(1 − β cos θ_obs) − β(−cos θ_obs)] / (1 − β cos θ_obs)²
       = G∥ / (1 − β cos θ_obs)²
```

(the `β cos θ_obs` terms in the numerator cancel exactly). Divide by `ω = β G∥ / (1 − β cos θ_obs)`:

```
dω/ω = (dβ/dω)⁻¹ ... → dω/ω = (dβ/β) / (1 − β cos θ_obs)          (★)
```

This reproduces the intermediate result stated in the docstring exactly.

### 2.2 From kinetic-energy spread to `dβ/β`

Use `β² = 1 − 1/γ²` with `γ = 1 + T/(m_e c²)`. Differentiating:

```
2β dβ = 2 dγ/γ³  ⟹  dβ/β = dγ/(γ³β²)
```

`γ = 1 + T/(m_e c²)` gives `dγ = dT/(m_e c²)`. Writing the relative energy deviation as `δ = dT/T` and using `T = (γ−1) m_e c²`:

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

**This is exactly the claimed closed form**, independently re-derived from the cited resonance condition and the standard `γ(T)`, `β(γ)` relations, with no free parameters or fitted constants.

Equivalently, `S = (E₀/ω) dω/dE` at fixed geometry (chain rule on `δ = dE/E₀` applied to `dω/ω = S δ`), which matches the alternate definition quoted in the task.

## 3. Limiting-case and monotonicity checks (independent numerics)

Using `m_e c² = 510.99895` keV and the closed form above, computed from scratch (no import of `pyrite`):

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

- **Non-relativistic limit.** Taylor-expanding `γ = 1+ε` for small `ε`: `β² = 1 − 1/(1+ε)² ≈ 2ε` to leading order, and `β cos θ_obs → 0` as `β → 0`, so `S ≈ ε/(2ε) = 1/2`. Numerically confirmed to 5 significant figures at `T = 1 meV`. **Matches claim.**
- **Monotone falloff 30–300 keV.** The tabulated values above strictly decrease with energy at `cos θ_obs = 0`; `S(30 keV) ≈ 0.459` is the largest in that band, consistent with the ledger's stated worst case (`S ≈ 0.46` at 30 keV, matching the `test_catalog_margin_...` docstring comment verbatim). **Matches claim.**
- **Forward vs. side observation.** At 300 keV, `δ = 0.01`: side (`cos θ_obs = 0`) gives `S δ ≈ 0.00244`; forward (`cos θ_obs = 1`) gives `S δ ≈ 0.01090`, i.e. forward observation amplifies the shift by roughly 4.5×, consistent with relativistic Doppler beaming making the shift purely geometric via the `(1 − β cos θ_obs)` denominator. **Matches claim** (and matches the qualitative direction of `test_forward_observation_raises_the_sensitivity`).
- **`δ → 0`.** `S·0 = 0` trivially — no shift. **Matches claim.**

All limiting-case and monotonicity filters pass; no discrepancy found in the `S` formula.

## 4. Sampling / independence claim (`E_i = E₀(1+δ_i)`, `⟨t δ⟩ = 0`)

`simulate_trajectories` draws `δ_i` as i.i.d. standard normals scaled by `energy_spread_frac`, from a dedicated RNG child stream (`SeedSequence(seed).spawn(6)[5]`), applied as `E_i = E₀ · (1 + f · u_i)`. The longitudinal arrival-time offsets are drawn from a *separate* child stream (`spawn(4)[3]`, per the docstring). Because the two quantities are constructed from statistically independent RNG streams (disjoint `SeedSequence` spawns, no shared draws or shared transformation), `Cov(t, δ) = 0` in expectation by construction — this is a structural independence argument, not a numerical coincidence, and requires no separate derivation beyond confirming the two draws do not share any random input. `σ_δ → 0`/`energy_spread_frac` unset reproduces the monoenergetic beam bit-for-bit (guarded by `if energy_spread_frac:`), and a non-positive drawn energy is a hard `ValueError` rather than a silent clamp, consistent with the docstring's stated Gaussian-model validity range (`σ_δ ≪ 1`).

## 5. Comparison against the implementation

- `energy_grid/bounds.py::line_shift_fraction`: computes `gamma = 1 + E/m_ec²`; `beta_sq = 1 − 1/gamma**2`; `sensitivity = (gamma−1)/(gamma**3 * beta_sq * (1 − sqrt(beta_sq)*cos_theta_obs))`; returns `sensitivity * energy_spread_frac`. This is symbolically identical, term for term, to the independently re-derived `S` and `S·δ` above — same numerator `(γ−1)`, same `γ³β²` factor, same `(1 − β cos θ_obs)` Doppler denominator, no extra factors of 2, no sign inversion.
- `montecarlo/spectrum/cxr/_kernels.py::_cxr_kin_core` / its caller: uses `denom = 1 − v·n` (i.e. `1 − n·v`) and `omega_res = v_dot_g/denom`, confirming the cited resonance convention `ω = v·g/(1 − n·v)` is in fact the one the derivation started from (not a mismatched sign or a `1+n·v` variant).
- `montecarlo/transport/api.py::_simulate_trajectories`: `energy_spread_frac` branch matches the claimed sampling law `E_keV = E_keV * (1 + f * standard_normal(Ne))` on its own RNG substream, with the same hard-error-on-non-positive-energy behavior described above.

No divergent term, sign, exponent, or convention found between the independent derivation and the implementation.

## Verdict

- **Claim**: `beam-energy-spread-injection` — `montecarlo/transport/api.py::_simulate_trajectories`, `energy_grid/bounds.py::line_shift_fraction` — PXR resonance `ω = v·g/(1−n·v)` (`spectrum.py::_cxr_kin_core`), differentiated at fixed geometry to give `S = (γ−1)/(γ³β²(1−β cos θ_obs))`.
- **Filters**: units `pass`; limits `pass` (`γ→1 ⇒ S→1/2`, `δ→0 ⇒` no shift, monotone fall 30–300 keV at `cos θ_obs=0`); signs/conventions `pass` (`S>0` everywhere physical, resonance-denominator sign matches `_cxr_kin_core`'s `1 − n·v`).
- **Re-derivation**: `matches` — the closed form for `S`, the sampling law `E_i = E₀(1+δ_i)`, and the RNG-stream independence argument for `⟨t δ⟩ = 0` all reproduce the implementation exactly; no divergent term found.
- **Verdict**: `filtered` (re-derivation corroborates; ledger status unchanged pending anchoring/human sign-off — this write-up does not itself flip the ledger).
- **Write-up**: `docs/validation/beam-transport/beam-energy-spread-injection.md`.
- **Suggested ledger change**: none required by this re-derivation; existing `filtered` status and cited anchor tests (`tests/montecarlo/transport/test_transverse.py`, `tests/energy-grid/test_bounds.py`, `tests/montecarlo/spectrum/test_beam_energy_spread_grid.py`, all green locally) are consistent with promotion to `anchored` if a human wants to formalize that the cited tests already pin the reference values used above (`S(30 keV, cos θ_obs=0) ≈ 0.4588`, `γ→1 ⇒ S→0.5`). Sign-off remains a human action.

## Counter-addressed energy draws (#361 re-verification, 2026-10-07)

`beam_entry.py::initial_energies_keV` sets $E_e=E_0(1+f\,Z_{e,0})$, with $Z_{e,0}=\Phi^{-1}(u_{e,0})$ in the `spawn(6)[5]` child. The construction is derived in [beam-phase-space-injection §6](beam-phase-space-injection.md). Independent checks at $E_0=100$ keV, $f=0.01$ and $N=10^6$ gave mean $\delta=6.5\times10^{-6}$ and $\sigma_\delta/f=0.99987$. Row $e=5$ equals the reference exactly, the block `start=500` equals the slice, and $f\in\{0,\texttt{None}\}$ gives exactly $E_0$. $\rho(\delta, t_{\rm bunch})=5.6\times10^{-4}$, consistent with zero, from disjoint children. The non-positive-energy guard is unchanged. No discrepancy.

## Independent verification of the 6σ clamp (fresh context, 2026-11-11)

This is a fresh-context check at commit `491208cde`. I read the ledger row, the
signatures and the docstrings first. I derived the law below before reading the
body of `initial_energies_keV` or `upper_beam_energy_keV`.

### Cheap filters

- **Units:** $\delta$ and $z$ are dimensionless, and $E_i=E_0(1+\delta_i)$ is in keV. Pass.
- **Limits:**
  - $\sigma_\delta\to0$ gives $E_i=E_0$.
  - $k\to\infty$ recovers the unclamped Gaussian.
  - $\sigma_\delta<1/k=1/6$ guarantees $E_i>0$.
  - Pass.
- **Symmetry:** the clamp is symmetric, so the law stays even in $z$. Pass.

### Clamped Gaussian law

The draw is $z\sim\mathcal N(0,1)$. It is clipped, not truncated and
renormalized: $\tilde z=\max(-k,\min(k,z))$ with $k=6$. All tail mass is moved
onto the endpoints $\pm k$.

- **Clipped mass:** $P(\lvert z\rvert>k)=2\Phi(-k)=1.9732\times10^{-9}$.
- **Mean:** $\tilde z$ is an odd function of a symmetric law, so $\langle\tilde z\rangle=0$ exactly.
- **Variance:**

  $$
  \langle\tilde z^2\rangle
  =\bigl[2\Phi(k)-1-2k\varphi(k)\bigr]+k^2\,2\Phi(-k)
  =1-2\bigl[k\varphi(k)-(k^2-1)\Phi(-k)\bigr],
  $$

  so $1-\langle\tilde z^2\rangle=3.849\times10^{-9}$ at $k=6$ (`mpmath`, 40 digits). The
  RMS spread is therefore $\sigma_\delta(1-1.9\times10^{-9})$. That difference is
  negligible, but the ledger statement "RMS $\sigma_\delta E_0$" is exact only to
  this order.
- **Strict upper energy:** $E_i\le E_0(1+k\lvert\sigma_\delta\rvert)=E_{\rm up}$
  holds with equality only at the clip. In IEEE arithmetic
  `E0 * (1.0 + f * 6.0)` and `E0 * (1.0 + 6.0 * abs(f))` are bit-identical,
  including for $f<0$ at $z=-6$. So $E_{\rm up}$ is strict without a rounding
  margin. I checked 200 000 random $(E_0,f)$ pairs at $z=\pm6$ and found 0
  violations.
- **Line shift:** the fractional line shift
  $S=(\gamma-1)/[\gamma^3\beta^2(1-\beta\cos\theta_{\rm obs})]$ is a pointwise
  first-order derivative of the resonance in $\delta$. It does not depend on the
  law of $\delta$, so the clamp leaves its validity unchanged. Re-derived:
  $2\beta\,\mathrm d\beta=2\,\mathrm d\gamma/\gamma^3$ and
  $\mathrm d\gamma=(\gamma-1)\delta$ give
  $\mathrm d\beta/\beta=(\gamma-1)\delta/(\gamma^3\beta^2)$.

### Comparison with the implementation

`initial_energies_keV` matches the derived law:

- It computes `z = np.clip(counter_normals(...)[:, 0], -6.0, 6.0)`.
- The energy is then `E = E0 * (1.0 + f * z)`.
- The whole branch runs only `if energy_spread_frac:`, so `None` and `0` keep
  `np.full(Ne, E0)` exactly.
- A nonfinite or non-positive energy still raises.

`upper_beam_energy_keV` returns `E0 * (1.0 + 6.0 * abs(f or 0.0))`, which also
matches. Before the clamp, `counter_normals` already bounded
$\lvert z\rvert\le8.2095$ (the `ndtri` limit of the smallest uniform). The clamp
therefore changes only draws in $6<\lvert z\rvert\le8.2095$.

### Numerics

All draws come from `initial_energies_keV`, with seed 12345 and
$4\times10^6$ electrons at $E_0=300$ keV:

| $f$ | max $E$ (keV) | $E_{\rm up}$ (keV) | mean $z$ | RMS $z$ | $z$ range |
|---|---|---|---|---|---|
| 0.01 | 314.954 | 318.0 | $5.0\times10^{-4}$ | 1.00026 | $[-5.20, 4.98]$ |
| 0.1 | 449.544 | 480.0 | $5.0\times10^{-4}$ | 1.00026 | $[-5.20, 4.98]$ |

The mean and RMS sit within $1\sigma$ of their sampling errors
($5\times10^{-4}$ and $3.5\times10^{-4}$). In both rows the maximum is at most
$E_{\rm up}$.

- **Monoenergetic limit:** $f\in\{0,\texttt{None}\}$ gives the same bytes as
  `np.full(1000, 300.0)`, and `upper_beam_energy_keV` returns exactly $E_0$.
- **Block addressing:** block `start=1000` equals the slice of the joined call.
- **Clamp branch:** the anchor
  `test_energy_spread_draw_is_clamped_at_the_declared_upper_energy` monkeypatches
  draws $\{-9,-6,0,2.5,6,8.2\}$ and gets $\{94,94,100,102.5,106,106\}$ keV. A
  natural draw with $\lvert z\rvert>6$ needs about $5\times10^8$ electrons, so the
  monkeypatch is the right test.
- **Test run:** the anchors in `test_transverse.py`, `test_bounds.py` and
  `test_beam_energy_spread_grid.py` pass (81 passed alongside the
  line-grid-policy module, 1 slow test skipped).

### Consumers

Every analytic spread consumer goes through `initial_energies_keV`:

- `sample_primary_entry`
- `host_pass`
- `block_transport`
- the `secondaries` table range
- the channeling `sample_incident_phase_space`, which calls the primary sampler

Every consumer that builds an automatic line grid or line ceiling evaluates
$E_{\rm up}$ from the same `energy_spread_frac` the case carries:

- `campaign/sweep.py::_automatic_cxr_grid_policy`
- `directional.py::line_window_ceiling_eV`

GDF sources refuse a spread. The channeling `IncidentEnergyEnsemble`
(`cutoff_sigma`) is a separate diagnostic quadrature that is not wired to line
grids.

### Documentation drift (non-blocking; not edited)

- The `transport/api.py::_simulate_trajectories` docstring still says
  `u ~ Normal(0, 1)` with no clamp.
- `docs/physics/beam-transport/beam-phase-space.md` still writes
  `delta_i ~ N(0, energy_spread_frac)` with no clamp.
- The `energy_grid/bounds.py::line_shift_fraction` docstring cites
  `montecarlo/spectrum/lines.py::_cxr_kin_core`, and the ledger `Source` cites
  `spectrum.py::_cxr_kin_core`. Neither file exists; the kernel is
  `montecarlo/spectrum/cxr/_kernels.py::_cxr_kin_core`.
- The `initial_energies_keV` docstring calls the clip a "truncation". It is a
  clamp onto the endpoints, not a renormalized truncation. The mean is
  unchanged either way.

### Verdict

`rederived`. The clamped law (zero mean, variance deficit $3.85\times10^{-9}$,
clipped mass $2\Phi(-6)$), the bitwise-strict $E_{\rm up}$, the bit-for-bit
$f=0$ limit and the unchanged $S$ all match the implementation. No human
sign-off is implied.
