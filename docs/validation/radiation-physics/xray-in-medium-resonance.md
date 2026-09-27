# `xray-in-medium-resonance`

## Claim and source

- Claim: CXR line kinematics closed on the in-medium photon dispersion `k = Re n(ω) ω n̂`: resonance `ω_res = v·g / (1 − Re n (v·n̂))`, `k·v = ω(1 − denom)`, `k·g = Re n ω (n̂·g)`, PXR detuning `|k+g|² − k² = g² + 2 k·g`, and PXR numerator `k² = (Re n ω)²`.
- Anchor: `montecarlo/spectrum/lines.py::_in_medium_kinematics`, `::mc_spectrum` (`xray_dispersion="refractive"`); CUDA port `montecarlo/spectrum/coherent_stream_jit_kernel.py::_coherent_prologue_kernel`.
- Source: energy–momentum conservation `ω = v·(k+g)` closed with the Maxwell dispersion relation in a homogeneous dielectric, `k² = (1+χ₀)ω²` (`xray-refractive-index`); Feranchuk–Spence 2000 Eq. (10)/(13) with `k² → εω²` rather than an ad hoc `n` inserted into the vacuum result.
- Intended quantity/signature (from the derivation docstring, read before the implementation body): `_in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab) -> (denom, n_re)`, with `denom` and `n_re` broadcast to the shape of `v_dot_g`; the docstring states the governing implicit equation `omega_res = v.g / (1 - Re n(omega_res)(v.n_hat))`, that only `Re n` enters (`Im n` is folded into the existing Beer–Lambert `mu(E)` escape factor, so applying it again here would double-count absorption), that the equation is solved by 3-pass fixed-point iteration from the vacuum root because the map's derivative is `~delta~1e-5`, and that the caller reconstructs `k·v = omega(1-denom)` exactly while `k·g` and `k²` pick up one and two powers of `n_re` respectively. The companion `mc_spectrum` docstring adds: `xray_dispersion="vacuum"` (default) keeps `k=omega` bit-for-bit; `"refractive"` uses `n = sqrt(1+chi_0)` (`materials.crystal.refractive_index`, `xray-refractive-index`), only the real part is applied, and the model is bulk-only (no interface/Fresnel refraction, so grazing observation geometry is out of scope).

## Independent derivation

### 0. Vacuum baseline (already-established result)

`line-energy-dispersion` / `coherent-emission` fix the repository's sign convention: for a charge on a straight trajectory `r(t) = r_c + v(t-t_c)` radiating into an outgoing vacuum mode `k = ω n̂` while coupling to a lattice harmonic `g` (susceptibility reconstructing as `χ_g exp(-i g·r)` in real space, per the resolved `coherent-emission` sign pivot), the phase is

```
Φ_g(t) = ω(t - n̂·r(t)) - g·r(t)
```

and stationarity (`dΦ/dt = 0`) gives exactly the energy–momentum matching condition cited by this claim,

```
ω = v·(k+g) = k·v + g·v.                                            (A)
```

Substituting the vacuum dispersion relation `k = ω n̂` into (A) gives the already-rederived vacuum resonance `ω_res = v·g/(1 - v·n̂)`. This claim's job is to redo the same closure with the medium's dispersion relation instead of the vacuum one, and to track how the extra factors of `n` propagate into the downstream algebraic pieces used by the PXR amplitude.

### 1. A vector identity that needs no physics

For *any* vector `k` and fixed `g`,

```
|k+g|² - k² = k² + 2k·g + g² - k² = g² + 2k·g.                      (B)
```

This is pure algebra — true whatever dispersion relation fixes `k`. It is recorded here because the claim calls it "PXR detuning" and because, below, substituting the *in-medium* `k` into it is the only step needed to carry the detuning from vacuum to medium; no new derivation is required at this step, only bookkeeping of which power of `n` enters.

### 2. Closing energy–momentum conservation with the medium dispersion relation

`xray-refractive-index` establishes the Maxwell dispersion relation for a homogeneous dielectric, `k² = (1+χ₀(ω))ω² ≡ ε(ω)ω²`, with `n(ω) = √(1+χ₀(ω))` so `|k| = n(ω) ω`. This claim's scope note says only the real part of `n` propagates into the *kinematics* (the imaginary part already enters the amplitude through the existing Beer–Lambert transmission factor, so re-applying it in `k` would double-count absorption), and that the bulk medium has no interface/Fresnel term — i.e. the direction of the emitted in-medium photon is taken to remain the vacuum observation direction `n̂` (no ray bending at the exit surface is modelled; consistent with `δ ~ 10⁻⁵–10⁻³` making any such bending negligible pointwise, the same scope caveat carried by `xray-refractive-index` itself). Under those two restrictions the in-medium wavevector entering the kinematics is

```
k = Re n(ω) · ω · n̂.                                                 (C)
```

Substituting (C) into the *same* energy–momentum matching condition (A) used in the vacuum case (Feranchuk–Spence Eq. (10)/(13), closed with `k → εω²` rather than adjusting the vacuum answer after the fact) gives

```
ω = Re n(ω) ω (n̂·v) + g·v
ω [1 - Re n(ω) (n̂·v)] = g·v.
```

Because `Re n` is itself evaluated at the resonance frequency, this is an implicit equation for the resonance, exactly the claim's

```
ω_res = v·g / (1 - Re n(ω_res)(v·n̂)).                                (D)
```

Defining `denom ≡ 1 - Re n(ω_res)(v·n̂)` (matching the vacuum-limit definition `denom = 1 - v·n̂` when `Re n → 1`) makes (D) `ω_res = v·g/denom`, formally identical in shape to the vacuum resonance with `denom` promoted from a `g`-independent constant to a segment-and-`g`-dependent root of an implicit equation.

**Existence/uniqueness of the root and the stated fixed-point rate.** Write `F(ω) = v·g/(1-Re n(ω)(v·n̂))`. Off-edge, `Re n(ω) = 1-δ(ω)` with `δ ~ 10⁻⁵–10⁻³` and `dδ/dω` smooth and small over the relevant line width, so

```
F'(ω) = v·g · (v·n̂) · dRe n/dω / (1-Re n(v·n̂))²,
```

which is `O(δ)` relative to unity (the same order as the fractional index contrast) whenever `|v·n̂|<1` keeps the denominator bounded away from zero — this reproduces the docstring's stated contraction rate `(v·n̂)(dn/dE)(dE/ddenom) ~ delta ~ 1e-5`. A contraction with ratio `~10⁻⁵` started at the vacuum root `ω_res^(0) = v·g/(1-v·n̂)` therefore gains `~5` decimal digits per iterate; two iterates already reach float64 rounding and the third is margin, matching the docstring's stated "three passes taken."

```{warning}
The premise of this subsection — `Re n(ω) = 1-δ` with `δ ~ 10⁻⁵–10⁻³`, and
`|v·n̂|<1` keeping the denominator bounded away from zero — was falsified on
2026-08-20. It holds off-edge in the X-ray regime, and the argument above is
correct there, but it is not unconditional: see **Addendum 2026-08-20: the
contraction is conditional**, at the end of this write-up. The code no longer
relies on it.
```

### 3. Downstream kinematic identities in terms of the same `denom`

From (C), `k·v = Re n(ω) ω (n̂·v)`. Using the definition of `denom`, `Re n(ω_res)(v·n̂) = 1-denom`, so, evaluated at `ω=ω_res`,

```
k·v = ω_res (1 - denom).                                             (E)
```

This is an identity that holds by construction of `denom` (it is exactly how `denom` was defined from (D)) — it is *not* an independent physical assumption, only bookkeeping that makes `k·v` cheap to recompute from the resonance solve without re-forming the vector `k`. It reduces to the vacuum identity `k·v = ω(1-denom) = ω(v·n̂)` when `Re n → 1`.

From (C) again, taking the dot product with `g` directly,

```
k·g = Re n(ω_res) ω_res (n̂·g).                                       (F)
```

`k·g` therefore carries exactly **one** power of `Re n` relative to the vacuum value `ω(n̂·g)` — the same one power that entered the resonance denominator, applied consistently to the same vector `k`.

Substituting the in-medium `k` into the vector identity (B) gives the "PXR detuning,"

```
|k+g|² - k² = g² + 2k·g = g² + 2 Re n(ω_res) ω_res (n̂·g),            (G)
```

using (F). No new physics enters here beyond (B) and (F): the detuning is just the same algebraic identity evaluated with the medium's `k`.

Finally, the squared magnitude of the in-medium wavevector, which enters the PXR amplitude as the coefficient of the `v·e` term (Feranchuk–Spence Eq. (13)'s `k²` — the term that is `ω²` in vacuum), is, from (C),

```
k² = (Re n(ω_res) ω_res)².                                           (H)
```

`k²` therefore carries **two** powers of `Re n`, i.e. `O(δ)` corrections entering `k·g` and `O(δ²)` (times a compensating `2` from squaring — concretely `(1-δ)²≈1-2δ`) entering `k²`. This asymmetry — one power of `n` in the linear-in-`k` term `k·g`, two powers in the quadratic-in-`k` term `k²` — is exactly what distinguishes *closing momentum conservation with the actual dispersion relation* from an *ad hoc* substitution: an ad hoc scheme that simply multiplied every occurrence of `ω` in the vacuum formula by a single common factor of `n` would not reproduce this graded power structure, since `k·g` and `k²` are respectively linear and quadratic in the same vector `k`. Equations (D), (E), (F), (G), (H) are the complete kinematic content that Feranchuk–Spence's Eq. (10) (resonance/energy conservation) and Eq. (13) (amplitude, with its `k²`-weighted term) require once `k` is defined through `k² = εω²` rather than left at its vacuum value.

## Cheap filters (before comparison)

- **Units.** `v, n̂` dimensionless; `g, k, ω` in Å⁻¹ (`c=1`); `Re n` dimensionless. `denom` is dimensionless, `k·v`/`ω_res` dimensionless, `k·g`/`g²`/`k²` all Å⁻², `E = ℏc ω` in eV. All consistent.
- **Limiting case `Re n → 1`.** (D)→ vacuum `ω_res = v·g/(1-v·n̂)`; (E)→ `k·v = ω(v·n̂)`; (F)→ `k·g = ω(n̂·g)`; (G)→ vacuum detuning `g²+2ω(n̂·g)`; (H)→ `k² = ω²`. Every expression collapses to the already-`rederived`/pre-existing vacuum form bit-for-bit, matching the ledger's stated `xray_dispersion="vacuum"` no-op limiting case.
- **Sign.** With the detector upstream of the beam so `v·n̂ < 0` (the hopg/100 keV/θ_obs=119° regression geometry), `Re n < 1` makes `Re n(v·n̂) > v·n̂` (both negative, `Re n(v·n̂)` less negative), so `denom = 1-Re n(v·n̂) < 1-v·n̂ = denom_vac`; since `ω_res = v·g/denom` with `v·g` fixed by the emission harmonic, a smaller `denom` gives a **larger** `ω_res`, i.e. the line moves up in energy — matching the ledger's recorded sign check.
- **Order of the shift.** `denom - denom_vac = (1-Re n)(v·n̂) = δ(v·n̂)`, so to first order `Δω_res/ω_res ≈ -Δdenom/denom_vac = -δ(v·n̂)/(1-v·n̂)`, matching the closed form the ledger cites for the fractional line shift.
- **Convention parity with the vacuum amplitude.** In vacuum, Feranchuk–Spence Eq. (13)'s amplitude numerator uses `ω²` multiplying the `v·e` term; this claim's role is only to identify which physical quantity `ω²` stands for once the photon is on the medium's mass shell rather than the vacuum one — namely `k²`, not `n²ω²` inserted as an independent multiplicative correction after the fact. (H) shows these are the same thing given (C), so there is no separate convention choice here.

## Implementation comparison

`_in_medium_kinematics(v_dot_n, v_dot_g, n_re_tab, E_tab)` solves

```python
denom = 1.0 - v_dot_n
for _ in range(3):
    E_res = HBARC_EV_ANG * (v_dot_g / denom)
    n_re = interp(n_re_tab, E_res)
    denom = 1.0 - n_re * v_dot_n
return denom, n_re
```

This is exactly the fixed-point iteration for (D): initialize at the vacuum root (`Re n = 1`), then repeatedly evaluate `Re n` at the current resonance estimate and refold it into `denom`. The CUDA port (`_coherent_prologue_kernel`, lines around the `use_medium` branch) performs the identical 3-pass loop in float32 with the same initialization and update, sharing the same docstring-cited contraction argument.

`_line_kin_core` (CPU batched path) and its equivalent inline block in `_accumulate_reflection` (CPU per-hkl path) and `_coherent_prologue_kernel` (CUDA) each then compute, given `denom` and (for the refractive branch) `n_re`:

```python
k_mag = om if n_re_seg is None else om * n_re_seg  # (H)'s |k|
k_dot_v = om * (1.0 - denom)  # (E)
k_dot_g = k_mag * n_dot_g  # (F), n_dot_g already n̂·g
v_dot_kg = v_dot_g + k_dot_v  # v·(k+g) = ω, trivially
detuning = g2 + 2.0 * k_dot_g  # (G)
```

and the PXR amplitude (`lines.py` line ~1070, Feranchuk–Spence Eq. (13)):

```python
A_PXR = chi / detuning * (v_dot_kg * g_dot_e - k_mag**2 * v_dot_e)
```

uses `k_mag**2`, i.e. `(H)`, in exactly the slot the vacuum amplitude fills with `ω²`. Every implementation line maps one-to-one onto (D)–(H):

| independent result | implementation |
|---|---|
| (D) `ω_res = v·g/(1-Re n(ω_res)(v·n̂))` | `_in_medium_kinematics` fixed point |
| (E) `k·v = ω(1-denom)` | `k_dot_v = om * (1.0 - denom)` |
| (F) `k·g = Re n ω (n̂·g)` | `k_dot_g = k_mag * n_dot_g` with `k_mag = om*n_re` |
| (G) `|k+g|²-k² = g²+2k·g` | `detuning = g2 + 2.0 * k_dot_g` |
| (H) `k² = (Re n ω)²` | `k_mag**2` in `A_PXR`'s numerator |

No divergent sign, factor, or power of `n` was found: `k·g` carries exactly one power of `n_re` (through `k_mag`), `k²` carries exactly two (through `k_mag**2`), and `k·v` is left in the `denom`-only form (E) that requires no explicit `n_re` at all — precisely the graded structure predicted by closing momentum conservation on `k = n(ω)ω n̂` rather than scaling the vacuum formula by a single global `n` factor. The `Re`-only restriction (`Im n` excluded to avoid double-counting the Beer–Lambert `mu(E)` factor) and the bulk/no-interface scope note are carried verbatim from the docstring into both the CPU and CUDA implementations and match the assumptions used in Section 2 above. The vacuum branch (`n_re_seg is None` / `use_medium=False`) leaves `k_mag = om`, `denom` unchanged, and every quantity above reduces identically to the vacuum forms, matching the `xray_dispersion="vacuum"` bit-for-bit no-op the ledger records.

## Adjudication

**rederived.**

The resonance condition, its implicit closure and fixed-point contraction rate, the `k·v`/`k·g`/detuning/`k²` identities, and their placement inside the Feranchuk–Spence Eq. (13) amplitude (`k_mag**2` standing in for the vacuum `ω²`) all follow directly from closing energy–momentum conservation `ω = v·(k+g)` with the Maxwell dispersion relation `k² = (1+χ₀)ω²` and `k = Re n(ω) ω n̂`, matching the CPU (batched and per-hkl) and CUDA implementations term for term with no divergent sign, factor, or power of `n`. The graded one-power-in-`k·g` / two-powers-in-`k²` structure is specifically the signature that distinguishes this from an ad hoc insertion of a single index factor, and it is present in the implementation exactly as derived.

Scope carried over unchanged (not re-litigated here, and not undermining the `rederived` verdict): real part only (imaginary part is the existing Beer–Lambert absorption, so it is deliberately not duplicated in `k`); bulk response only, no interface/Fresnel term, so grazing exit geometry is out of scope; this claim depends on `xray-refractive-index` (`filtered`, not yet independently `rederived`) for the dispersion relation `k²=(1+χ₀)ω²` and `n=√(1+χ₀)` itself — a discrepancy discovered in that upstream claim would propagate here.

Suggested ledger action: advance `xray-in-medium-resonance` from `filtered` to `rederived`. Human applies the ledger edit.

## Addendum 2026-08-19: the vacuum-dispersion switch was removed

`xray_dispersion` no longer exists; the in-medium resonance is unconditional. The derivation and its agreement with the implementation are unaffected, and the `rederived` determination stands. Two evidence changes:

- The limiting case `xray_dispersion="vacuum"` -> `k = omega`, bit-for-bit, is no longer a production code path. It survives at kernel level, where the CUDA prologue still accepts `Re n = 1` and collapses onto the vacuum kinematics exactly (`test_prologue_unit_index_reproduces_the_vacuum_kinematics`).
- The host anchor no longer differences a refractive run against a vacuum one. It solves the implicit in-medium root in closed form and asserts the spectral peak lands on it to inside one grid step, then checks the displacement from the vacuum root against `-delta (v.n_hat)/(1 - v.n_hat)` (`test_line_sits_on_the_in_medium_resonance_not_the_vacuum_one`). That pins the root itself, which the differential form did not.

## Addendum 2026-08-20: the contraction is conditional

Section 2's fixed-point rate is derived off-edge, from `Re n(ω) = 1-δ` with `δ ~ 10⁻⁵–10⁻³`, and from `|v·n̂|<1` keeping `denom = 1 - Re n (v·n̂)` bounded away from zero. Both premises fail together outside the X-ray regime, and the resulting defect was found downstream on `feature/relativistic-bethe-stopping` while measuring thick-target spectra: a small fraction of seeds returned a characteristic-line total ~10 orders of magnitude too large, finite rather than NaN, so nothing flagged it.

How it happens, on the traced sample (graphite, 100 μm, 300 keV):

1. A segment scattered nearly perpendicular to `g` gives `v·g = 1.727e-3` against a median `|v·g|` of 1.27, so the vacuum root `E_res = ħc (v·g)/(1 - v·n̂)` is only 6.57 eV.
2. At 6.57 eV carbon's tabulated `Re n` reads **2.07** — correct physics, not a data defect. The tabulation runs down to 1 eV and carries `Re n > 1` over 6.24–285 eV, peaking at 4.766 at 6.40 eV. This is the optical/UV regime, not the regime the solve was derived for.
3. With `v·n̂ = 0.4815`, `Re n (v·n̂) = 0.99931`, so `denom = 6.9e-4`: a spurious Cherenkov-like near-zero, reachable whenever `Re n > 1/(v·n̂)`.
4. The map is then not a contraction but an **expansive 2-cycle**, oscillating between `E_res ~ 6.57 eV` (`Re n ~ 2.07`, `denom ~ 1e-3`) and a keV-scale root (`Re n ~ 1`, `denom ~ 0.5186`). Three hard-coded passes return whichever half of the cycle pass 3 lands on — here `denom = 5.94e-4`, `E_res = 4942.7 eV`, matching the observed peak bin to the bin.
5. That root clears the `E_res > 10` eV keep window, so the cut `cbs-amplitude` names load-bearing for keeping `v·g` away from zero does not stop it: the cut bounds `v·g = ω denom`, so as `denom → 0` it stops bounding `v·g` at all.
6. `A_CBS ~ 1/(γ (v·g)²)` then gives `|A|² = 81.5` against a median `1e-10`.

The perturbative expansion has genuinely failed on such samples, by `cbs-amplitude`'s own stated validity condition `|U_g| g²/(γ mc² (v·g)²) ≪ 1`: with `U_g/mc² = 1.842e-5`, `g² = 3.506 Å⁻²`, `γ = 1.392` and `v·g = 1.727e-3`, the left-hand side is **15.6**. So the correct handling is to reject them, not to solve the root more carefully.

**What the implementation now does.** `_in_medium_kinematics` checks convergence instead of assuming it: the last fixed-point pass must move `denom` by less than `_RESONANCE_ROOT_RTOL = 1e-3`; pairs that fail carry NaN out of `denom` and drop on the caller's existing finite mask, the same route out-of-range tabulation energies already take. A genuine contraction moves `denom` by ~`δ³` ~ 1e-15 in float64 and is floored by float32 rounding (~1e-7) on the device twin, so the tolerance has five orders of margin either side. The CUDA prologue kernel in `coherent_stream_jit_kernel.py` carries the identical guard with the tolerance passed in, the way `hbarc` already is.

Measured: the traced seed falls from 7602 to 8.03e-07, inside the healthy population (6.6e-07–1.05e-06), and healthy seeds are bit-identical. At the catalog's 1000 Å production thickness the guard rejects **zero** pairs (hopg 30 keV, hopg 300 keV, silicon 100 keV); at 1e6 Å it rejects 0.233%. No golden moved.

New anchors, all in `tests/montecarlo/test_xray_dispersion.py`: `test_the_two_cycle_root_is_rejected_rather_than_returned`, `test_the_rejected_root_would_otherwise_have_passed_the_energy_window` (which pins that the 10 eV cut cannot serve as the guard), `test_a_converged_root_is_untouched_by_the_guard`, and `test_the_guard_is_inert_across_the_xray_regime`.

**Still owed on this row.** Section 2 above still presents the contraction rate as unconditional; it wants rewriting to state the domain in which it holds, and the `rederived` verdict wants re-confirming in fresh context at the same time. That belongs to whoever owns this validation, not to the branch that found the defect. `cbs-amplitude` carries the matching correction for its `v·g → 0` row.
