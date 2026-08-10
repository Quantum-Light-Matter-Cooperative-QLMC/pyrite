# `alexs-charge-diffusion` — independent verification

- **Code**: `src/pyrite/detectors/grating.py::charge_cloud_sigma_um`
- **Ledger claim**: drift-diffusion charge-cloud spread,
  `σ² = 2(kT/q)·t·(t−min(L_abs(E),t))/v_dep`
- **Cited source**: Einstein relation (`D = μkT/q`) + drift-diffusion,
  "standard back-illuminated-CCD treatment (e.g. Janesick, *Scientific
  Charge-Coupled Devices*, SPIE 2001, Ch. 4)"
- **Signature**: `charge_cloud_sigma_um(E_eV, active_um=None, v_dep=None,
  temp_c=None) -> sigma_um` (per-photon lateral RMS charge-cloud spread at
  the collecting electrodes, in micrometers)

## 1. Independent re-derivation (before reading the implementation body)

Setup: back-illuminated, fully depleted Si sensor, active thickness `t`,
uniform bias `v_dep` across it (parallel-plate / fully-depleted
approximation), so the drift field is uniform:

    E_field = v_dep / t

A photon absorbed at depth `z` from the back (entrance) surface creates a
charge cloud that must drift a distance `d = t − z` to reach the front
(collecting) electrodes.

**Drift.** For a carrier of mobility `μ`:

    v_d = μ * E_field = μ * v_dep / t
    t_drift = d / v_d = d * t / (μ * v_dep)

**Diffusion.** Einstein relation: `D = μ kT/q`. During transit time
`t_drift`, 1-D thermal diffusion gives mean-square lateral spread (per
transverse axis, standard 1-D random walk, `⟨x²⟩ = 2 D τ`):

    σ² = 2 D t_drift = 2 (μ kT/q) * d*t/(μ v_dep) = 2 (kT/q) * t * d / v_dep

**μ cancels exactly** — this is not a coincidence, it is a structural
consequence of Einstein's relation: `D ∝ μ` and `v_d ∝ μ` (Ohmic drift,
field-independent mobility), so `t_drift ∝ 1/μ`, and the product `D·t_drift`
is `μ`-independent. Verified numerically: plugging in electron mobility
μ=1350 cm²/(V·s) vs. an arbitrary μ=450 cm²/(V·s) (factor-of-3 difference)
into the full physical chain (E_field → v_d → t_drift → D → σ) gives
*identical* σ_um = 0.90915 µm in both cases (t=30 µm, v_dep=40 V, T=−60 °C,
z→0 soft-photon limit). This confirms the cancellation is exact, not an
approximation.

Substituting `d = t − z`, `z ≈ min(L_abs(E), t)` (the single-number
stand-in for the absorption depth, exact only as `L_abs≪t`):

    σ² = 2 (kT/q) · t · (t − min(L_abs(E), t)) / v_dep

This is **exactly** the code's formula.

## 2. Units check

- `kT_over_q = k_B[eV/K] * T[K]` → units eV/(elementary charge) = **Volts**
  (1 eV per electron charge = 1 V, numerically k_B = 8.617333e-5 eV/K is
  used directly as V/K). Code comment confirms this convention.
- `t_um`, `drift_um` in µm, `v_dep` in V.
- `σ_um = sqrt(V · µm · µm / V) = µm`. **Consistent.**

## 3. Limiting cases

**Soft-photon limit** (`L_abs → 0`, i.e. `E→0`, or `t ≫ L_abs`):
`z→0` ⇒ `drift → t` ⇒ `σ → t·sqrt(2 kT/(q v_dep))`, the *maximum* blur —
matches the independently-derived "absorbed at the entrance surface, full
thickness to drift" case above. Numerically reproduced: `t=1000 µm,
v_dep=40 V, T=−60 °C` gives σ_soft matching `t·sqrt(2·kT/q/v_dep)` to the
test's `rel=1e-3` tolerance (test `test_charge_cloud_sigma_soft_photon_is_maximum_blur`
passes).

**Hard-photon limit** (`L_abs ≥ t`): `z_um = min(L_abs,t) → t` (capped) ⇒
`drift → 0` ⇒ `σ → 0`. Physically sound: absorption right at the front
electrodes needs no drift. Caveat noted in the ledger/docstring and
reproduced here: in this same regime QE is falling (per `alexs-qe-absorption`),
so a zero-drift detection event is real but increasingly rare — not a
contradiction, just a selection effect on the *conditional* distribution of
detected events. This is a physically-correct caveat, not a bug.

**Monotonicity**: since `min(L_abs(E),t)` is non-decreasing in `L_abs` and
`L_abs(E)` is (over the relevant band, away from absorption edges)
decreasing with `E`, `drift_um = t − min(L_abs,t)` is non-increasing in `E`,
so `σ(E)` is monotonically non-increasing with energy until it saturates at
0 once `L_abs ≥ t`. Matches
`test_charge_cloud_sigma_decreases_with_energy_over_Si_absorption_band`
(200 eV > 4000 eV > 8000 eV≈0, with L_abs(8 keV)=68.7 µm > t=30 µm).

## 4. "Mobility cancels" — coincidence or structural?

Confirmed **structural, not a coincidence**: it follows purely from
`D = μkT/q` (Einstein) combined with Ohmic (field-independent-mobility)
drift `v_d = μE`. Any consistent single-carrier drift-diffusion treatment
with these two ingredients will cancel `μ`, *regardless of which carrier's
mobility is used* — so the code's failure to specify electron vs. hole
mobility is actually irrelevant to the result, not an omission. Two
caveats worth flagging (neither invalidates the result, both worth a ledger
note):

- **Ambipolar vs. single-carrier**: the derivation implicitly assumes
  single-carrier (minority-carrier) drift in an already-separated field,
  not full ambipolar diffusion (which couples electron and hole densities
  when both species' concentrations are comparable and mutually
  interacting via space-charge). For a single X-ray photon (hundreds to a
  few thousand e-h pairs, per `W_Si`), the injected carrier density is far
  below what's needed for ambipolar coupling to matter — the standard
  literature treatment (Janesick; also Holland 1990, Groom 2006) is
  single-carrier drift-diffusion in the applied field, matching this
  derivation. Fine as-is; worth a one-line caveat.
- **Field-independent mobility (no velocity saturation)**: at the modest
  fields typical of these devices (v_dep/t ~ 40 V / 30 µm ≈ 1.3×10⁴ V/cm),
  this is close to where Si electron drift velocity begins departing from
  strict Ohmic (μ constant) behavior toward saturation (~10⁴–10⁵ V/cm
  regime). Since μ cancels in the final formula, this doesn't change the
  algebra, but it's the underlying reason "any μ" is licensed — if
  mobility were strongly field-dependent, `D=μkT/q` (Einstein) and
  `v_d=μE` would need field-dependent μ substituted at the *same* field,
  and would still cancel (both use the same local μ(E_field)), so the
  cancellation actually survives even a field-dependent-but-shared mobility
  model. Not a discrepancy.

## 5. Test suite

`UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/detectors/test_grating.py -k charge_cloud -v`
→ **4 passed**:
`test_charge_cloud_sigma_soft_photon_is_maximum_blur`,
`test_charge_cloud_sigma_zero_when_absorbed_at_front`,
`test_charge_cloud_sigma_positive_and_bounded_by_soft_limit`,
`test_charge_cloud_sigma_decreases_with_energy_over_Si_absorption_band`.
Each is numerically independent of the code's internals (reconstructs
`kT_over_q` and `max_sigma` from first principles in the test itself) and
correctly exercises the limiting cases and monotonicity derived above.

## 6. Citation specificity

The formula (Einstein relation + uniform-field drift-diffusion, mobility
cancellation) is **generic semiconductor-detector physics**, not something
that requires page-level verification against Janesick's book — it can be
derived from two textbook relations (`D=μkT/q`, `v=μE`) in a few lines, as
above. Citing "Janesick, *Scientific Charge-Coupled Devices*, Ch. 4" as
*a* place this treatment appears is reasonable (Ch. 4 does cover CCD charge
collection/diffusion physics), but the docstring/ledrow should not imply a
specific verified equation number from that book was checked — the author
does not have page-level access. Suggest softening the ledger source text
from implying a specific verified citation to something like: "Einstein
relation + drift-diffusion (standard semiconductor-detector treatment,
e.g. as covered in Janesick Ch. 4, not page-checked)" — this is a wording
nuance, not a physics problem, and the current docstring already hedges
with "e.g."

## Verdict

**Re-derivation matches the code exactly**, term-for-term, including the
non-obvious mobility cancellation (independently confirmed numerically with
two different mobility values). Units consistent. Both limiting cases
(soft-photon maximum blur, hard-photon zero blur) reproduced independently
and match the tests. No discrepancy found.

Caveats for the ledger notes (not blockers):
1. Single-carrier (not ambipolar) drift-diffusion assumption — fine for
   single-photon e-h pair counts, could be noted explicitly.
2. `z(E) ≈ min(L_abs(E), t)` remains a single-number stand-in for the true
   depth-resolved exponential absorption profile (already noted in the
   ledger/docstring).
3. Citation to Janesick Ch.4 should be understood as "standard treatment,
   not page-verified" rather than a checked equation reference.
