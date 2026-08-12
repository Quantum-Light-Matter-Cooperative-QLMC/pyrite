# X-ray refractive index for coherent phase tracking

Branch: `fix/xray-refractive-index`. Backlog: `main:TODO.md` Active item 1.

## Problem

The coherent radiation kernel assumes vacuum dispersion `k = omega`
everywhere. The crystal's bulk X-ray dielectric response — the `g=0`
susceptibility `chi_0(E)` — is never formed, so the real dispersive part of
the refractive index is missing from resonance kinematics and from the
segment-to-segment propagation phase.

The imaginary part is already accounted for: `optical_constants` computes
`beta(E)` from Henke `f2` and `absorption_length_ang` folds it into `mu(E)`
for Beer-Lambert attenuation. The new work is primarily the real part.

## Design brief (user)

The next physics upgrade is to include the **bulk X-ray dielectric response**
of the crystal through the `g=0` susceptibility `chi_0(E)`. From the same
Henke `f_1`/`f_2` data already used for `chi_g`, `U_g`, and absorption,
compute the complex refractive index

$$
n(E) = \sqrt{1 + \chi_0(E)} \approx 1 - \delta + i\beta
$$

The existing Beer-Lambert absorption already accounts for the imaginary part
through `mu(E)`; the new work is primarily to include the real dispersive part
consistently.

The refractive correction should then be propagated through the **coherent
radiation kinematics**, rather than added only as an extra phase term. In
particular, the photon wavevector should use the in-medium dispersion
relation, the PXR/CBS resonance condition should include the `n(E)`-dependent
Doppler denominator, and the coherent segment-to-segment propagation phase
should use the corresponding in-medium wavevector. Any PXR denominator that
currently assumes `k^2 = omega^2` should also be updated consistently from the
dielectric Maxwell dispersion relation rather than by inserting `n` ad hoc.

Initially, this should be implemented as an optional model, e.g.
`xray_dispersion="vacuum"` versus `"refractive"`, so the existing result
remains a reproducible baseline. Validation should compare resonance energies,
accumulated phases, and coherent spectra for representative crystal
thicknesses and photon energies, with particular attention to whether small
values of `delta` accumulate into order-unity phase shifts over micron-scale
trajectories. Interface refraction/Fresnel effects can remain a later
extension unless the observation geometry approaches grazing incidence.

## Affected code

- `src/pyrite/materials/crystal.py` — `chi_g` (:231), `U_g` (:254),
  `absorption_length_ang` (:284), `optical_constants` (:315). No `chi_0`
  exists; `optical_constants` already returns per-element `delta`/`beta` from
  Henke and is the natural base for a cell-level `chi_0`.
- `src/pyrite/montecarlo/spectrum/lines.py` — vacuum dispersion is baked in at
  the detuning/resonance core (:336-:349) and the coherent propagation phase
  (:685-:689, :1009).
- `src/pyrite/montecarlo/spectrum/coherent_stream_jit_kernel.py` — JIT/CUDA
  counterpart must stay in lockstep with the CPU core.

## Checklist

- [x] `chi_0(crystal, photon_E_eV)` in `materials/crystal.py` from Henke
      `f1`/`f2` summed over the unit cell; complex, dimensionless.
      Ledger `xray-chi-zero`, status `filtered`.
- [x] `refractive_index` consistent with the existing `optical_constants` and
      `absorption_length_ang` conventions — `-Re(chi_0)/2` and `-Im(chi_0)/2`
      reproduce `delta`/`beta` to 1e-12 rel. Ledger `xray-refractive-index`,
      status `filtered`.
- [x] `xray_dispersion` model switch plumbed to `mc_spectrum` (and to
      `mc_spectrum_solid_angle` through its `**kwargs`), default `"vacuum"`,
      bit-for-bit. Runner/campaign-config plumbing is NOT done — see remainder.
- [x] In-medium dispersion in the resonance denominator, derived from the
      Maxwell dispersion relation rather than an ad hoc `n` insertion. Covers
      the resonance, `k.g`, the detuning, and the PXR numerator's `k^2`, on both
      the batched and the per-hkl accumulation paths. Ledger
      `xray-in-medium-resonance`, status `filtered`.
- [ ] In-medium wavevector in the coherent segment propagation phase.
      Currently `refractive` + `coherent=True` raises `NotImplementedError`.
- [ ] JIT/CUDA kernels in lockstep with the CPU core.
- [x] Physics ledger rows + in-code `Validation: <id>` markers (for the
      landed slices).
- [ ] Validation: accumulated phase vs thickness, coherent spectra at
      representative thickness/energy. Resonance energy shift is done
      (`tests/montecarlo/test_xray_dispersion.py`).

## Decisions

- Fresnel/interface refraction is out of scope (non-grazing geometry).
- `chi_0` uses `Z_TABLE[el] + f'` for the forward factor rather than
  `cromer_mann_f0(el, 0)`, so it shares one normalization with
  `optical_constants` / `absorption_length_ang` exactly instead of to within
  the Cromer-Mann `f0(0) ~ Z` fit residual.
- `refractive_index` takes the square root exactly rather than linearizing.
  The in-medium wavevector is defined from `n`, and the linearized form is
  only an O(chi_0^2) approximation to it. Measured residual vs the linearized
  `1 - delta - i beta` in Si: `delta/2` and `delta` respectively (9.2e-5 and
  1.8e-4 rel at 1.5 keV).
- Sign convention `n = 1 - delta - i beta` (time factor `exp(+i omega t)`)
  is forced by the existing coherent propagation phase `exp{i[omega t - k.r]}`
  in `lines.py:685` and matches `optical_constants`.
- Only `Re n` enters the kinematics. `Im n` is the same absorption already
  carried as the Beer-Lambert `mu(E)` escape factor, so folding it in here
  would double-count it.
- The whole in-medium correction is routed through two existing quantities —
  the resonance `denom` and `n_hat.g` — plus `k_mag` for the PXR numerator's
  `k^2`. `k.v = omega (1 - denom)` survives the substitution exactly, so
  `_line_kin_core` and `_line_amp_sq_core` are unchanged; they just receive
  in-medium arguments. Under `refractive` both `denom` and `n_hat.g` stop being
  hoistable (they become `(n_seg, N_g)`), which is the cost of the model.
- The implicit resonance `omega_res = v.g / (1 - Re n(omega_res) v.n_hat)` is
  solved by fixed-point iteration from the vacuum root; the map contracts at
  rate ~`delta` ~ 1e-5, so 3 passes are far past float64 rounding.

## Remainder / next slices

- Coherent propagation phase on the in-medium wavevector. Structural note: the
  reduction kernels factor the phase as `exp(i * d_j * E)` with a per-segment
  scalar `d_j = t_abs,j - n_hat.r_j`. In medium the space part picks up an
  energy-dependent `Re n(E)`, so that factorization breaks — the phase becomes
  `exp(i(omega(E) a_j - k(E) b_j))` with per-segment `a_j = t_abs,j` and
  `b_j = n_hat.r_j` and two per-ENERGY tables. That is a signature change
  across the CPU core, the JIT kernel, and the CUDA kernel, which is why it is
  its own slice.
- Runner/campaign plumbing: `runner._lines_for_segments` derives `coherent`
  from the case dict; `xray_dispersion` should come the same way
  (`case.get("xray_dispersion", "vacuum")`), which also touches campaign config
  validation.

## Pre-existing failures on `main` (not caused by this branch)

- `pyrite-dev lint`: 2x `F841` in `montecarlo/transport.py` (:1638, :1942).
- `pyrite-dev typecheck`: 52 diagnostics, mostly unresolved optional imports.
- `tests/materials/test_crystal_lattice.py` fails to collect (`plotly` not
  installed); `test_material_catalog.py::test_packaged_profiles_have_explicit_membership`
  fails. Both verified failing with this branch's changes stashed.
- `pyrite-dev format` reformats 8 files unrelated to this task; those reverts
  are deliberate, keep them out of this branch's diff.
