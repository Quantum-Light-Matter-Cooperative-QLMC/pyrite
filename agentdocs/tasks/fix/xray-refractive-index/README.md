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

- [ ] `chi_0(crystal, photon_E_eV)` in `materials/crystal.py` from Henke
      `f1`/`f2` summed over the unit cell; complex, dimensionless.
- [ ] `refractive_index` / `delta`+`beta` accessor consistent with the
      existing `optical_constants` and `absorption_length_ang` conventions
      (cross-check: `mu = 2 k beta` must reproduce today's `mu`).
- [ ] `xray_dispersion` model switch plumbed to the spectrum entry points,
      default `"vacuum"` so existing goldens are unchanged.
- [ ] In-medium dispersion in the resonance denominator, derived from the
      Maxwell dispersion relation rather than an ad hoc `n` insertion.
- [ ] In-medium wavevector in the coherent segment propagation phase.
- [ ] JIT/CUDA kernels in lockstep with the CPU core.
- [ ] Physics ledger rows + in-code `Validation: <id>` markers.
- [ ] Validation: resonance energy shift, accumulated phase vs thickness,
      coherent spectra at representative thickness/energy.

## Decisions

- Fresnel/interface refraction is out of scope (non-grazing geometry).
