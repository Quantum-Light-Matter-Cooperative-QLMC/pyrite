# Detector solid-angle integration

The line energy `E_res = ħc·(v·g)/(1 − v·n̂)` depends on the observation direction `n̂`. A real detector subtends a finite solid angle, so different parts of its face see slightly different line energies and intensities. This page describes how that finite acceptance is represented.

## Single-angle treatment

The spectrum is computed at one `n̂` (from `montecarlo.tilted_geometry`). The finite detector enters through two separate approximations:

1. **Flux.** The per-steradian intensity is multiplied by a flat solid angle: `results.store_result` records `scale = domega_sr · PER_NA`, and `Detector.score()` applies it at read time. This assumes `d²N/dE dΩ` is constant across the face. The acceptance lives on `Detector` and is lowered to the scalar case keys, so the stored one-detector record is unchanged.
2. **Line width.** An analytic, polar-only, symmetric Gaussian (`montecarlo.aperture_fwhm_eV`, from the detector polar span Δθ) is added in quadrature with the EDS resolution and applied via `convolve_detector`.

This evaluates the spectrum at a single θ_obs and treats Δθ_obs as Gaussian line broadening. It is accurate for small acceptances such as the Timepix (Δθ ≈ 1.76°, Ω ≈ 9.5×10⁻⁴ sr).

## Face integration

The flat detector face is tiled with directions `n̂_i`, and the spectrum is integrated with solid-angle weights, `Σ_i dΩ_i · spec(n̂_i)`. The weights follow

`dΩ_i = dA_i cos ψ_i / r_i²`

(inverse-square plus obliquity, with ψ_i measured from the chip normal, the central line of sight) and are rescaled so that `Σ_i dΩ_i = Ω`, conserving the detector's total solid angle.

The result is the Ω-integrated line spectrum, already multiplied by Ω. It has different units from the per-steradian single-angle spectrum, which is the form the checkpoint workflow retains (flat-Ω flux scaling and analytic aperture broadening).

### Geometry

- An `n_side × n_side` grid lies on the flat rectangular chip at distance d, facing the source, in the lab frame (`montecarlo.detector_directions`). It supplies per-cell directions and solid-angle weights.
- One grid axis spreads in the scattering plane (the polar, Δθ direction) and the other out of plane (azimuth).
- Each direction is mapped into the sample frame through the same tilt rotation as the single `n̂`.
- `n_side = 1` returns the single central direction with weight Ω, reproducing the single-angle spectrum × Ω to machine precision.

### Per-direction quantities

These are recomputed for every direction in the grid:

- the polarization pair `e_s`/`e_p`
- `denom = 1 − v·n̂` and `E_res`
- the sinc width `a_width`
- the photon kinematics `k = ω n̂`, `k + g`, and the detuning
- the `T_abs` escape branch (the sign of `n̂_z` selects the exit face)

### Behavior

- The integrated line shows the asymmetric lineshape, the `n̂`-resolved line shift, and the intensity gradient across the face. Both the flat-Ω flux and the symmetric aperture Gaussian are replaced by the single integral.
- For a wide detector (Δθ ≈ 37°) the integrated line shifts by about −50 eV and broadens from 13 to 123 eV. The integrated width is narrower than the symmetric `aperture_fwhm_eV` estimate (196 eV).
- For the Timepix (Δθ ≈ 2°) the centroid shift is about 0.02 eV.
- As Δθ → 0 the integrated width converges to the true two-dimensional width, which is generally narrower than the symmetric box-Gaussian approximation; exact agreement with `aperture_fwhm_eV` is not expected.

### Cost and conventions

- **Cost.** Work scales as `N_dir ×` the per-reflection sinc² matmul, which is the device hot loop. The GPU path runs serially in one CUDA context, so a 5×5 grid costs about 25× per line case. Directions are looped rather than broadcast along an `n̂` axis to keep memory flat.
- **Unit convention.** Because the integrated `spec` already includes Ω, the flat `domega_sr` in `results.store_result`'s `scale` and the `aperture_fwhm_eV` term in `fwhm` must be dropped for integrated records (the EDS-resolution term stays). The `scale`/`fwhm` fields are consumed by `detected_background`, `summary_table`, `line_metrics`, `best_azimuth`/`selection_score`, and the Timepix/EagleXO forward models, and bremsstrahlung (single-`n̂`, flat-Ω) shares the scale field. Integrated and single-angle records therefore need an `integrated` flag that every consumer branches on.
- **Frames.** The chip tiling is defined in the lab frame while per-direction physics is evaluated in the sample frame. The chip's in-plane roll relative to the scattering plane must be carried through the rotation, or the asymmetry is biased.
- **Bremsstrahlung.** Bremsstrahlung emission is isotropic (1/4π) and depends on `n̂` only through `T_abs`. It is kept at a single `n̂`; in the wide, grazing regime `L_esc ∝ 1/n̂_z` varies across the face and this is an additional approximation.

### Applicability

`n_side = 1` is the exact, fast default. Face integration matters for wide detectors (Δθ ≈ 12–17°), where the lineshape is visibly asymmetric and the intensity gradient across the face is significant. For the Timepix the shift and asymmetry are sub-percent and the single-angle treatment suffices.
