# Sample-tilt convention (Zhai's)

PyRITE reports and grids sample tilt in Zhai et al.'s convention. This note is
the canonical reference; `montecarlo/geometry.py::tilted_geometry` and
`::detector_directions` implement it and should be read alongside this file.

## Definition

`tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0)` builds the
lab-frame sample normal

```
normal = [sinθ·cosφ, sinθ·sinφ, cosθ]      (θ = tilt_polar_rad, φ = tilt_azim_rad)
```

and rotates the fixed-`+z_lab` beam and the detector (at lab polar angle
`theta_obs_rad`, azimuth 0) into the sample frame accordingly.

- **Polar `θ`, positive** = the reciprocal (inverse-lattice) vector `g`
  (∥ the slab normal for the (00l)-type reflections cxr currently scans) tilts
  **toward the detector**. At `θ_obs = 119°`, `θ = +10°, φ = 0`: `g` has dot
  `−0.326` with the detector direction, vs `−0.630` at `θ = −10°` — i.e. `+10°`
  sits closer to the detector, confirming the sign.
- **Azimuthal `φ`, positive** = a **CCW roll about the electron-beam `+z`
  axis**, reported in **`[0°, 180°]`** (Zhai's SI reports values above 90°).
  `φ = 0` places the tilt in the scattering `x–z` plane (the plane containing
  the beam and the detector). As `φ` increases from 0 the normal's azimuth
  sweeps `+x → +y`, matching Zhai's positive-φ sense.

This matches Zhai et al.'s SI Fig. 1 spherical `(θ, φ)` parametrization
directly — cxr's formula is not a relabeling or a sign flip of some other
convention, it *is* Zhai's parametrization, evaluated at Zhai's angles.

## This was a grid change, not a math change

`tilt_deg` has always flowed into `tilted_geometry` with **no negation** at
any call site. The spherical formula above was already Zhai's positive-θ
convention; what changed (2026-07-11) is that the per-material scan grids
(now in `data/materials.toml`, consumed through `config.py`, `sweep.py`, and `scan.py`, plus
`src/cxr_mc/apps/anchor_figures.py`) previously populated only the **negative** half
of the polar range (`−85…0`, etc.) and a negative azimuth span (`−80…0`).
Those grids simulated the mirror configuration — reciprocal vector tilted
**away** from the detector — at a different intensity. In the reproduced
WSe₂ spot check below, the old negative mirror has the higher peak. This is
not a universal claim that either tilt sign always increases or decreases the
full-model intensity. The rotation math in `tilted_geometry` did not change;
only the angles fed into it did.

## Default zero-scattering scalars are invariant to the polar sign

For the default beam-aligned, `g ∥ normal`, `φ = 0` geometry, both scalar
products in the production dispersion relation
`ω = v·g / (1 − v·n̂)` are **even in θ**. The denominator is invariant under
`θ → −θ`, while `v0·g = β|g| cos(θ)` has the same value at equal and opposite
tilts. Flipping the polar sign therefore leaves the zero-scattering line
energy unchanged, and default `v0·g` cannot explain an intensity asymmetry.

The full simulated intensity can nevertheless differ between opposite tilts
through downstream direction-sensitive transport, escape geometry,
polarization/amplitude effects, or a reciprocal vector that is not aligned
with the sample normal.

Empirical check (WSe₂, 55 nm, `θ_obs = 119°`, identical transport seed,
`+10°` vs `−10°`):

- **Line energy: unchanged** — 981.5 eV both ways.
- **Peak intensity: differs 2×** —
  `I(+10°)/I(−10°) ≈ 0.505`, so the positive-tilt peak is roughly half the
  negative-tilt peak in this spot check.
- **Integrated-flux ratio:** `F(+10°)/F(−10°) ≈ 0.65`.

These are full-model observations for this configuration, not consequences of
the default zero-scattering `v0·g` scalar or a universal monotonic tilt rule.

Practical consequence: line-energy validations computed under the old
(negative-tilt) grids remain valid — the line positions did not move.
Intensity- and enhancement-dependent validations (peak height, integrated
flux, bulk-vs-film enhancement ratios) must be re-run against the corrected
(positive-tilt) grids before being trusted; see
`docs/validation/physics-validation-ledger.md` and
`docs/validation/literature-ref/zhai-supplementary.md`.

## The `n`/`g` split hook (plumbed, unused)

`_orientation_R` and `mc_spectrum` carry an optional parameter that lets the
**reciprocal vector `g`** tilt independently of the **physical slab normal
`n̂`**. Its default (`None`) is a strict no-op: `g ∥ normal`, today's — and
every current grid's — behavior, bit-for-bit.

The formula above (`θ` = angle between `g` and the detector direction, via
the slab normal) is exact only when `g ∥ n̂`, which holds for the (00l)-type
symmetric reflections cxr currently scans (TMD basal plane, HOPG c-axis,
h-BN). It stops being exact for an **asymmetric reflection** — a crystal cut
or miscut where the diffracting planes are not parallel to the physical
surface, so `g` and `n̂` differ by a fixed offset. The `n`/`g` split hook is
the escape hatch for that case: it applies an extra rotation to `g` only,
leaving `n̂` (and therefore transport and beam/detector geometry) fixed. It is
not wired into any grid or study today — introducing an asymmetric-reflection
material is the trigger to use it.
