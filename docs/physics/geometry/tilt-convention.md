# Sample-tilt convention (Zhai et al.'s)

PyRITE reports and grids sample tilt in Zhai et al.'s convention. This note is
the canonical reference; {math}`montecarlo/geometry.py::tilted_geometry` and
`::detector_directions` implement it and should be read alongside this file.

## Definition

`tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0)` builds the
lab-frame sample normal

```{math}
normal = [sin\theta \cdot cos\phi, sin\theta \cdot sin\phi, cos\theta]      (\theta = tilt_polar_rad, \phi = tilt_azim_rad)
```

and rotates the fixed-`+z_lab` beam and the detector (at lab polar angle
`theta_obs_rad`, azimuth 0) into the sample frame accordingly.

- **Polar {math}`\theta`, positive** = the reciprocal (inverse-lattice) vector {math}`g`
  (∥ the slab normal for the (00l)-type reflections cxr currently scans) tilts
  **toward the detector**. At {math}`\theta_\mathrm{obs} = 119\deg`, {math}`\theta = +10\deg, \phi = 0`: {math}`g` has dot
  {math}`−0.326` with the detector direction, vs {math}`−0.630` at {math}`\theta = −10\deg` — i.e. {math}`+10\deg`
  sits closer to the detector, confirming the sign.
- **Azimuthal {math}`\psi`, positive** = a **CCW roll about the electron-beam {math}`+z`
  axis**, reported in {math}`[0\deg, 180\deg]` (Zhai et al.'s SI reports values above 90\deg).
  {math}`\phi = 0` places the tilt in the scattering {math}`x–z` plane (the plane containing
  the beam and the detector). As {math}`\phi` increases from 0 the normal's azimuth
  sweeps {math}`+x \rightarrow +y`, matching Zhai et al.'s positive-\phi sense.

This matches Zhai et al.'s SI Fig. 1 spherical {math}`(\theta, \phi)` parametrization
directly — cxr's formula is not a relabeling or a sign flip of some other
convention, it *is* Zhai et al.'s parametrization, evaluated at their angles.

## Default zero-scattering scalars are invariant to the polar sign

For the default beam-aligned, {math}`g \| normal`, {math}`\psi = 0` geometry, both scalar
products in the production dispersion relation
`\omega = v \cdot g / (1 − v \cdot n̂)` are **even in \theta**. The denominator is invariant under
`\theta → −\theta`, while {math}`v_0\cdot g = \beta|g| \cos(\theta)` has the same value at equal and opposite
tilts. Flipping the polar sign therefore leaves the zero-scattering line
energy unchanged, and default {math}`v_0 \cdot g` cannot explain an intensity asymmetry.

The full simulated intensity can nevertheless differ between opposite tilts
through downstream direction-sensitive transport, escape geometry,
polarization/amplitude effects, or a reciprocal vector that is not aligned
with the sample normal.

Empirical check (WSe₂, 55 nm, {math}`\theta_\mathrm{obs} = 119\deg`, identical transport seed,
{math}`+10\deg` vs {math}`−10\deg`):

- **Line energy: unchanged** — 981.5 eV both ways.
- **Peak intensity: differs 2×** —
  {math}`I(+10\deg)/I(−10\deg) \approx 0.505`, so the positive-tilt peak is roughly half the
  negative-tilt peak in this spot check.
- **Integrated-flux ratio:** {math}`F(+10\deg)/F(−10\deg) \approx 0.65`.

These are full-model observations for this configuration, not consequences of
the default zero-scattering {math}`v_0 \cdot g` scalar or a universal monotonic tilt rule.

## The {math}`n`/{math}`g` split (written, currently unused)

`_orientation_R` and {math}`mc_spectrum` carry an optional parameter that lets the
**reciprocal vector {math}`g`** tilt independently of the **physical slab normal
`n̂`**. Its default (`None`) is a strict no-op: {math}`g \| normal`, today's — and
every current grid's — behavior, bit-for-bit.

The formula above (`\theta` = angle between {math}`g` and the detector direction, via
the slab normal) is exact only when {math}`g \| \hat{n}`, which holds for the (00l)-type
symmetric reflections cxr currently scans (TMD basal plane, HOPG c-axis,
h-BN). It stops being exact for an **asymmetric reflection** — a crystal cut
or miscut where the diffracting planes are not parallel to the physical
surface, so {math}`g` and {math}`\hat{n}` differ by a fixed offset. The {math}`n`/`g` split applies an extra
rotation to {math}`g` only, leaving {math}`\hat{n}` (and therefore transport and beam/detector geometry) fixed. It is
not wired into any grid or study today — introducing an asymmetric-reflection
material is the trigger to use it.
