# Sample-tilt convention

PyRITE reports and grids sample tilt in Zhai et al.'s convention. This note is the canonical reference; `montecarlo/geometry.py::tilted_geometry` and `::detector_directions` implement it and should be read alongside this file.

## Definition

The defining references for our geometry are the beam axis, defined to be $z_\mathrm{beam}=z_\mathrm{lab}$, and the detector observation direction unit vector, $\hat{n}$. Our lab-frame origin is defined as the point where $z_\mathrm{lab}$ intersects with the crystal surface. This origin, coupled with the physical location of the detector, then fixes the direction of $\hat{n}$:

```{math}
\hat{n} = [\sin\theta_\mathrm{obs},\, 0,\, \cos\theta_\mathrm{obs}]
```

Where $\theta_\mathrm{obs}$ is defined as the polar angle between $\hat{n}$ and $+z_\mathrm{lab}$. The equation for $\hat{n}$ above defines $x_\mathrm{lab}$: $z_\mathrm{lab}$ forms a plane with $\hat{n}$, where $+x_\mathrm{lab}$ is defined to be the direction transverse to the beam which moves towards the detector. $y_\mathrm{lab}$ is then defined according to the right-hand rule.

`tilted_geometry(theta_obs_rad, tilt_polar_rad, tilt_azim_rad=0.0)` builds the lab-frame sample normal:

```{math}
\frac{\vec{g}}{|g|} = \left[\sin\theta_\mathrm{tilt} \cos\phi_\mathrm{tilt},\, \sin\theta_\mathrm{tilt} \sin\phi_\mathrm{tilt},\, \cos\theta_\mathrm{tilt} \right]
```

(where {math}`\theta_\mathrm{tilt}` = `tilt_polar_rad`, {math}`\phi_\mathrm{tilt}` = `tilt_azim_rad`) and rotates the fixed {math}`z_\mathrm{lab}` beam and the detector (at lab polar angle {math}`\theta_\mathrm{obs}` = `theta_obs_rad`, {math}`\phi_\mathrm{obs} = 0^\circ`) into the sample frame accordingly.

- **Polar {math}`\theta_\mathrm{tilt}`, positive** = the reciprocal (inverse-lattice) vector {math}`\vec{g}` (parallel to the slab normal {math}`\hat{n}` for the `(00l)`-type reflections PyRITE currently scans) tilts **toward the detector**.
  - At {math}`\theta_\mathrm{obs} = 119^\circ`, {math}`\theta_\mathrm{tilt} = +10^\circ, \, \phi_\mathrm{tilt} = 0`, we find {math}`\vec{g} \cdot \hat{n} = −0.326`
  - At {math}`\theta_\mathrm{tilt} = −10^\circ, \, \phi_\mathrm{tilt} = 0`, we find {math}`\vec{g} \cdot \hat{n} = −0.630`, i.e. {math}`+10^\circ` sits closer to the detector, confirming the sign.
- **Azimuthal {math}`\phi_\mathrm{tilt}`, positive** = a **CCW roll about the electron-beam {math}`+z` axis**, reported in {math}`[0^\circ, 180^\circ]` (Zhai et al.'s SI reports values above {math}`90^\circ`). {math}`\phi_\mathrm{tilt} = 0` places the tilt in the scattering {math}`x–z` plane (the plane containing the beam and the detector). As {math}`\phi_\mathrm{tilt}` increases from 0 the normal's azimuth sweeps {math}`+x \rightarrow +y`, matching Zhai et al.'s positive-$\phi_\mathrm{tilt}$ sense.

This matches Zhai et al.'s SI Fig. 1 spherical {math}`(\theta_\mathrm{tilt}, \phi_\mathrm{tilt})` parametrization directly — PyRITE's formula is not a relabeling or a sign flip of some other convention, it *is* Zhai et al.'s parametrization, evaluated at their angles.

## Default zero-scattering scalars are invariant to the polar sign

For the default beam-aligned, {math}`\vec{g} \: \| \: \hat{n}`, {math}`\psi = 0` geometry, both scalar products in the production dispersion relation {math}`\omega = \vec{v} \cdot \vec{g} \, / \, (1 − \vec{v} \cdot \hat{n})` are **even in {math}`\theta_\mathrm{tilt}`**. The denominator is invariant under {math}`\theta_\mathrm{tilt} \rightarrow −\theta_\mathrm{tilt}`, while {math}`\vec{v_0} \cdot \vec{g} = \beta \, |g| \cos(\theta_\mathrm{tilt})` has the same value at equal and opposite tilts. Flipping the polar sign therefore leaves the zero-scattering line energy unchanged, and default {math}`\vec{v_0} \cdot \vec{g}` cannot explain an intensity asymmetry.

The full simulated intensity can nevertheless differ between opposite tilts through downstream direction-sensitive transport, escape geometry, polarization/amplitude effects, or a reciprocal vector that is not aligned with the sample normal.

Empirical check (WSe₂, 55 nm, {math}`\theta_\mathrm{obs} = 119 ^\circ`, identical transport seed, {math}`+10^\circ` vs {math}`−10^\circ`):

- **Line energy: unchanged** — 981.5 eV both ways.
- **Peak intensity: differs 2×** — {math}`I(+10^\circ) / I(−10^\circ) \approx 0.505`, so the positive-tilt peak is roughly half the negative-tilt peak in this spot check.
- **Integrated-flux ratio:** {math}`F(+10^\circ) / F(−10^\circ) \approx 0.65`.

These are full-model observations for this configuration, not consequences of the default zero-scattering {math}`\vec{v_0} \cdot \vec{g}` scalar or a universal monotonic tilt rule.

## The {math}`\hat{n} / \vec{g}` split (written, currently unused)

`_orientation_R` and `mc_spectrum` carry an optional parameter that lets the **reciprocal vector {math}`\vec{g}`** tilt independently of the **physical slab normal {math}`\hat{n}`**. Its default (`None`) is a strict no-op: {math}`\vec{v_0} \cdot \vec{g} \: \| \: \hat{n}`, today's behavior, bit-for-bit.

The formula above ({math}`\theta` = angle between {math}`\vec{g}` and the detector direction, via the slab normal) is exact only when {math}`\vec{g} \: \| \: \hat{n}`, which holds for the (00l)-type symmetric reflections PyRITE currently scans (TMD basal plane, HOPG c-axis, h-BN). It stops being exact for an **asymmetric reflection** — a crystal cut or miscut where the diffracting planes are not parallel to the physical surface, so {math}`\vec{g}` and {math}`\hat{n}` differ by a fixed offset. The {math}`\hat{n} \, / \, \vec{g}` split applies an extra rotation to {math}`\vec{g}` only, leaving {math}`\hat{n}` (and therefore transport and beam/detector geometry) fixed. It is not wired into any grid or study today — introducing an asymmetric-reflection material is the trigger to use it.
