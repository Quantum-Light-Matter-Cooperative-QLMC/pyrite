# Transport geometry and boundaries

Where an electron enters, what volume confines it, and how a flight is cut short
by a surface or an interface. The physical models this page constrains —
[elastic scattering](../beam-transport/elastic-scattering.md) and
[stopping power](../beam-transport/stopping-power.md) — do not change with geometry; what changes
is which material is active and how far a flight is allowed to run.

## Frames and the entrance plane

The transport volume is the sample frame: the entrance face lies at $z = 0$ and
depth runs along $+z$. The nominal beam direction is $+z$ at normal incidence;
`beam_dir` supplies any other direction, and `tilted_geometry` builds it from the
[tilt convention](tilt-convention.md).

Every beam quantity in [Beam phase space](../beam-transport/beam-phase-space.md) refers to the
crystal entrance face. There is no beamline upstream of it.

### Beam frame and slope injection

Per-electron slopes $(x', y')$ from the Courant–Snyder policy are $dx/dz$ and
$dy/dz$ **about the beam axis**, not about the lab $z$. `beam_frame_basis` builds
an orthonormal frame whose third column is `beam_dir` exactly and whose
transverse columns coincide with lab $x$ and $y$ when the beam is on axis, so the
on-axis case reduces to the identity and slopes compose with tilt without a
second convention.

### Projection onto a tilted face

A transverse entry point is drawn in the lab plane perpendicular to the beam,
then intersected with the tilted entrance plane. The footprint stretches by
$1/\cos(\text{tilt}_{\rm polar})$ along the tilt azimuth:

```{math}
:label: eq-geometry-footprint-stretch

A_{\rm face} = \frac{A_{\rm beam}}{\cos(\text{tilt}_{\rm polar})}.
```

At normal incidence the projection is the identity, bit-for-bit. At grazing
incidence it is the mechanism by which a beam overruns a finite sample and the
lost electrons register as misses rather than silently illuminating the crystal.

## Confining volume

```{list-table} Transport volumes.
:name: tbl-geometry-volumes
:header-rows: 1

* - Configuration
  - Volume
  - Exit classification
* - `crystal_width_mm` / `crystal_height_mm` unset
  - laterally infinite slab $0 \le z \le$ thickness
  - backscattered, transmitted
* - both set
  - rectangular prism centered on the beam origin
  - backscattered, transmitted, side-exited, missed
```

Both transverse dimensions must be supplied together, or neither. With a finite
footprint, each free flight is capped at the smallest positive ray–face
intersection $\mathbf p + s\,\mathbf d$, which assumes an axis-aligned rectangular
footprint.

An electron whose projected entry point falls outside the footprint is counted in
`n_missed` and produces no segment — but it stays in $N_e$. This is deliberate:
every yield in the package is normalized per *incident* electron, so overrunning
the sample reduces the yield rather than being renormalized away. Grazing-incidence
overlap loss is a physical effect here, not a bookkeeping artifact.

The laterally infinite limit is not merely a large prism. With no footprint, no
downstream model reads the transverse position — not the scattering, not the
stopping, not the escape path — so a finite beam spot rigidly translates each
trajectory and cannot change the spectrum. Once a footprint exists, transverse
position feeds missed incidence, side escape, and the six-face escape
attenuation, and beam size becomes a spectral parameter.

## Layer stacks

A stack is a contiguous list of `(z_top, z_bot, composition)` entries, entrance
first, whose deepest `z_bot` supersedes `thickness_ang`. The active layer is
determined by the electron's current depth, and it sets the free path, the
stopping power, and the element drawn at a collision.

A flight that would cross an internal boundary is **truncated at the boundary**:
the segment ends there, no collision is recorded, no deflection is applied, and
the electron continues in the neighboring material with a fresh flight. The
position is nudged past the interface by a fixed epsilon so the layer lookup
cannot re-select the layer just left.

Truncating and redrawing is exact rather than approximate because the collision
distance is exponentially distributed and therefore memoryless: the surviving
optical depth after any truncation has the same distribution as a fresh draw.
The same argument licenses re-entry after a vacuum excursion.

Because boundary crossings do not scatter, a substrate feeds path length back
into the film through ordinary backscatter, which is the effect a thin-film
approximation loses. See
[Multilayer materials](../materials/multilayer-materials.md).

## Blazed grooves

`groove` replaces the flat entrance face with a periodic blazed sawtooth, in a
restricted geometry ($\theta_{\rm obs} = 90°$, tilt azimuth $180°$,
$0 < \text{tilt}_{\rm polar} < 90°$). Two plane families define the profile:

```{math}
:label: eq-geometry-groove-facets

\hat{\mathbf n}\cdot\mathbf r = k\,d\cos t_p
\quad\text{(working facet)},
\qquad
\hat{\mathbf b}\cdot\mathbf r = k\,d\sin t_p
\quad\text{(relief facet)},
```

with $d$ the groove spacing and $t_p$ the polar tilt. Closing the unit cell fixes
the depth $h = d\sin t_p\cos t_p$, and a candidate crossing is physical only when
$s > 0$ and its depth lies in $[0, h]$.

The working facet is perpendicular to the observation direction and the relief
facet is parallel to the exit rays, so the profile shortens photon escape paths
without shadowing them — the point of the geometry.

Transport-side consequences:

- a material-to-vacuum crossing truncates the radiating segment at the facet;
- a vacuum leg does not scatter, does not stop, and does not radiate, but it
  **does** advance the transport clock by $L_{\rm vacuum}/\beta$, so timing and
  coherent phase stay correct across the gap;
- re-entry resumes material transport with a fresh free-path draw (memorylessness
  again);
- a ray with no later re-entry is a permanent entrance-face exit;
- vacuum legs are returned as separate `vacuum_*` arrays and never enter the
  material segment sum;
- exit/re-entry pairs consume a separate per-electron event budget rather than
  `max_steps`; exhausting it raises.

Photons are transported along $\hat{\mathbf n}$ by incoherent Beer–Lambert
attenuation, consistent with the line kernel; there is no wave optics on the
groove profile. The profile is invariant along $y$, so only the $x$–$z$ section
of the sawtooth enters the crossing test.

## Termination

An electron's history ends in exactly one of:

```{list-table} Termination outcomes and their tallies.
:name: tbl-geometry-termination
:header-rows: 1

* - Outcome
  - Tally
* - exits through the entrance face
  - `n_backscattered`
* - exits through the rear face
  - `n_transmitted`
* - exits a side face (finite footprint only)
  - `n_side_exited`
* - falls below `E_cut_keV`
  - `n_cutoff_stopped` (alias `n_stopped`)
* - never entered the footprint
  - `n_missed`
* - exhausts `max_steps`
  - `n_step_limited`
```

`n_step_limited` is a failure mode, not a physical channel: an incomplete history
raises rather than returning truncated arrays, so a run never silently reports
partially transported electrons. The default budget is 20000 steps per electron.

## Assumptions

- Interfaces are sharp, static, and planar; no roughness, no interdiffusion, no
  thermal motion of the boundary.
- The footprint is an axis-aligned rectangular prism; other sample shapes are not
  represented.
- Vacuum is truly empty — no residual-gas scattering on the groove excursions.
- Geometry is fixed for the whole run; nothing moves between electrons.
- Grooved runs are restricted to the observation geometry above, and the groove
  profile is invariant along $y$.

## Validation

`Validation: finite-beam-size`, `finite-transverse-crystal`,
`grazing-beam-projection`, `multilayer-stack`, and `blazed-groove-geometry`. See
the [physics validation ledger](../../validation/physics-validation-ledger.md).
