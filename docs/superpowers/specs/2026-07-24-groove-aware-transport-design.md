# Groove-Aware Electron and X-Ray Transport

**Date:** 2026-07-24  
**Status:** implemented; fresh-context verification pending  
**Supersedes:** electron-transport and bremsstrahlung limitations documented in
`docs/superpowers/plans/2026-07-23-blazed-groove-geometry.md`

## Goal

Treat the complete periodic sawtooth profile as a material/vacuum boundary
throughout transport. Electrons may leave one facet, cross a groove in vacuum,
and re-enter through a later facet. Only material path contributes scattering,
stopping, and radiation. Coherent and bremsstrahlung photons use the grooved
escape boundary consistently.

`groove=None` remains bit-for-bit identical to flat-slab behavior.

## Original Defect

Before this correction, `simulate_trajectories` applied `entry_points()` once, then tested every
later electron exit against flat `z=0`. Segments therefore continue through
the cut-out groove volume as radiating crystal material. Re-entry cannot occur.

Coherent-line attenuation already used `escape_distance_ang()` correctly
for the supported observation direction. Bremsstrahlung attenuation used
the flat/slab or rectangular-prism escape distance and therefore over-attenuates
grooved cases.

## Geometry and Material Predicate

Sample coordinates retain nominal entrance `z=0`, depth `+z`, and back face
`z=thickness`. For period `Lambda`, polar tilt `tp`, groove depth

```text
h = Lambda sin(tp) cos(tp),
u = mod(x, Lambda),
x_valley = h tan(tp),
```

the periodic surface is

```text
z_surface(x) =
    u / tan(tp),                    0 <= u <= x_valley
    (Lambda - u) tan(tp),           x_valley < u < Lambda.
```

A point belongs to crystal material exactly when

```text
z_surface(x) <= z <= thickness
```

and, for a finite footprint, its transverse coordinates are inside the
rectangular bounds.

The two surface facet families remain

```text
working: n . r = k Lambda cos(tp)
relief:  b . r = k Lambda sin(tp)
```

with physical facet band `0 <= z <= h`.

## Exact Surface Events

`montecarlo/groove.py` owns pure geometry helpers:

- `surface_depth_ang(x, spec)` for surface-depth evaluation;
- `in_material(position, thickness_ang, spec, width_ang=None,
  height_ang=None)` for material membership;
- `first_surface_event(position, direction, spec, transition=None)` for the
  first forward material-to-vacuum or vacuum-to-material crossing and later
  re-entry search.

For each facet family, substitute ray `r(s) = p + s d` into its plane equation.
This gives one linear candidate `s` for each periodic plane. Candidates are
accepted only when `s > epsilon`, their intersection depth lies in `[0, h]`,
and a two-sided predicate check confirms the requested transition. The nearest
accepted candidate is the exact event. Period indices come from adjacent
floor/ceiling values of the transformed ray coordinate; no spatial ray
marching or step-size tolerance enters production physics.

Degenerate tangency does not change material state and is skipped. At exact
apex/valley ties, deterministic facet ordering plus the two-sided predicate
prevents zero-length event loops. Geometry epsilon scales with groove period
and remains many orders below physical transport lengths.

## Electron State Machine

Existing material transport remains vectorized and unchanged when no groove is
present. Grooved electrons add explicit boundary events:

1. Sample elastic free path in current material.
2. Compare collision, layer, footprint, back-face, and sawtooth event distances.
3. If collision comes first, record normal radiating material segment, apply
   continuous stopping, and scatter.
4. If sawtooth exit comes first, truncate and record only material portion.
   Apply stopping and clock advance for that material length; do not scatter at
   boundary.
5. Search same ray for next vacuum-to-material facet event.
   - If found, advance through vacuum with unchanged energy and direction.
     Advance electron clock by `L_vacuum / beta`. Resume material transport.
   - If absent, count permanent entrance-face/backscattered exit and terminate.

Resampling elastic free path after re-entry is statistically exact because
exponential collision distance is memoryless. Vacuum distance does not consume
material optical depth.

Material segments remain sole radiation source arrays. Vacuum legs use separate
diagnostic arrays so spectrum kernels cannot consume them accidentally:

```text
vacuum_start_ang   (V, 3)
vacuum_end_ang     (V, 3)
vacuum_E_keV       (V,)
vacuum_t_ang       (V,)
vacuum_elec_id     (V,)
```

Repeated exit/re-entry is supported until collision, permanent exit, energy
cutoff, or a separate bounded groove-event limit. Re-entry does not consume
the material `max_steps` budget or increment exit counters. Exhausting the
groove-event limit raises rather than silently classifying a live electron as
stopped.

## Trajectory Plots

2D and 3D plots render material segments with existing energy styling.
Stored vacuum legs render faintly using the same visual treatment as normal
electron exits. This preserves a continuous physical trajectory while making
clear that groove-gap travel caused no scattering, stopping, or emission.

## X-Ray Escape

For coherent radiation, supported geometry fixes observation direction

```text
n = (cos(tp), 0, -sin(tp)).
```

This ray crosses a working facet normally and is parallel to relief facets.
After the first outward crossing it cannot re-enter crystal. Existing
`escape_distance_ang()` therefore already returns complete material path, not
merely the first interval.

`mc_brem_spectrum` accepts the same `groove=` argument and uses
`escape_distance_ang()` for its fixed far-field `n_hat`. It enforces the
same restricted direction and single-slab rules as `mc_spectrum`. Runner paths,
including regenerated/wide bremsstrahlung, must pass the groove specification.

Finite-footprint behavior remains the existing approximation: launch and
electron side exits honor the real footprint, while photon groove escape treats
the surface as laterally periodic. For an emitter at depth `z`, the photon
travels transversely

```text
Delta x = L_esc cos(tp) = z cot(tp) + O(Lambda).
```

The affected side-edge fraction is therefore
`min(z cot(tp) / crystal_width + O(Lambda / crystal_width), 1)`.
`O(Lambda / crystal_width)` describes only the periodic-phase correction, not
the total finite-side error unless relevant emission depths are `O(h)`.

## Failure Handling and Compatibility

- `groove=None`: strict bit-for-bit regression.
- Unsupported layers or observation directions: explicit `ValueError`.
- No re-entry intersection: permanent exit, never an infinite vacuum search.
- Tangent or zero-length event: deterministic epsilon nudge and bounded event
  count; exceeding bound raises instead of silently treating vacuum as material.
- Existing segment keys and meanings remain unchanged.
- New vacuum arrays are additive and typed even when empty.

## Tests

Fast CPU regression coverage will include:

1. surface predicate against analytic sawtooth profile;
2. exact facet intersections against fine reference marching;
3. material-to-vacuum exit followed by later-facet re-entry;
4. permanent groove exit with no later intersection;
5. repeated crossings without zero-length loops;
6. every radiating segment midpoint lies in material;
7. vacuum legs preserve energy and direction while advancing time;
8. re-entered electrons resume scattering/stopping/radiation;
9. `groove=None` bit-for-bit transport and spectra;
10. coherent escape remains unchanged;
11. bremsstrahlung uses grooved escape and gains the expected analytic
    Beer--Lambert transmission;
12. runner forwards grooves through normal, line-only, brem-only, and rebuilt
    wide-bremsstrahlung paths;
13. 2D/3D plot data expose faint vacuum legs without treating them as radiating
    segments.

Fresh-context physics validation must check signs, facet bands, tangencies,
flat-surface limit, no-re-entry proof for `n_hat`, and exponential-free-path
memorylessness. Only a human may mark the ledger row `signed-off`.

## Expected Files

- `src/cxr_mc/montecarlo/groove.py`
- `src/cxr_mc/montecarlo/transport.py`
- `src/cxr_mc/montecarlo/spectrum.py`
- `src/cxr_mc/montecarlo/runner.py`
- `src/cxr_mc/plots/trajectories.py`
- `src/cxr_mc/plots/altair_trajectories.py`
- `src/cxr_mc/plots/plotly_trajectories.py`
- `tests/test_groove.py`
- `tests/test_trajectories.py`
- focused runner and plotting regression files
- `docs/physics-validation-ledger.md`

Unrelated transport, remote, notebook, or catalog behavior stays out of scope.
