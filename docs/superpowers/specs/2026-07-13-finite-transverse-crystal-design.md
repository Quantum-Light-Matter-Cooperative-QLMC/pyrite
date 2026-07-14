# Finite Transverse Crystal Design

## Goal

Model a crystal as an optional finite rectangular prism rather than an
infinite transverse slab. The model must account for electron loss through
the four lateral faces and photon self-absorption to whichever of the six
faces a fixed far-field photon ray reaches first.

## Scope

The first version models a sample-frame, axis-aligned rectangular prism:

```
-width / 2 <= x <= width / 2
-height / 2 <= y <= height / 2
0 <= z <= thickness
```

The pre-existing slab thickness is the full z extent. The public case and
`Sweep` parameters are optional full dimensions `crystal_width_mm` and
`crystal_height_mm`; public values use millimetres and physics routines use
angstroms after one boundary conversion.

This design deliberately retains the current fixed far-field detector
direction `n_hat`. It does not add detector distance, detector pixels,
parallax, non-rectangular footprints, or arbitrary face orientation.

## Configuration and compatibility

- `crystal_width_mm=None` and `crystal_height_mm=None` mean a laterally
  infinite slab and must preserve the legacy result bit-for-bit.
- A finite footprint requires both dimensions. Supplying exactly one, zero,
  or a negative value is invalid and raises `ValueError` at the public
  geometry/configuration boundary.
- Both dimensions are optional `Sweep` axes. They flow through generated case
  dictionaries, normal and bremsstrahlung transport, the coherent spectrum,
  and bremsstrahlung spectrum.
- The footprint is centered on the existing transverse origin, so the current
  point beam and Gaussian `beam_fwhm_mm` spot are centered on the sample.

## Shared geometry boundary

Introduce one focused geometry helper that computes the first forward
intersection of a ray starting inside the rectangular prism with its boundary.
It accepts ray origins and directions, z thickness, and optional transverse
width/height, and returns forward distance and face identity. It treats a
zero direction component as no intersection with the corresponding pair of
faces and chooses the nearest strictly forward intersection. The infinite
footprint branch delegates to the existing top/bottom slab behavior.

Transport uses this helper (or its vectorized equivalent built on the same
rules) to truncate an electron free flight at the earliest of the six faces.
Top and bottom retain their existing backscatter and transmission counters;
lateral departures increment a new `n_side_exited` diagnostic.

For a finite Gaussian beam, an entry point outside the rectangular footprint
is a missed incident electron: it produces no segments, increments
`n_missed`, and still contributes to `Ne`. Therefore all current
per-electron spectrum normalization stays per incident electron and naturally
captures beam spillover as reduced flux.

## Radiation and attenuation

`mc_spectrum` and `mc_brem_spectrum` use the same first-intersection rules for
each segment midpoint and the existing fixed photon direction `n_hat`. Their
Beer--Lambert path length is the distance to the first reached face, not only
the entrance or exit z face. Layered attenuation remains scoped to z-layer
composition: after choosing the finite-prism exit distance, extend the existing
layer optical-depth helper to integrate only the segment of the ray from its
origin to that exit. It must therefore include every z layer crossed before a
top/bottom exit and stop within the current z layer for a side exit.

For an omitted footprint, both spectrum functions must retain their current
single-slab/layered results exactly. The finite-footprint geometry changes
only escape attenuation and which electron segments exist; it does not change
PXR/CBS amplitudes, resonance energies, detector solid angle, or the
far-field observation direction.

## Validation strategy

Add unit tests for the shared geometry rules: every face, a corner tie,
directions parallel to one or more faces, and the infinite-footprint fallback.
Add transport tests that force a lateral exit and confirm its counter, and
that demonstrate Gaussian beam spillover increments `n_missed` while retaining
the incident-electron `Ne` normalization. Add regression coverage that omitted
dimensions preserve legacy arrays and spectra.

For radiation, use controlled segment fixtures and a lateral-facing `n_hat`
to show coherent and bremsstrahlung attenuation uses the nearer side face;
compare it with the longer infinite-slab escape route. Cover a layered
fixture to verify the selected ray still accumulates the correct z-layer
optical depth. New or changed physics functions carry derivation docstrings
with source equation/assumptions/limiting case, a `Validation:` marker, and a
row plus independent write-up in the physics validation ledger.

## Success criteria

- A finite rectangular width and height reduces transport and/or detected
  flux when electrons or photons encounter a side face first.
- Beam spillover reduces the per-incident-electron yield without changing the
  definition of `Ne`.
- The current laterally infinite slab behavior is unchanged when both
  dimensions are omitted.
- All geometry, transport, line-spectrum, bremsstrahlung, sweep plumbing, and
  validation regression tests pass.
