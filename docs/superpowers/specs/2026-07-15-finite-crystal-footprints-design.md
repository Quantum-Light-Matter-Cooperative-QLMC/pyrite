# Finite crystal footprints by default

## Goal

Run every standard `Sweep` with a finite rectangular crystal footprint of 5 mm
by 5 mm, while retaining a simple, centrally named setting and the existing
per-sweep override API.

## Design

`cxr_mc.sweep` will define named transverse-dimension defaults, both initially
set to 5.0 mm.  The `Sweep.crystal_width_mm` and `Sweep.crystal_height_mm`
dataclass defaults will use these names.  This makes the default effective for
CLI and notebook material sweeps as well as direct `Sweep(...)` construction.

Existing explicit dimensions continue to override the defaults.  Supplying
`crystal_width_mm=None` and `crystal_height_mm=None` explicitly retains the
laterally infinite-slab mode for comparisons.

## Validation

Add a focused sweep test that asserts default-built cases carry the two named
5 mm dimensions.  Keep the existing finite-footprint Cartesian-product and
invalid-input tests as coverage for override behavior.
