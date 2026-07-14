# Oriented V2O5 and 1T-TiS2 material presets

**Date:** 2026-07-13

## Goal

Register two bulk (10 micrometre) single-crystal scan presets whose radiation
reflection is parallel to the physical slab normal:

- alpha-V2O5 on the (010) cut;
- 1T-TiS2 on the (003) cut.

This is intentionally symmetric-cut support. It does not introduce an
asymmetric-reflection or reciprocal-miscut model.

## Crystal data

Add `v2o5` to `crystal_structures.toml` as the ambient alpha phase:

- orthorhombic Pmmn, with a complete two-formula-unit conventional basis;
- standard-Pmmn lattice constants `a = 11.512`, `b = 3.564`, and `c = 4.368`
  Angstrom;
- the explicitly expanded V and three inequivalent O sites use the cited
  crystallographic fractional coordinates.

Add `tis2` as the 1T (CdI2-type) P-3m1 phase:

- hexagonal lattice, `a = 3.407`, `c = 5.695` Angstrom;
- Ti at `(0, 0, 0)` and the two sulfur atoms on the 2d sites, using
  `z = 0.2493` and its symmetry mate.

Data provenance is recorded in the crystal-file comments and validation ledger:

- V2O5 lattice and Pmmn phase: McColl, Johnson, and Cora, *Phys. Chem. Chem.
  Phys.* **20**, 15002 (2018), DOI 10.1039/C8CP02187B; fractional coordinates:
  Sipr et al., *Phys. Rev. B* **60**, 14115 (1999),
  DOI 10.1103/PhysRevB.60.14115.
- TiS2: Brown et al., *Phys. Rev. B* **57**, 5101 (1998),
  DOI 10.1103/PhysRevB.57.5101.

## Preset geometry and scan coverage

Both materials use the existing perfect-single-crystal model (`mosaic=False` by
default), `B_ang2 = 0.6`, a 10 micrometre slab, 25/30/35 keV incident energies,
and the standard positive tilt/azimuth grids.

| Material | Slab-normal direct axis | Pinned radiation family | Reason |
| --- | --- | --- | --- |
| `v2o5` | `[001]` | `+-(001)` | Represents the V2O5 layered cut conventionally reported as `(010)` in the historical Pmnm setting; in standard Pmmn it is `(001)`, so `g || n`. |
| `tis2` | `[001]` | `+-(003)` | Represents the requested 1T-TiS2(003) basal cut with `g || n`. |

The requested reflections are pinned rather than selected by
`dominant_reflections`. V2O5 uses two axis conventions in the literature:
the historical Pmnm `(010)` layered cut maps to standard-Pmmn `(001)`. Although
standard-Pmmn `(010)` is extinct, it is a different physical direction and is
not the requested layered reflection. Automatic selection might choose an
asymmetric plane; that would require the separate `n`/`g` split model and is
out of scope. The V2O5 line grid will cover its lower-energy `(001)` line and
the TiS2 grid will extend sufficiently high for `(003)`.

## Transport and validation

Add V and Ti to `TRANSPORT_ELEMENTS` so trajectory transport can resolve each
crystal composition. Use NIST elemental atomic weights and ESTAR/ICRU-37 mean
excitation energies: V `Z=23`, `A=50.9415`, `J=0.245 keV`; Ti `Z=22`,
`A=47.867`, `J=0.233 keV`.

No new control flow is required. Existing crystal loading, composition
normalization, and transport lookup continue to reject malformed structures or
unknown elements.

Tests will assert:

- catalog membership, lattice values, cell volume, and basis stoichiometry;
- nonzero finite structure factors, `chi_g`, and `U_g` for `(010)` and `(003)`;
- V/Ti transport availability;
- the resolved `beam_uvw` and exact pinned reflection lists in the material
  registry/case builder.

The physics-validation ledger will identify these entries as structure and
registry validation only; it will not claim experimental PXR/CBS agreement.
