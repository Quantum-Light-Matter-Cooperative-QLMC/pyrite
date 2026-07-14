# Crystallography adapter decision

## Adopted production boundary

cxr-mc uses `crystals` 1.7 to parse bundled CIF files and expand
crystallographic symmetry. `materials._cif.load_crystal_from_cif` converts the
result to deterministic cell parameters, an expanded fractional basis, and
cell volume. The immutable catalog in `data/materials.toml` points only to
phase-specific CIFs inside packaged `data/cifs/`.

Production loading is offline-only. It does not query Materials Project, COD,
or any other remote structure service at import or run time.

The adapter is intentionally structural. cxr-mc owns and validates:

- atomic form factors and dispersive corrections;
- structure factors, `chi_g`, and `U_g`;
- reflection-family selection and pinned-reflection policy;
- attenuation and multilayer optical depth;
- electron transport and PXR/CBS radiation.

The catalog rejects partial CIF occupancy rather than silently interpreting a
fractional site as a whole atom. Pinned `hkl_families` use positive
representatives, expand to both signs, and require a reason.

## Validation boundary

The `crystals-cif-adapter` ledger row covers lattice parameters, basis, volume,
non-P1 symmetry expansion, deterministic ordering, and partial-occupancy
rejection. Per-phase `validation_id` values in `materials.toml` point to
separate ledger rows for structure provenance or migration equivalence.

`validation_oracles.py` remains optional validation-only plumbing for
`Dans_Diffraction` comparisons of lattice, reciprocal geometry, and
`|F_hkl|^2`. It does not participate in production loading or replace cxr-mc's
scattering physics.

## Distribution constraint

The locked `crystals` 1.7.0 dependency is GPLv3. A licensing review is required
before distributing cxr-mc source, wheels, binaries, or containers that include
it.

Earlier candidate-branch comparisons are retained in git history rather than
as current setup instructions.
