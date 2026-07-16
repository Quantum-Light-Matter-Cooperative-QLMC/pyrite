# Vendor-Like vdW Crystal Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Follow TDD and commit each task.

**Goal:** Add every TODO crystal in `mats_to_sim.toml` as a sourced, vendor-like CIF-backed catalog material with exact cleavage-plane orientation.

**Architecture:** Extend the crystal catalog with a mutually exclusive reciprocal-plane `surface_hkl` orientation while preserving legacy `beam_uvw`. Add 21 missing crystal/material pairs in reviewed batches, reuse and activate PtS2, then consolidate shared catalog fixtures and the manifest once.

**Tech Stack:** Python 3.14, NumPy, pytest, TOML, CIF, crystals 1.7, xraydb, uv.

## Global Constraints

- Work only in the isolated feature worktree and preserve unrelated changes in the main checkout.
- Use vendor-like/exfoliable ambient bulk phases; prefer primary experimental refinements and stable, redistributable CIF data.
- Every new CIF records phase, source/accession, assumptions, limiting checks, and its `Validation:` ID in comments.
- New structural ledger rows remain `unverified`; only a human may mark them `signed-off`.
- Default to `B_ang2 = 0.6`, profile `standard`, and the existing 350--3500 eV crystal grid unless the approved research packet documents an exception.
- Use lowercase keys: `fete`, `gep`, `ges`, `gese`, `gese2`, `nbte2`, `pds2`, `pdte2`, `ptbi2`, `pts2`, `ptte2`, `res2`, `rese2`, `2h_tas2`, `2h_tase2`, `tate2`, `tite2`, `vse2`, `vte2`, `wte2`, `zrte3`, `zrte5`.
- Each implementation task uses RED-GREEN-REFACTOR and reports exact focused commands/results.
- Batch agents must not regenerate `tests/data/material_catalog_golden.json`; Task 8 owns all shared fixture/count reconciliation.

---

### Task 1: Reciprocal surface orientation

**Files:** Modify `src/cxr_mc/materials/catalog.py`, `src/cxr_mc/montecarlo/geometry.py`, `src/cxr_mc/montecarlo/spectrum.py`, `src/cxr_mc/montecarlo/detector.py`, `src/cxr_mc/montecarlo/runner.py`, `src/cxr_mc/sweep.py`, relevant focused tests, and `docs/physics-validation-ledger.md`.

Implement `CrystalSpec.surface_hkl: tuple[int, int, int] | None` and make `beam_uvw` optional, requiring exactly one. `_orientation_R` must rotate the reciprocal vector for `surface_hkl` onto sample `+z`, while legacy `beam_uvw` remains bit-for-bit compatible. Plumb `surface_hkl` through crystal params, cases, radiators, spectra, detector mosaic geometry, and runner paths. Explicit Sweep/Layer `beam_uvw` overrides clear the catalog surface orientation. Add `Validation: surface-hkl-orientation`, parser errors, orthogonal equivalence, nonorthogonal alignment, handedness/azimuth, end-to-end case, and legacy-regression tests.

### Task 2: Fe/Bi/Re/Ta transport foundation

**Files:** Modify `src/cxr_mc/materials/_transport_data.py`, conditionally `src/cxr_mc/materials/crystal.py`, `tests/test_form_factors.py`, and `tests/test_montecarlo.py`.

Add sourced Z, atomic mass, and mean excitation energy for Fe, Bi, Re, and Ta. Verify xraydb form factors and analytic transport fallback. Review actual soft-X-ray edge placement before changing `_EDGE_PRONE`; do not expand it mechanically.

### Task 3: Germanium-family crystals

**Files:** Create CIFs and a focused test module; modify `src/cxr_mc/data/materials.toml` and the validation ledger.

Implement GeP layered C2/m with surface `(20-1)`; GeS and GeSe ambient orthorhombic Pnma with surface `(100)` and `(200)` reflection; micaceous beta-GeSe2 with its source-confirmed basal surface and lowest allowed parallel reflection. Pin sourced lattice, volume, basis multiplicity/stoichiometry, orientation, and finite nonzero `F_g`, `chi_g`, `U_g`.

### Task 4: Nb/Ti/V crystals

**Files:** Create CIFs and a focused test module; modify catalog and ledger.

Implement distorted monoclinic NbTe2, 1T P-3m1 TiTe2 and VSe2, and vendor-like bulk distorted VTe2. Use source-confirmed basal surfaces and lowest non-extinct parallel reflections.

### Task 5: Pd/Pt crystals

**Files:** Create CIFs and a focused test module; modify catalog and ledger.

Implement pentagonal Pbca PdS2 `(002)`, 1T P-3m1 PdTe2, layered trigonal beta-PtBi2, and 1T PtTe2 with basal orientations. Reuse the existing PtS2 CIF and structural source; update only its orientation/reflection contract if needed for exact symmetric-cut consistency.

### Task 6: Re/Ta crystals

**Files:** Create CIFs and a focused test module; modify catalog and ledger.

Implement distorted-1T triclinic ReS2/ReSe2, 2H P6_3/mmc TaS2/TaSe2 `(002)`, and ambient distorted monoclinic TaTe2, excluding low-temperature superstructures.

### Task 7: Fe/W/Zr crystals

**Files:** Create CIFs and a focused test module; modify catalog and ledger.

Implement room-temperature tetragonal PbO-type FeTe, ambient Td orthorhombic WTe2 `(002)`, monoclinic ZrTe3 with its basal `(00l)` family, and orthorhombic ZrTe5 with surface `(010)` and reflection `(020)`.

### Task 8: Catalog and manifest consolidation

**Files:** Modify `mats_to_sim.toml`, shared catalog/count tests, `tests/data/material_catalog_golden.json`, and any manually maintained expected-key collections.

Copy the current user-owned manifest into the worktree, normalize/uncomment all 22 TODO entries, remove their stale TODOs, preserve the explicit excluded TiS2 comment, and add a test that the real manifest contains unique catalog keys and can build one case per material. Reconcile catalog counts to 48/48, current h-BN thickness behavior, stale energy-grid fingerprints, and `check-config` output. Regenerate rather than hand-edit the golden fixture and validate its JSON.

### Task 9: Whole-branch verification and integration

Run `cxr check-config`, the focused catalog/crystallography/sweep/form-factor/transport suites, canonical `scripts/dev.py verify`, and a fresh whole-branch physics/crystallography review. Fix all Critical/Important findings, reverify, then fast-forward the verified branch into main while preserving and auditing the original dirty diff.
