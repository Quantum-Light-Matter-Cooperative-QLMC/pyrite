# Add graphitic carbon nitride (g-C3N4) as a catalog material

Branch: `feature/material-g-c3n4`
TODO scope: P0 #1, "Add graphitic carbon nitride as material".

## Problem

g-C3N4 (graphitic carbon nitride) is not in the material catalog. Add it as a
first-class layered material alongside HOPG / h-BN so it can be selected in
profiles, swept, and simulated.

## Scope

Catalog + crystal-DB addition. One new CIF, one `[crystals.g_c3n4]` block, one
`[materials.g_c3n4]` block in `src/cxr_mc/data/materials.toml`, golden-fixture
regeneration, a `Validation: <id>` marker + physics-ledger row for the new
crystal, and (if it is to be simulated) profile membership + line-grid energy
bounds.

Out of scope: any new physics kernels (pure data addition); reworking the
crystal-DB pipeline.

## Implementation path & likely owners

Catalog data + schema (from existing HOPG/h-BN entries):

- `src/cxr_mc/data/cifs/g_c3n4.cif` — new structure file. 48 CIFs live in
  `src/cxr_mc/data/cifs/`; mirror the layered `hopg.cif` / `hbn.cif` pattern.
- `src/cxr_mc/data/materials.toml`:
  - `[crystals.g_c3n4]` (see `[crystals.hopg]` at ~L694, `[crystals.hbn]` ~L707).
    Required fields per `src/cxr_mc/materials/catalog.py:396`: `cif`,
    `validation_id`, `B_ang2`. HOPG/h-BN also set `full_name`, `phase`,
    `cod_id`, `beam_uvw = [0, 0, 1]`, `E_grid`, `hkl_families = [[0,0,2],[0,0,4]]`,
    `hkl_reason` (basal (00l) c-axis family for fiber-textured layered crystals).
  - `[materials.g_c3n4]` with `label = "g-C3N4"` (see `[materials.hopg]` ~L1177).
- Golden fixture: `tests/data/material_catalog_golden.json`, validated by
  `tests/test_material_catalog.py`. MUST regenerate after any materials.toml /
  catalog.py change — use the `regen-golden` skill (pattern: commit `df14789`).

Physics ledger (AGENTS.md "Backlog and physics"):

- New crystal needs a `validation_id` (e.g. `g-c3n4-structure`, mirroring
  `hbn-structure`) plus a ledger row: source (COD entry), assumptions, limiting
  case, `Validation: <id>`. Fresh context verifies; only a human marks
  `signed-off`.

In-use / simulation wiring (only if g-C3N4 is to be swept, not just cataloged):

- Add `g_c3n4` to the relevant profile `materials = [...]` lists in
  `materials.toml` and to the in-use manifest (`mats_to_sim.toml`,
  `config.MATS_FILE`).
- Derive bespoke line-grid `E_grid_line_by_energy` bounds (see the line-grid
  bounds memory: all in-use materials need per-energy grids; a remote qlmc job
  derives them). This is a remote-GPU step — route via `cxr remote`.

## Stepwise checklist

- [ ] Survey polymorphs; apply the phase-selection rule (one stable/basal phase,
      else all three) → final list of entries to add.
- [ ] Add `cifs/<phase>.cif` per selected phase; verify each parses via loader.
- [ ] Add `[crystals.<phase>]` + `[materials.<phase>]` per phase to `materials.toml`.
- [ ] Add `validation_id` + physics-ledger row per phase; in-code `Validation:` marker.
- [ ] Regenerate `tests/data/material_catalog_golden.json` (`regen-golden`).
- [ ] `cxr-dev test tests/test_material_catalog.py` green; full suite green.
- [ ] Decide in-use vs catalog-only. If in-use: profile membership +
      line-grid bounds job on qlmc.
- [ ] Fresh-context physics verification of the crystal entry.

## Phase selection rule (decided)

Survey the candidate g-C3N4 polymorphs (triazine-based / s-triazine;
tri-s-triazine / heptazine-based, the thermodynamically stable bulk form; and
any third crystalline model in COD/literature). **Selection rule:** if a
polymorph has a stable form realized as large crystals with strong basal-plane
((00l)) reflections, add that one. If the candidates are structurally comparable
for our layered (00l) model, add **all three** as separate catalog entries
(e.g. `g_c3n4_triazine`, `g_c3n4_heptazine`, ...), each with its own CIF,
`cod_id`, and ledger row.

Note the physical caveat for the survey: bulk g-C3N4 is often turbostratic /
poorly crystalline; crystalline triazine-based films (TGCN) exist via
ionothermal growth. The worker records which forms actually give sharp basal
reflections when picking.

## Decisions / open questions

- **CIF source + `cod_id`.** Each selected phase needs a specific COD entry (or
  cited literature CIF) — crystal-DB entries carry a `cod_id`.
- **`B_ang2` (Debye-Waller).** No measured value on hand; may reuse a placeholder
  as h-BN did (`B_ang2 = 3.45`) — but that feeds the P1 Debye-Waller audit debt.
  Flag the placeholder in the ledger row.
- **In-use or catalog-only?** If g-C3N4 is only being cataloged for now, skip
  profile membership + the line-grid bounds job.

## Delegation slices & required skills

- Slice A (data + golden): add CIF, crystals/materials blocks, regen golden,
  suite green. Skills: `regen-golden`, `scientific-library`. Tier: normal.
- Slice B (physics): validation_id + ledger row + fresh-context verification.
  Skills: `physics-validation`, `physics-review`. Tier: lead / human sign-off.
- Slice C (in-use wiring, optional): profile membership + qlmc line-grid bounds.
  Skills: `remote-gpu-jobs`, `monte-carlo`. Tier: normal.

## Acceptance checks

- `cxr-dev test`, `lint`, `typecheck` green.
- `g_c3n4` appears in the catalog; golden fixture regenerated and committed.
- Ledger row present with `Validation: <id>`; in-code marker resolves
  (physics-ledger-auditor clean).
- If in-use: g-C3N4 selectable in its profile and has line-grid bounds.
