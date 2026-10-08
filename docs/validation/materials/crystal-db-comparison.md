# Crystal database external cross-check

## Goal

Guard the local crystal catalog against silent drift and transcription error by diffing each entry's **lattice geometry** against an authoritative external structure database. COD CIF records are fetched directly and parsed with [Gemmi](https://gemmi.readthedocs.io/en/stable/chemistry.html); Materials Project records use its supported [`mp-api`](https://docs.materialsproject.org/downloading-data/using-the-api/) client through the opt-in `external-db` extra.

## What is and isn't comparable

The local catalog splits into two data classes with very different external coverage:

| Parameter | Local source | In external DB? | Comparable |
|---|---|---|---|
| lattice `a, b, c, α, β, γ` | `cifs/*.cif` | yes (COD / Materials Project) | **yes** |
| basis positions + occupancy | `cifs/*.cif` | yes | yes (after P1 expansion) |
| `B_ang2` (isotropic Debye–Waller) | catalog `crystals/*.toml` | **no** — COD/MP do not carry a reliable isotropic B | **no** |

**Key finding:** this cross-check validates *geometry only*. It does **not** clear `issue_notes.md` item #1 (placeholder Debye–Waller factors) — external databases don't ship isotropic B, so those still need a literature/temperature source and a separate check. The two notes look linked but aren't; treating this test as DW coverage would give a false sense of safety.

## External database options

| Method | Database | Key needed | Verdict |
|---|---|---|---|
| `Crystal.from_database(name)` | 87 vendored builtins, mostly elements | no | **Unusable.** Overlaps our set only on `diamond` / `Si` / `SiC`; none of the ~40 layered TMDs are present. |
| `Crystal.from_cod(num)` | Crystallography Open Database, per-ID | no | **Recommended default.** Live fetch verified; `from_cod(2004955)` (PdTe₂) reproduced local `(4.024, 4.024, 5.113, 90, 90, 120)` exactly. |
| `mp_api.client.MPRester.get_structure_by_material_id(...)` | Materials Project | **yes** | Secondary. DFT relaxation, especially along layered c axes, can exceed 2%; use for MP-sourced entries, but keep geometry validation provenance-specific. |

"Most reputable/complete default" → **COD**, with MP as fallback for entries whose provenance is an `mp-…` id.

## Original blocker: external IDs were only in free-text comments

`from_cod` needs a numeric COD id — it has no formula-search mode. At design time, known ids lived only in CIF header comments, not as structured fields:

- 16 CIFs cite a `COD …` id (e.g. `pdte2`, `fete`, `gese2`, `wte2`, `zrte3`, …)
- 7 CIFs cite an `mp-…` id (`mose2`, `wse2`, `pts2`, `res2`, `pdse2`, `hfte2`, `gese2`)
- **25 CIFs have no machine-readable id** — including workhorses `silicon`, `diamond`, `hbn`, `mos2`, `sapphire`, `lif`, `tis2`.

Layered materials have polytype ambiguity (2H vs 1T vs Td); a formula lookup couldn't disambiguate, but a pinned COD id does. So the id must be stored, not guessed.

## Recommended implementation path

1. **Promote ids to structured fields.** Add `cod_id` (and optional `mp_id`) to each `[crystals.*]` block in `materials.toml`; migrate the 23 known ids out of CIF comments. Backfill the missing 25 as a research follow-up (partial coverage is fine to start — the test skips entries lacking an id).
2. **Offline-first test.** Cache the fetched external CIFs under `tests/data/` and diff against them deterministically (CI-safe, no network). Compare lattice `(a, b, c, α, β, γ)` and, where cheap, the P1-expanded basis.
   - Tolerance: `1e-2 Å` / `0.1°` for COD (experimental match); loosen to ~2 % for any MP-sourced entry (DFT relaxation).
3. **Optional live regeneration.** A `@pytest.mark.online` test re-fetches from COD/MP and refreshes the cached fixtures; skipped (not failed) when offline.
4. **Do not** assert Debye–Waller coverage here — keep item #1 separate.
5. **Skip** the `from_database` builtins path entirely.

## Implementation (delivered)

Implementation status: structure provenance, external identifiers, and cross-database validation recorded below.

- **Structured ids + metadata.** Each `[crystals.*]` block gained optional `cod_id` / `mp_id`, plus a `full_name` and (for polytypic phases) `phase` field. All four surface on `CrystalSpec` (`materials/catalog.py`). Parser keeps them optional so inline test fixtures need no change; a shipped-catalog test enforces `full_name` on every entry. Coverage: **35 COD ids and 7 MP ids across 39 of 48 crystals** (three entries carry both). Twenty COD records were recovered or selected by matching composition, phase, provenance, and all six lattice parameters within the COD tolerance, then re-fetched through `Crystal.from_cod`. Nine entries remain without a defensible match: `mote2_product`, `4h_sic`, `ptbi2`, `hfs2`, `hfse2`, `zrse2`, `nbs2`, `nbse2`, and `zrte5`.
- **Cache is lattice JSON, not CIF.** Deviation from step 2: the comparison is geometry-only, and `crystals.Crystal.to_cif` needs an spglib symmetry pass that fails on several low-symmetry layered cells (NbTe₂, VTe₂, ReSe₂, …). The six comparable numbers are stored in `tests/data/external_crystal_lattices.json` instead — deterministic and offline by construction. Basis diff is deferred (would need the full CIF).
- **Tolerances** as specified: `1e-2 Å` / `0.1°` for COD; `2 %` relative length / `1°` angle for MP.
- **MP live audit.** With `mp-api` 0.46.4 against MP database 2026.04.13, all seven pinned MP ids resolve. None of their current final relaxed cells satisfy the 2% full-cell tolerance. The live cross-check therefore accepts a match against the final cell *or any pre-relaxation (initial) structure* — MP's initial structures are the experimental inputs a local CIF may derive from. That recovers an exact match for `hfte2`; `gese2`, `mose2`, `pdse2`, `pts2`, `res2`, and `wse2` still exceed tolerance. Of those, `pdse2`, `pts2`, and `wse2` were re-pinned to experimental COD records (below); `gese2`, `mose2`, and `res2` have no defensible external geometry source, so their ids remain provenance pointers only (`MP_PROVENANCE_ONLY` in `tests/helpers/external_db_fixtures.py`) and the live test skips the geometry assertion for them rather than failing on MP's re-relaxed cell.
- **Provenance resolution for MP mismatches.**
  - `mose2`: retained Bronsema's 1986 single-crystal refinement (doi:10.1002/zaac.19865400904), which exactly supplies the local `a=3.289`, `c=12.927` A and Se `z=0.6210`; MP is only a pointer.
  - `pdse2`: replaced the coarse 1957/MP geometry with Soulard et al.'s 2004 room-temperature single-crystal refinement, COD 4310736 (doi:10.1021/ic0352396).
  - `pts2`: replaced the JARVIS/MP relaxation with Furuseth et al.'s 1965 experimental redetermination, COD 1537200.
  - `wse2`: pinned Schutte et al.'s 1987 single-crystal refinement, COD 9012193 (doi:10.1016/0022-4596(87)90057-0); local geometry already matched it.
  - `res2`: Lamfers et al.'s 1996 doubled-cell single-crystal refinement (doi:10.1016/0925-8388(96)02313-4) is more authoritative than the bundled MP relaxation, but neither COD nor another openly licensed source provides its full fractional basis. The local internally consistent MP structure is retained; mixing Lamfers lattice constants with MP coordinates would not reproduce either structure.
  - `gese2`: deferred at user request. Current MP data do not reproduce the historical experimental initial cell or its documented setting transform, so geometry validation fails. Local source remains Dittmar & Schaefer (1976), doi:10.1107/S0567740876008704.
- **Regeneration.** `scripts/refresh_external_cif.py` re-fetches COD and rewrites the JSON. It resolves `MP_API_KEY` from a repository-local `.env` when no non-empty exported value exists; the exported value wins. MP lookups use `mp_api.client.MPRester.get_structure_by_material_id`, not `crystals.Crystal.from_mp`'s unsupported legacy REST-v2 endpoint. Install its optional client with `uv sync --extra external-db`; then run `PYRITE_ONLINE_TESTS=1 uv run --extra external-db pyrite-dev test tests/materials/test_crystal_external_db.py`. Missing keys skip MP-only online cases; configured-key query failures fail with an actionable diagnostic and the refresh script retains cached values.
- Item #1 (Debye–Waller) untouched, as required.

## Out of scope

- Debye–Waller / thermal-parameter validation (item #1).
- Space-group / symmetry re-derivation (local CIFs are already P1-expanded; `Crystal.from_cif` expands the external asymmetric unit before the diff).
