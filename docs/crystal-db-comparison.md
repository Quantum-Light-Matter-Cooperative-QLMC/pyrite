# Crystal database comparison — external-DB cross-check

Scoped design note for `issue_notes.md` item #2: *"add tests comparing local
database crystal parameters to external database (use `crystals` library methods
with selectable database from their log — select most reputable/complete for
default)."*

## Goal

Guard the local crystal catalog against silent drift and transcription error by
diffing each entry's **lattice geometry** against an authoritative external
structure database, via the already-vendored [`crystals`](https://pypi.org/project/crystals/)
library (v1.7).

## What is and isn't comparable

The local catalog splits into two data classes with very different external
coverage:

| Parameter | Local source | In external DB? | Comparable |
|---|---|---|---|
| lattice `a, b, c, α, β, γ` | `cifs/*.cif` | yes (COD / Materials Project) | **yes** |
| basis positions + occupancy | `cifs/*.cif` | yes | yes (after P1 expansion) |
| `B_ang2` (isotropic Debye–Waller) | `materials.toml` | **no** — COD/MP do not carry a reliable isotropic B | **no** |

**Key finding:** this cross-check validates *geometry only*. It does **not**
clear `issue_notes.md` item #1 (placeholder Debye–Waller factors) — external
databases don't ship isotropic B, so those still need a literature/temperature
source and a separate check. The two notes look linked but aren't; treating this
test as DW coverage would give a false sense of safety.

## External database options (`crystals` methods)

| Method | Database | Key needed | Verdict |
|---|---|---|---|
| `Crystal.from_database(name)` | 87 vendored builtins, mostly elements | no | **Unusable.** Overlaps our set only on `diamond` / `Si` / `SiC`; none of the ~40 layered TMDs are present. |
| `Crystal.from_cod(num)` | Crystallography Open Database, per-ID | no | **Recommended default.** Live fetch verified; `from_cod(2004955)` (PdTe₂) reproduced local `(4.024, 4.024, 5.113, 90, 90, 120)` exactly. |
| `Crystal.from_mp(query, api_key)` | Materials Project | **yes** | Secondary. DFT-relaxed cells carry a systematic ~1–2 % offset vs experiment; API key is a CI-secret burden. Use only for the MP-sourced entries. |

"Most reputable/complete default" → **COD**, with MP as fallback for entries
whose provenance is an `mp-…` id.

## Blocker: external IDs are only in free-text comments

`from_cod` needs a numeric COD id — **there is no formula search**. Today the ids
live only in CIF header comments, not as structured fields:

- 16 CIFs cite a `COD …` id (e.g. `pdte2`, `fete`, `gese2`, `wte2`, `zrte3`, …)
- 7 CIFs cite an `mp-…` id (`mose2`, `wse2`, `pts2`, `res2`, `pdse2`, `hfte2`, `gese2`)
- **25 CIFs have no machine-readable id** — including workhorses `silicon`,
  `diamond`, `hbn`, `mos2`, `sapphire`, `lif`, `tis2`.

Layered materials have polytype ambiguity (2H vs 1T vs Td); a formula lookup
couldn't disambiguate, but a pinned COD id does. So the id must be stored, not
guessed.

## Recommended implementation path

1. **Promote ids to structured fields.** Add `cod_id` (and optional `mp_id`) to
   each `[crystals.*]` block in `materials.toml`; migrate the 23 known ids out
   of CIF comments. Backfill the missing 25 as a research follow-up (partial
   coverage is fine to start — the test skips entries lacking an id).
2. **Offline-first test.** Cache the fetched external CIFs under `tests/data/`
   and diff against them deterministically (CI-safe, no network). Compare
   lattice `(a, b, c, α, β, γ)` and, where cheap, the P1-expanded basis.
   - Tolerance: `1e-2 Å` / `0.1°` for COD (experimental match);
     loosen to ~2 % for any MP-sourced entry (DFT relaxation).
3. **Optional live regeneration.** A `@pytest.mark.online` test re-fetches from
   COD/MP and refreshes the cached fixtures; skipped (not failed) when offline.
4. **Do not** assert Debye–Waller coverage here — keep item #1 separate.
5. **Skip** the `from_database` builtins path entirely.

## Implementation (delivered)

Built on branch `feat/crystal-db-crosscheck`.

- **Structured ids + metadata.** Each `[crystals.*]` block gained optional
  `cod_id` / `mp_id`, plus a `full_name` and (for polytypic phases) `phase`
  field. All four surface on `CrystalSpec` (`materials/catalog.py`). Parser
  keeps them optional so inline test fixtures need no change; a shipped-catalog
  test enforces `full_name` on every entry. Coverage: **15 COD + 7 MP = 22**
  of 48 crystals carry an external id (the rest lack a machine-readable one —
  research follow-up per the blocker above).
- **Cache is lattice JSON, not CIF.** Deviation from step 2: the comparison is
  geometry-only, and `crystals.Crystal.to_cif` needs an spglib symmetry pass
  that fails on several low-symmetry layered cells (NbTe₂, VTe₂, ReSe₂, …).
  The six comparable numbers are stored in
  `tests/data/external_crystal_lattices.json` instead — deterministic and
  offline by construction. Basis diff is deferred (would need the full CIF).
- **Tolerances** as specified: `1e-2 Å` / `0.1°` for COD; `2 %` relative
  length / `1°` angle for MP.
- **Regeneration.** `scripts/refresh_external_cif.py` re-fetches and rewrites
  the JSON. The `@pytest.mark.online` test in
  `tests/test_crystal_external_db.py` diffs local vs a live fetch; both are
  gated on `CXR_ONLINE_TESTS=1` and skip MP entries when `MP_API_KEY` is unset.
- Item #1 (Debye–Waller) untouched, as required.

## Out of scope

- Debye–Waller / thermal-parameter validation (item #1).
- Space-group / symmetry re-derivation (local CIFs are already P1-expanded;
  `Crystal.from_cif` expands the external asymmetric unit before the diff).
