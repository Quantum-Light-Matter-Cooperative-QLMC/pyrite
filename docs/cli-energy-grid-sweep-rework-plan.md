# CLI energy-grid / sweep / profile rework — triaged plan

Status: approved direction (2026-07-26). Replaces the lost original plan doc;
incorporates the audit of TODO P1 #1 subitems 1–15 and the four decisions
conferred with the user.

## Background (previously approved, partially landed)

- Rename `cxr line-grid` → `cxr energy-grid`; split `line` / `brem`
  subcommands.
- Add `cxr sweep` (read/write catalog scan-parameter ranges) — landed in
  `src/cxr_mc/cli/sweep.py`.

## Decisions (2026-07-26)

1. **Terminology.** Catalog `[profiles.*]` keeps the word *profile*. The
   `SweepProfile` `full`/`survey` reduction policies are renamed to
   **fidelity** (`--fidelity full|survey`). Checkpoint dirname parsing
   (`profiles.py` `<mat>--(full|survey)-<hash>` regex) keeps a legacy alias;
   existing checkpoint directories stay readable. TODO P3.1
   (`[profiles.*]` → `[scan_defaults.*]`) is **dropped** — superseded by this
   decision.
2. **Schema inversion (subitem 15).** Profiles reference materials, not the
   reverse. A profile is a named campaign:

   ```toml
   [profiles.sub_100keV]
   energy_keV = { values = [30.0, 40.0, 50.0, 60.0, 100.0] }
   materials = ["hopg", "mose2"]        # optional; absent ⇒ all in-use materials

   [profiles.sub_100keV.overrides.hopg]
   thickness_ang = { values = [...] }   # per-material override, scoped to this profile
   ```

   - `[materials.*].profile` field is removed.
   - Materials without explicit entry inherit profile defaults.
   - "All in-use materials" = the `mats_to_sim.toml` in-use list.
   - Same material may appear in many profiles with different configs — by
     design; result naming is profile-qualified (see remote pull selector).
3. **Derived line-grid bounds** move out of profiles into a shared
   **per-material store** with provenance
   (`source = "derived" | "manual"`). Profiles reference beam energies only;
   grids resolve from the material store. Expensive Monte-Carlo-derived
   bounds can never be deleted by profile edits; deletion only via an
   explicit `cxr energy-grid` command with confirmation.
4. **Uncommitted `sweep set` diff** is reworked before commit: drop
   line-grid pruning and copy-on-create grid duplication, drop
   `[profiles.test]` from `materials.toml`; keep `--yes`, the overwrite
   confirmation, and the test-helper `input=` support.

## Phases

### Phase 0 — independent bugfixes (no schema change)

- **P0.1** Rework pending `sweep.py` diff per decision 4; commit.
- **P0.2** Fix `cxr remote pull` for survey checkpoints (subitem 13): pull
  currently misses `<material>--survey-<hash>/` identity dirs; resolve via
  the existing identity regex + checkpoint `meta.json`.
- **P0.3** Interim fix for the 150 keV error (subitem 5): when validating
  requested energies, consult the standard/source profile grid rows before
  erroring. (Structurally fixed by Phase 1; this unblocks use now.)

### Phase 1 — schema centerpiece

- **P1.1** Rename `full`/`survey` → `--fidelity` across `scan.py`,
  `_remote/cli.py`, `reline.py`, `rebrem.py`, `recompute_defaults.py`;
  legacy `--profile full|survey` aliases warn and forward. Checkpoint
  dirname format unchanged.
- **P1.2** Invert schema per decision 2: catalog parse
  (`materials/catalog.py`), `materials.toml` migration, golden regen
  (`regen-golden` skill), `cxr sweep show/set` retargeted (per-material
  override editing moves under profiles).
- **P1.3** Shared derived-grid store per decision 3, with
  `source = "derived" | "manual"` provenance. `cxr energy-grid apply`
  writes here. Migration moves existing per-profile/per-material
  `E_grid_line_by_energy` rows in.
- **P1.4** Docs: `docs/sweep-profiles.md` (fidelity rename), CLI reference
  regen, repo map.

### Phase 2 — CLI surface (`cxr profile` group)

- **P2.1** New `cxr profile` verb group (subitem 10), retiring
  `cxr sweep set --profile`:
  - `list`, `show NAME` (`cxr profile NAME` aliases show), `create NAME`,
    `set NAME [ranges]`, `add NAME --energy 75 ...` /
    `remove NAME ...` (subitem 8; no overwrite prompt except on
    `standard`), `delete NAME -y` (subitem 9; forbid `standard`; block
    while referenced, list referents), material membership verbs
    (`add-material` / `remove-material` or `--materials`).
  - Explicit `create` required — `set` on unknown name errors with
    suggestions (fixes silent-create footgun).
  - Shell completion for profile names everywhere `NAME` appears
    (subitem 4).
- **P2.2** Range syntax `start:stop:step`, stop-inclusive, mixable with CSV
  (`--energy 30,50:100:25`) on all range options (subitem 2).
- **P2.3** `cxr energy-grid set MATERIAL --energy E --stop S ...` manual
  bound escape hatch, stored `source = "manual"` (subitem 6).
- **P2.4** `--ne-line` / `--ne-brem` as profile settings, single-value grids,
  sweepable (subitem 12). Collapses these out of fidelity policy over time.
- **P2.5** Every command runs through the `cli-ui-ux` skill for design,
  implementation, tests.

### Phase 3 — remote integration

- **P3.1** `cxr remote submit --profile` accepts catalog profile names;
  `--fidelity full|survey` orthogonal (subitem 7). No `survey` catalog
  profile is auto-generated — fidelity covers that role.
- **P3.2** Pull selector `cxr remote pull MATERIAL@PROFILE` (subitem 14):
  resolve via `meta.json` `dataset_identity` (profile name + hash); on-disk
  names stay hash-based. Multiple hashes for one profile → newest wins,
  `--hash` pins, listing available.
- **P3.3** Remnant of subitem 11: run-time `cxr scan --all --profile NAME`
  flag; never writes the catalog; no hidden session state. (The en-masse
  swap need mostly dissolves under the inverted schema — running a profile
  *is* the selection.)

## Subitem → plan mapping

| Subitem | Disposition |
| --- | --- |
| 1 (scan_defaults rename) | Dropped; superseded by fidelity rename (P1.1) |
| 2 (start:stop:step) | P2.2 |
| 3 (bounds immortal) | P1.3 |
| 4 (tabcomplete) | P2.1 |
| 5 (150 keV bug) | P0.3 interim, P1.3 structural |
| 6 (force-set bounds) | P2.3 |
| 7 (remote submit profiles) | P3.1 |
| 8 (incremental add) | P2.1 |
| 9 (delete profiles) | P2.1 |
| 10 (`cxr profile` group) | P2.1 |
| 11 (en-masse transient) | P3.3 (mostly dissolved by inversion) |
| 12 (ne-line/ne-brem) | P2.4 |
| 13 (pull survey broken) | P0.2 |
| 14 (pull by profile) | P3.2 |
| 15 (inversion) | P1.2 (centerpiece) |
