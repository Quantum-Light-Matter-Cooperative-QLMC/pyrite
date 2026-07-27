# CLI energy-grid / sweep / profile rework — triaged plan

Status: approved direction (2026-07-26).

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

- **P2.1** New `cxr profile` verb group, retiring
  `cxr sweep set --profile`:
  - `list`, `show NAME` (`cxr profile NAME` aliases show), `create NAME`,
    `set NAME [ranges]`, `add NAME --energy 75 ...` /
    `remove NAME ...` (subitem 8; no overwrite prompt except on
    `standard`), `delete NAME -y` (subitem 9; forbid `standard`; block
    while referenced, list referents), material membership verbs
    (`add-material` / `remove-material` or `--materials`).
  - Explicit `create` required — `set` on unknown name errors with
    suggestions (fixes silent-create footgun).
  - Shell completion for profile names everywhere `NAME` appears.
- **P2.2** Range syntax `start:stop:step`, stop-inclusive, mixable with CSV
  (`--energy 30,50:100:25`) on all range options.
- **P2.3** `cxr energy-grid set MATERIAL --energy E --stop S ...` manual
  bound escape hatch, stored `source = "manual"`.
- **P2.4** `--ne-line` / `--ne-brem` as profile settings, single-value grids,
  sweepable. Collapses these out of fidelity policy over time.
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

---
Handoff — feature/cli-profile-rework

Landed (committed, verified: lint clean, affected tests green)

- 160c55e Phase 1.1 — --profile full|survey → --fidelity full|survey (scan + remote scan/rebrem/reline/submit + checkpoint recompute); legacy --profile = hidden deprecated alias that warns + forwards; both flags = UsageError. Dirname format / dataset_identity / profiles.py untouched.
- 793a951 — venv/hook bootstrap (your authorized infra change; keep).

Landed since handoff
- 5891fbf Phase 1.2 schema inversion — [materials.*] now identity-only (profile field removed); [profiles.NAME] + [profiles.NAME.overrides.MATERIAL]; 36 per-material overrides migrated into [profiles.standard.overrides.*]; cxr sweep show/set retargeted. DEVIATION: kept single active profile "standard"; true multi-profile materials=[...] membership/selection DEFERRED to Phase 2/3. Tests 1725 pass / 41 skip / 1 baseline fail. Typecheck clean at the 5 baseline diagnostics (IDE may show stale catalog.py ty errors from the interrupted mid-edit state — ignore; committed tree is clean).
- (uncommitted, this session) Phase 1.3 shared derived-grid store — new top-level `[energy_grids.<material>]` per decision 3, `line_by_energy = [{ energy_keV, grid, source }]`; materials.toml data migration (33 rows out of `[profiles.standard.overrides.*]`, plus the profile-default 9 rows into `[energy_grids.standard]`, a sentinel bucket keyed by profile name that materials without their own entry fall back to — all-or-nothing per material, matching pre-1.3 override semantics) landed as uncommitted WIP before this session started; this session wrote the code side. catalog.py/_catalog_decode.py: `_parse_energy_grids` parses the store; `E_grid_line_by_energy` removed from `_SCAN_KEYS` (profiles/overrides no longer carry it, only `E_grid_line` remains as the flat-grid escape hatch); `_scan()` resolves a material's per-energy grid from `energy_grids.get(key, energy_grids.get("standard"))`, coverage-checked against the material's effective `energy_keV`; extra store rows beyond what's needed are NOT an error (decision 3: store may be a superset). DELETED the P0.3 fallback_energies/`_line_energy_set` plumbing entirely — structural fix supersedes it, subitem 5 done. `cxr energy-grid` (line_grid/apply.py) rewritten from regex text-surgery onto tomlkit (format-preserving, matching sweep.py's existing pattern) — the old regex code targeted `[materials.<material>]` for both line and brem, which was already stale/broken against the Phase 1.2 schema (brem lives under `[profiles.standard.overrides.<material>]`); fixed as part of this rewrite. Line-grid provenance (`source`) now lives inline per-row in the catalog, not in `line_grid_provenance.toml`; that sidecar is now brem-only plus optional line-grid notes. New `cxr energy-grid line delete MATERIAL --energy E [--energy E ...]` verb (designed via cli-ui-ux skill): only sanctioned way to remove store rows; confirms unless `-y`/`--dry-run`; `--json` requires `-y` (no prompts in machine mode); safety enforced by pre-write catalog validation alone (deleting a row a live profile still needs fails validation, no separate reference check). `cli/json.py::line_grid_show` and `line_grid/__init__.py::_show()` re-wired to the new store + a `apply.effective_brem()` helper (profiles.standard.overrides.MATERIAL, else profiles.standard default) instead of the broken `materials[material]` lookup. Migrated tests: test_material_catalog.py, test_line_grid_apply.py, test_cli_json.py, test_cli_json_wiring.py, test_line_grid_cli.py (delete verb). Golden (`regen-golden --check`) clean — migration is bit-for-bit numerically identical. cli-reference regenerated for the new delete verb. Tests 1739 pass / 41 skip / 2 baseline fail (both pre-existing, confirmed via git stash: test_line_grid_golden's installed-wheel check, test_agent_tooling's skills-mirror check against untracked `.agents/skills/cavecrew` etc. — unrelated to this branch). Typecheck clean at the 5 baseline diagnostics.

Remaining (not started)

- P1.4 docs — docs/sweep-profiles.md fidelity rename, docs/repo_map.md; the cli-reference regen itself landed with P1.3's delete verb.
- Phase 2 — cxr profile list|show|create|set|add|remove|delete group; start:stop:step (stop-inclusive, CSV-mixable) on range options; cxr energy-grid set MAT --energy E --stop S (source=manual); -l/--ne-line / -b/--ne-brem as profile settings.
- Phase 3 — remote submit --profile = catalog names (orthogonal to --fidelity); remote pull MATERIAL@PROFILE selector via meta.json dataset_identity; scan --all --profile NAME.

Gotchas / facts discovered (carry forward)

- The plan's "pending sweep set diff" (Decision 4) never existed in the tree — no _reconcile_line_grids, no [profiles.test]. P0.1 was implemented fresh; ignore the "before commit" framing.
- 150 keV error has two distinct origins: load-time _catalog_decode.py::_line_grids_by_energy missing = sorted(configured - mapped) (what P0.3 patched via a fallback_energies param threaded from the standard profile), vs. runtime sweep.py:382 _line_grid_for_energy "no E_grid_line configured for beam energy X keV" (untouched, fires only when MC cases build). P1.3 should make the load-time fallback obsolete.
- Current pre-inversion schema: [profiles.standard] holds thickness/energy/tilt/azimuth + E_grid_line_by_energy + E_grid_brem; every [materials.NAME] has profile = "standard" (~24 materials, lines ~552+ in materials.toml).
- Fidelity plumbing: cli/_core.py now exposes fidelity_option / resolve_fidelity; internal profile= strings still carry full/survey for dirname stability. profiles.py PROFILE_NAMES=("full","survey") + _VARIANT_STEM_RE unchanged — leave them.
- CLI reference regens with scripts/generate_cli_reference.py --write (never hand-edit docs/cli-reference.md).
- Every CLI change must route through the cli-ui-ux skill (design/impl/tests) per CLAUDE.md.

Baseline noise (NOT regressions — ignore in all phases)

- tests/test_line_grid_golden.py::test_installed_wheel_layout_fails_with_source_checkout_error fails on a clean tree (exit-code assertion).
- ty check reports 5 pre-existing diagnostics: 3 tomlkit.exceptions submodule warnings in cli/sweep.py, 2 optional-dep unresolved-import (imageio.v3, kaleido) in plots/render_trajectories.py.