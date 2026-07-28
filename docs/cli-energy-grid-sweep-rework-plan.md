# CLI energy-grid / sweep / profile rework — triaged plan

Status: approved direction (2026-07-26).

## Background (previously approved, partially landed)

- Rename `cxr line-grid` → `cxr energy-grid`; split `line` / `brem`
  subcommands.
- Add `cxr sweep` (read/write catalog scan-parameter ranges) — landed in
  `src/cxr_mc/cli/sweep.py`.

## Decisions

1. **Terminology.** Catalog `[profiles.*]` keeps the word *profile*. The
   `SweepProfile` `full`/`survey` reduction policies are renamed to
   **fidelity** (`--fidelity full|survey`). Checkpoint dirname parsing
   (`profiles.py` `<mat>--(full|survey)-<hash>` regex) keeps a legacy alias;
   existing checkpoint directories stay readable. TODO P3.1
   (`[profiles.*]` → `[scan_defaults.*]`) is **dropped** — superseded by this
   decision.
2. **Schema inversion.** Profiles reference materials, not the
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
    `remove NAME ...` (no overwrite prompt except on
    `standard`), `delete NAME -y` (forbid `standard`; block
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
  `--fidelity full|survey` orthogonal. No `survey` catalog
  profile is auto-generated — fidelity covers that role.
- **P3.2** Pull selector `cxr remote pull MATERIAL@PROFILE:
  resolve via `meta.json` `dataset_identity` (profile name + hash); on-disk
  names stay hash-based. Multiple hashes for one profile → newest wins,
  `--hash` pins, listing available.
- **P3.3** run-time `cxr scan --all --profile NAME`
  flag; never writes the catalog; no hidden session state. (The en-masse
  swap need mostly dissolves under the inverted schema — running a profile
  *is* the selection.)

---
Handoff — feature/cli-profile-rework

Landed (committed, verified: lint clean, affected tests green)

- 160c55e Phase 1.1 — --profile full|survey → --fidelity full|survey (scan + remote scan/rebrem/reline/submit + checkpoint recompute); legacy --profile = hidden deprecated alias that warns + forwards; both flags = UsageError. Dirname format / dataset_identity / profiles.py untouched.
- 793a951 — venv/hook bootstrap (your authorized infra change; keep).
- 5891fbf Phase 1.2 schema inversion — [materials.*] now identity-only (profile field removed); [profiles.NAME] + [profiles.NAME.overrides.MATERIAL]; 36 per-material overrides migrated into [profiles.standard.overrides.*]; cxr sweep show/set retargeted. DEVIATION: kept single active profile "standard"; true multi-profile materials=[...] membership/selection DEFERRED to Phase 2/3. Tests 1725 pass / 41 skip / 1 baseline fail. Typecheck clean at the 5 baseline diagnostics (IDE may show stale catalog.py ty errors from the interrupted mid-edit state — ignore; committed tree is clean).
- 77a346a Phase 1.3 shared derived-grid store — new top-level `[energy_grids.<material>]` per decision 3, `line_by_energy = [{ energy_keV, grid, source }]`; materials.toml data migration (33 rows out of `[profiles.standard.overrides.*]`, plus the profile-default 9 rows into `[energy_grids.standard]`, a sentinel bucket keyed by profile name that materials without their own entry fall back to — all-or-nothing per material, matching pre-1.3 override semantics) landed as uncommitted WIP before this session started; this session wrote the code side. catalog.py/_catalog_decode.py: `_parse_energy_grids` parses the store; `E_grid_line_by_energy` removed from `_SCAN_KEYS` (profiles/overrides no longer carry it, only `E_grid_line` remains as the flat-grid escape hatch); `_scan()` resolves a material's per-energy grid from `energy_grids.get(key, energy_grids.get("standard"))`, coverage-checked against the material's effective `energy_keV`; extra store rows beyond what's needed are NOT an error (decision 3: store may be a superset). DELETED the P0.3 fallback_energies/`_line_energy_set` plumbing entirely — structural fix supersedes it. `cxr energy-grid` (line_grid/apply.py) rewritten from regex text-surgery onto tomlkit (format-preserving, matching sweep.py's existing pattern) — the old regex code targeted `[materials.<material>]` for both line and brem, which was already stale/broken against the Phase 1.2 schema (brem lives under `[profiles.standard.overrides.<material>]`); fixed as part of this rewrite. Line-grid provenance (`source`) now lives inline per-row in the catalog, not in `line_grid_provenance.toml`; that sidecar is now brem-only plus optional line-grid notes. New `cxr energy-grid line delete MATERIAL --energy E [--energy E ...]` verb (designed via cli-ui-ux skill): only sanctioned way to remove store rows; confirms unless `-y`/`--dry-run`; `--json` requires `-y` (no prompts in machine mode); safety enforced by pre-write catalog validation alone (deleting a row a live profile still needs fails validation, no separate reference check). `cli/json.py::line_grid_show` and `line_grid/__init__.py::_show()` re-wired to the new store + a `apply.effective_brem()` helper (profiles.standard.overrides.MATERIAL, else profiles.standard default) instead of the broken `materials[material]` lookup. Migrated tests: test_material_catalog.py, test_line_grid_apply.py, test_cli_json.py, test_cli_json_wiring.py, test_line_grid_cli.py (delete verb). Golden (`regen-golden --check`) clean — migration is bit-for-bit numerically identical. cli-reference regenerated for the new delete verb. Tests 1739 pass / 41 skip / 2 baseline fail. Typecheck clean at the 5 baseline diagnostics.

Remaining (not started)

- Phase 3 — remote submit --profile = catalog names (orthogonal to --fidelity); remote pull MATERIAL@PROFILE selector via meta.json dataset_identity; scan --all --profile NAME.

## WIP 2026-07-27 (Phase 2 done, Phase 3 designed, not started)

Phase 1.4 + Phase 2 committed (52689f6..c5937b4 + 8037421 tilt-range fix, 2dbff7b test fixes). Phase 3 design settled:

- `--profile` reclaimed for catalog profile names on `scan`/`submit`; hidden legacy fidelity alias in `fidelity_option` (cli/_core.py) dropped (unreleased shim).
- Seam: `load_material_catalog(path, *, profile="standard")`; `_parse_materials` parametrized; `config.material_sweep(..., catalog_profile=)`; cache keyed (path, profile).
- `dataset_identity` gains conditional `catalog_profile` key only when != standard (hashes for standard stay bit-identical; pattern like `variant`).
- Stems stay hash-based; bare `<material>` canonical stem only for standard profile. `_VARIANT_STEM_RE` untouched.
- `scan --all --profile NAME`: intersect in-use list with `profiles.NAME.materials` when present; explicit non-member material → exit 2; unknown profile → exit 2 + available names.
- Submit: `--profile NAME` threaded lifecycle → scripts emit ` --profile NAME` when != standard (remote runs synced scan.py); metadata line added.
- Pull (part B): MATERIAL@PROFILE resolves via remote meta.json dataset_identity; newest hash wins, `--hash` pins, `--hash` needs listing; split before `_check_shell_tokens` (@ not in safe-token set).

Exploration facts (verified, from subagent reports): remote CLI all in `src/cxr_mc/_remote/cli.py` (submit=`start_command` :698, pull :927); fidelity emitted by `_remote/scripts.py` `_chunked_queue_script`/`_queue_script` only when != full; pull ignores remote meta.json today (dirname regex only, survey-only discovery `lifecycle._resolve_survey_stems` :762); scan.py `_resolved_run` :459 uses `config.material_sweep` via singleton `materials.CATALOG`; catalog `_parse_materials` :743 hardcodes standard, `load_material_catalog` :891 no profile param. Tests to update: legacy-alias tests in test_remote_click.py + test_local_click_cli.py; contract json regen via scripts/freeze_cli_contract.py; cli-reference regen.

Baseline failures to ignore: line_grid_golden wheel-layout, agent-tooling skill mirror, one crystals-library API failure (user-reported, pre-existing).

Gotchas / facts discovered (carry forward)

- 150 keV error has two distinct origins: load-time _catalog_decode.py::_line_grids_by_energy missing = sorted(configured - mapped) (what P0.3 patched via a fallback_energies param threaded from the standard profile), vs. runtime sweep.py:382 _line_grid_for_energy "no E_grid_line configured for beam energy X keV" (untouched, fires only when MC cases build). P1.3 should make the load-time fallback obsolete.
- Current pre-inversion schema: [profiles.standard] holds thickness/energy/tilt/azimuth + E_grid_line_by_energy + E_grid_brem; every [materials.NAME] has profile = "standard" (~24 materials, lines ~552+ in materials.toml).
- Fidelity plumbing: cli/_core.py now exposes fidelity_option / resolve_fidelity; internal profile= strings still carry full/survey for dirname stability. profiles.py PROFILE_NAMES=("full","survey") + _VARIANT_STEM_RE unchanged — leave them.
- CLI reference regens with scripts/generate_cli_reference.py --write (never hand-edit docs/cli-reference.md).
- Every CLI change must route through the cli-ui-ux skill (design/impl/tests) per CLAUDE.md.

Baseline noise (NOT regressions — ignore in all phases)

- tests/test_line_grid_golden.py::test_installed_wheel_layout_fails_with_source_checkout_error fails on a clean tree (exit-code assertion).
- ty check reports 5 pre-existing diagnostics: 3 tomlkit.exceptions submodule warnings in cli/sweep.py, 2 optional-dep unresolved-import (imageio.v3, kaleido) in plots/render_trajectories.py.

## WIP 2026-07-28 (test-suite debug session — suite green again, uncommitted)

Session goal was debugging the 24 failing tests, all fallout of the
half-finished P1.1 `profile` → `fidelity` rename plus one cache regression.
Fixes committed (see below; tree was 98b2758 + 23 dirty files incl. prior
Phase-3-seam WIP, which committed alongside since hunks entangled).

Root causes fixed (all verified: full suite **1755 passed / 39 skipped**,
ruff clean on src/tests/scripts, ty at baseline):

- `profiles.py:290` `_VARIANT_STEM_RE` group renamed `profile` → `fidelity`;
  callers (`profiles.py:298`, `_remote/lifecycle.py:787`) already read
  `match["fidelity"]` — was IndexError, root of the remote/scan_budget/
  sweep/slim KeyError clusters.
- `scan.py:544`, `slim.py:63` — `identity["profile"]` →
  `identity["fidelity"]` (dict renamed in `dataset_identity`, readers
  missed).
- `_remote/cli.py` (5 sites) + `rebrem.py:174,204` + `reline.py:155,185` —
  `_cli_*` handlers read `getattr(args, "profile", ...)` but the click layer
  passes `fidelity=` into the namespace; `--fidelity survey` silently
  dispatched as `full`. Changed to `getattr(args, "fidelity", ...)`.
- `reline.py` — `reline_checkpoints(profile=)` renamed to `fidelity=` to
  match `rebrem_checkpoints`; inner `recompute_options["profile"]` key kept
  (feeds `repair_checkpoint`, whose API is unchanged).
- `catalog.py` — the new `lru_cache` on `load_material_catalog` keyed on
  path only, so tests rewriting the same `materials.toml` path got a stale
  catalog (and cross-test pollution made
  `test_transitive_module_not_found...` order-dependent). Split into a
  stat-keyed wrapper (mtime_ns + size in the cache key) calling
  `_load_material_catalog_cached`; exposed `cache_clear` via `setattr`
  (ty rejects attribute assignment on functions);
  `test_catalog_startup_errors.py` now clears before asserting.
- Test-side updates where the rename was intended: `test_run.py`
  rebrem/reline calls pass `fidelity=`; `test_local_click_cli.py` scan
  expectation gains `catalog_profile: "standard"` (new intentional
  `--profile` surface from Phase 3 seam work).
- Lint: dropped unused `difflib` import (catalog.py), `zip(..., strict=True)`
  (sweep.py:483).

Baseline noise update (supersedes earlier note): ty now reports 16
diagnostics — 15 pre-existing (`cli/profile.py` tomlkit warnings + one
`invalid-argument-type` error at :271, `cli/sweep.py` tomlkit warnings) —
my edits add zero. `scripts/dev.py lint` still fails on 14 pre-existing
ruff errors in `.agents/.claude` caveman-skill scripts (not repo code;
untouched).

Still open / next:

- Phase 3 implementation per the 2026-07-27 design block above: remote
  submit `--profile` = catalog names, pull `MATERIAL@PROFILE` selector,
  `scan --all --profile NAME` (partially present: scan already accepts
  `--profile` → `catalog_profile`). Note: remote submit's dead
  `--profile` placeholder flag (accepted, never forwarded) was removed
  in this session's cleanup — re-add when wiring it through lifecycle.
- `docs/cli-reference.md` regen via `scripts/generate_cli_reference.py
  --write PATH` once Phase 3 lands (never hand-edit).

## WIP 2026-07-27 (Phase 3 landed, suite green, uncommitted)

All three Phase 3 items implemented and tested; see `TODO.md`'s Phase 3
section for the full landed-surface summary (catalog `profile_names`/
`profile_memberships`/`profile_materials()`, `scan.validate_catalog_profile()`
shared by local `scan --all/-A --profile` and remote `submit --profile`,
`dataset_identity`/stem-prediction `catalog_profile` threading with
standard-profile hash compatibility preserved, `_remote/scripts.py` emitting
`--profile` in queue scripts, `_remote/lifecycle.resolve_profile_stem()` for
`cxr remote pull MATERIAL@PROFILE` via remote `meta.json` inspection).

Design decisions made while implementing (not previously pinned down):

- `--all`/`-A` + `--profile NAME` where the profile has an explicit
  `materials` membership list: silently **intersect** (narrow to members) —
  a hard error would defeat the point of a broad `--all` sweep. An explicit
  `MATERIAL --profile NAME` outside that membership is a hard usage error
  instead (exit 2, lists members) since the user named it directly.
- An intersection that leaves zero materials is a usage error (exit 2), not
  a silent empty no-op.
- `catalog_profile` joins `dataset_identity`'s hashed payload only when it is
  not `"standard"` (mirrors the existing `n_electrons`/`n_electrons_brem`
  hash-compatibility pattern) — every pre-existing standard-profile
  checkpoint keeps its historical `parameter_sha256` and stem bit-for-bit.
  A non-standard catalog profile is therefore never `canonical_full`, even
  at fidelity `full` — the bare `<material>` stem stays reserved for
  `standard` so a differently profiled full run can't collide with it
  on disk.
- `cxr remote pull MATERIAL@PROFILE` resolution: on-disk stems never encode
  the catalog profile (only `<material>--<fidelity>-<hash>`), so resolution
  lists `checkpoints/` once, then reads each `<material>`-matching
  candidate's `meta.json` over its own ssh round trip (rare enough per
  material that this isn't worth a bulk remote-side JSON-parsing script).
  Newest `mtime` wins by default and prints the alternate hashes found;
  `--hash HEXPREFIX` pins one and requires exactly one `@`-qualified
  selector in the pull request.
- `@` is split out of `stems` before `transport._check_shell_tokens` (not a
  safe interpolation token); only the resolved on-disk stem ever reaches a
  remote command.

Verified: ruff clean on `src/`, `tests/`, `scripts/`; `ty check` at the
pre-existing 15-diagnostic baseline (zero added); full suite
**1773 passed / 39 skipped**, zero regressions; `docs/cli-reference.md`
regenerated via `scripts/generate_cli_reference.py --write`.

Not done / explicitly out of scope for this pass:

- `cxr remote scan` (the immediate-attach convenience wrapper, distinct from
  `submit`) was not given a `--profile` option — the Phase 3 design block
  only asked for `submit`.
- `rebrem`/`reline` were not threaded with `catalog_profile` — out of the
  design block's scope; their existing `--fidelity` handling is untouched.
- `_refuse_if_busy`'s collision-stem prediction ignores `fidelity` already
  (pre-existing gap, unrelated to this change) and was left as-is rather
  than also threading `catalog_profile` through it.
