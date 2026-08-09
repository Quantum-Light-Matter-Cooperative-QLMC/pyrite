# Named beam objects — promote the beam to a first-class CLI noun

**Status 2026-08-09: COMPLETE.** All eight slices A-H landed; every acceptance
check below is met. A-E landed on the task branch and are merged into `main`;
F-H were done directly on `main`. Dropped from `TODO.md` Active.

Branch: `feature/named-beam-objects`
Worktree: `../cxr-mc-worktrees/named-beam-objects`
TODO scope: Active item 2 `>user<` prose (2026-08-07), the follow-on to the
closed-out [`feature/beam-phase-space`](../beam-phase-space/README.md) checklist.

## Problem

`BeamSpec` is a complete physical object in code (`sweep.py:67-125`: energy,
spot, longitudinal policy, transverse Twiss policy, energy spread, charge,
rep-rate), but it has no independent existence for a user. It is only ever
spelled as an inline sub-table of one profile:

```toml
[profiles.hopg_emittance_demo.beam]
rep_rate_hz = 5000.0

[profiles.hopg_emittance_demo.beam.transverse]
normalized_emittance_x_mm_mrad = 0.1
beta_twiss_x_m = 0.05
```

written by nine flags bolted onto `cxr profile create` / `cxr profile set`
(`cli/commands/profile.py:99-160` `_beam_cli_options`). Consequences:

- A beam cannot be named, listed, shown, or reused. Sharing one beam across
  five profiles means copying the block five times, and they then drift.
- The beam has no lifecycle verbs, so there is no `show` to answer "what beam
  is this profile actually running?" without reading raw TOML.
- Every new beam field costs another `cxr profile` flag on two commands, on a
  command whose own subject is the sweep grid, not the beam.

User request: make the beam an independently configurable object *akin to
`material`* — created, named and modified through its own command, then
attached to a profile by name with a flag, the way
`cxr profile add/set/remove <profile> --material KEY,...` attaches materials.

## Scope

In scope: the catalog storage shape for named beams, the new noun group and its
verbs, profile attachment by name, decode/resolution, identity/hashing rules,
migration of the five shipped inline beam blocks, deprecation of the existing
`cxr profile` beam flags, docs and completion.

Out of scope: any change to beam *physics* or sampling. The Courant-Snyder
injection and energy-spread draw landed and were re-derived under
`feature/beam-phase-space`
([`docs/beam-phase-space.md`](../../../docs/beam-phase-space.md), ledger rows
`beam-phase-space-injection` / `beam-energy-spread-injection`). This task moves
where the numbers are *written*, not what they *mean*. No new `Validation:` row
should be needed; if one becomes necessary, that is a signal the task has
drifted out of scope.

## Current state

| Concern | Where it lives today |
| --- | --- |
| Beam dataclass | `sweep.py:67-125` `BeamSpec`, `beam_replace` |
| Inline TOML block | `[profiles.NAME.beam]`, `materials.toml:139-207` (5 profiles) |
| Decode | `materials/catalog.py:783-802` `_BEAM_KEYS`/`_BEAM_POSITIVE_KEYS`, `:840-842` `_TRANSVERSE_*_KEYS`, `_parse_longitudinal_policy` / transverse parser `:845-990` |
| Apply to sweep | `config.material_sweep` (`catalog.py:203`), `profiles.py:284-318` |
| Hashed payload | `profiles.py:283-318` — flat legacy projection; new fields join the hash only when they diverge from inert defaults |
| CLI | `cli/commands/profile.py:99-160` `_beam_cli_options`, `:215-269` `_collect_beam_updates`, `:272-294` `_apply_beam_updates`, key tuple `:342` |
| Material analogue | `[materials.NAME]` (`materials.toml:1208+`, `label` only), `cli/commands/material.py`, `_catalog_io.material_rows`, `_completion.complete_material{,_csv}` |

Note that `[materials.NAME]` is a *thin* table — the physics lives in
`materials/catalog.py`. A beam table is the opposite: the table is the whole
object. So "akin to material" is the right *surface* analogy and the wrong
*storage* analogy; see decision 2.

## Design sketch

```toml
[beams.rf_gun_200fs]
label = "RF gun, 200 fs"
rep_rate_hz = 5000.0
bunch_charge_pc = 1.0

[beams.rf_gun_200fs.longitudinal]
kind = "gaussian"
envelope_rms_fs = 200.0

[beams.rf_gun_200fs.transverse]
normalized_emittance_x_mm_mrad = 0.1
beta_twiss_x_m = 0.05

[profiles.hopg_hbn_gaussian_200fs]
beam = "rf_gun_200fs"
```

```
cxr beam list
cxr beam show rf_gun_200fs
cxr beam create rf_gun_200fs --emittance 0.1 --twiss-beta 0.05 ...
cxr beam set rf_gun_200fs --energy-spread 0.001
cxr beam rename rf_gun_200fs lab_gun
cxr beam delete rf_gun_200fs
cxr profile set hopg_hbn_gaussian_200fs --beam rf_gun_200fs
```

`energy_keV` stays on the profile, not on the beam: it is the primary swept
axis (`profiles.py:292`, `BeamSpec.energy_keV` is a *list* per sweep), and the
normalized-emittance convention exists precisely so one named beam is valid
across that axis (`docs/beam-phase-space.md` "Why normalized, not geometric").

## Decisions and open questions

Decisions 1-3 are recommendations, not settled; confirm before slice C.

1. **Noun spelling: `cxr beam`.** RFC D1 makes every command
   `cxr NOUN VERB` ([`docs/cli-redesign-rfc.md`](../../../docs/cli-redesign-rfc.md)
   L69-83), and `beam` is the name the TOML block, the dataclass field
   (`Sweep.beam`) and the docs already use. `bunch` is wrong — the bunch is one
   field of the beam (`bunch_charge_pc`, `bunch_length_fs`). `beamspec` leaks
   the class name into the user surface. **Open:** user picks; the rest of the
   plan is spelling-independent.
2. **Storage: top-level `[beams.NAME]` in `materials.toml`**, a sibling of
   `[materials.NAME]` and `[profiles.NAME]`, parsed by the same document. Not a
   separate file: `_catalog_io.catalog_text()` and every validation path assume
   one document, and a second file doubles the round-trip/validation surface
   for no user-visible gain.
3. **Identity: resolve the beam to values *before* hashing; the name must not
   enter `parameter_sha256`.** Two profiles pointing at the same beam must reuse
   each other's checkpoints, and renaming a beam must not orphan existing ones.
   This deliberately differs from the `catalog_profile` precedent
   (`profiles.py:331-335`), where the *name* does enter the payload. Carry the
   name as provenance/metadata only. This is the single highest-risk decision in
   the task: get it wrong and every shipped checkpoint stem moves.
4. **Inline block vs reference: hard error, not precedence.** A profile that has
   both `beam = "NAME"` and an inline `[profiles.NAME.beam]` table is a decode
   error naming both spellings — same rule the transverse/FWHM pair already uses
   (`_collect_beam_updates`, `catalog.py` mutual exclusion). No silent winner.
   **Open:** does the inline block survive the support window at all, or is it
   converted on first write? Recommend: keep decoding it (scripts and old
   catalogs exist), stop writing it.
5. **Attach verb shape.** `--material` is list-valued so it takes
   `add`/`set`/`remove`; a profile has exactly one beam, so recommend
   `cxr profile set <profile> --beam NAME` plus `cxr profile remove <profile>
   --beam` to detach. **Open:** confirm `remove --beam` (flag, no value) reads
   better than `--no-beam`.
6. **Fate of the nine `cxr profile` beam flags.** They become the deprecated
   spelling: one stderr warning naming `cxr beam ...`, kept for the published
   support window via the landed D7 substrate (`canonical_option(retired=[...])`
   is already used for `--materials`, `cli/commands/profile.py:931-938`).
   **Open:** deprecate all nine at once, or keep `--transverse-fwhm-mm` as
   sugar? Recommend all nine — a half-migrated surface is the failure mode this
   task exists to end.
7. **Migration of the five shipped profiles** (`hopg_hbn_gaussian_200fs`,
   `hopg_hbn_microtrain_200fs`, `hopg_hbn_compressed_microbunch`,
   `hopg_emittance_demo`, and any other inline block). Converting them to named
   beams must be value-identical, so `parameter_sha256` for every one is
   unchanged — that is the acceptance test, not a review claim.

## Implementation path

Dependency order, with likely owners:

- `src/cxr_mc/materials/catalog.py` — `[beams.*]` table parse (reuse the
  existing `_BEAM_KEYS` / `_TRANSVERSE_*` validators; do not fork them), the
  `beams`-vs-`profiles.*.beam` mutual exclusion, unknown-beam-name error with
  `difflib` suggestions (mirror `material.py:40-48`), resolution of the
  reference into the same dict the inline block produces today.
- `src/cxr_mc/cli/_catalog_io.py` — `beam_rows(document)` beside
  `material_rows` / `profile_rows`.
- `src/cxr_mc/cli/commands/beam.py` (new) — the noun group and its verbs,
  reusing `_beam_cli_options` / `_collect_beam_updates` / `_apply_beam_updates`
  lifted out of `profile.py` into a shared module so both surfaces validate
  identically during the deprecation window.
- `src/cxr_mc/cli/commands/profile.py` — `--beam NAME` on `set`/`remove`,
  retirement warnings on the nine beam flags, `beam` in the show payload.
- `src/cxr_mc/cli/_completion.py` — `complete_beam`.
- `src/cxr_mc/profiles.py:283-318` — resolve-then-hash (decision 3).
- `src/cxr_mc/data/materials.toml` — the named beams plus the five converted
  profiles.
- Docs: `docs/sweep-profiles.md` beam block reference, `docs/cli-reference.md`
  regen, `docs/cli-deprecations.md` regen, `docs/repo_map.md` pointer.

**Coordination risk:** an unfinished CLI redesign stack exists
(`feature/cli-surface-reshuffle`, `feature/cli-artifact-model`, plus the closed
`cli-vocab-controls` / `cli-deprecation-substrate` slices). A brand-new command
module under `cli/commands/` is the low-conflict shape; adding the noun to any
generated command table is the likely conflict point. Check
`docs/plans/cli-redesign-implementation-plan.md` before starting, and rebase
rather than merge.

## Checklist

- [x] **A. Catalog schema** — `[beams.NAME]` parse and validation, reusing the
      existing key sets and the signed/positive split
      (`_TRANSVERSE_SIGNED_KEYS`); unknown-name suggestions; both-spellings
      hard error. Round-trip test including a negative `alpha_twiss`.
      Landed `64992f0`: `_parse_beams` reuses `_parse_profile_beam` verbatim
      (plus optional `label`); unknown-beam error is a plain message (no
      difflib -- matches the existing `_parse_materials` cross-reference
      pattern, not the CLI-layer suggestion helper); both-spellings collide on
      the same TOML `beam` key, so `tomllib`'s own duplicate-key rule rejects
      them before catalog validation runs -- no bespoke check needed.
- [x] **B. Reference resolution + identity** — profile `beam = "NAME"` resolves
      to the same payload as the inline block; resolve-then-hash so
      `parameter_sha256` is unchanged for equal values and invariant under
      rename (decision 3). Test both directions explicitly.
      Landed `64992f0`: resolution happens entirely inside
      `_load_material_catalog_cached` (name/label stripped before
      `profile_beams` is built), so `profiles.py`'s hash payload -- which only
      ever reads `catalog.profile_beam(name)` -- needed zero changes. Tests:
      `test_named_beam_reference_resolves_to_same_payload_as_inline_block`,
      `test_named_beam_renaming_does_not_change_resolved_payload`.
- [x] **C. `cxr beam` noun group** — `list`, `show`, `create`, `set`, `rename`,
      `delete`, with the shared option/validation helpers extracted from
      `profile.py`. `--dry-run` and the `-y` confirmation conventions match the
      existing `profile` / `material` commands. Follow `cli-ui-ux`.
      Landed `7129dfb`: option/validation/writer helpers extracted into
      `cli/commands/_beam_shared.py` (`beam_cli_options`,
      `collect_beam_updates`, `write_beam_fields`, `apply_beam_updates`);
      `profile.py` now imports them with zero behavior change. New
      `cli/commands/beam.py` implements all six verbs on top of
      `_catalog_io.beam_rows`/`beams_table`, matching `profile.py`'s
      dry-run/atomic-write/validate and preview-then-confirm-delete flow;
      `rename` cascades to every referencing profile's `beam = "NAME"`
      string, `delete` is blocked while referenced. `beam` registered as a
      visible noun in `cli/__init__.py`, plus a `cli/beam.py` compatibility
      alias matching `profile.py`/`material.py`. Registered the new
      retired `--json` flags (list/show/delete) in `cli/_deprecations.py`
      and regenerated `docs/cli-deprecations.md` / `docs/cli-reference.md`.
      20 new tests in `tests/cli/test_beam.py`, including a value-equality
      check against a hand-written `[beams.NAME]` block.
- [x] **D. Profile attachment** — `cxr profile set <profile> --beam NAME`,
      `cxr profile remove <profile> --beam`, unknown-beam error, `--beam` in
      `cxr profile show` output and its JSON payload.
      Landed `7fd0043`: `--beam NAME` on `create`/`set` (mutually exclusive
      with the nine inline flags), `_unknown_beam` mirrors `cxr beam`'s own
      suggestion/creation-hint style, `remove --beam` is a flag (no value)
      that detaches named or inline beams and errors if none is present.
      `show` gains a `beam_ref` field (text + JSON) alongside the unchanged
      `beam` inline-table field. Writing inline flags onto a profile that
      already carries a named reference errors instead of silently
      detaching it (decision 4). 14 new tests in `tests/cli/test_profile.py`.
- [x] **E. Deprecate the inline flags** — the nine `_beam_cli_options` flags
      keep working, warn once naming `cxr beam`, and appear in the generated
      deprecation reference. `cxr-dev cli-deprecations` regenerated.
      Landed `7fd0043`: the existing `RetiredOption`/`canonical_option`
      substrate only merges a retired spelling into a *different* flag on
      the *same* command, which doesn't fit "flag keeps its own spelling,
      whole feature moved to an unrelated command" — added a
      `SELF_WARNING_FLAGS` carve-out (flag-level analogue of the existing
      `SELF_WARNING` command-path carve-out) to `cli/_deprecations.py`;
      `profile.py` calls `warn_flag` manually, once per flag per
      invocation. `docs/cli-deprecations.md` / `docs/cli-reference.md`
      regenerated.
- [x] **F. Migrate shipped profiles** — five inline blocks become named beams;
      golden regen (`tests/data/material_catalog_golden.json`, via
      `cxr-dev regen-golden`) and a test pinning every migrated profile's
      `parameter_sha256` to its pre-migration value.
      Done 2026-08-09. **Four** inline blocks, not five -- the original count
      included `promising_low_ne`, which had already been converted to
      `beam = "default"` before this slice. New `[beams.gaussian_200fs]`,
      `[beams.microtrain_200fs]`, `[beams.compressed_microbunch]`,
      `[beams.emittance_demo]` (each with a `label`, stripped before hashing),
      and the four profiles now carry `beam = "NAME"`. All 8
      (profile, material) digests are bit-identical pre/post, pinned as literals
      in `test_named_beam_migration_keeps_shipped_profile_digests_bit_for_bit`
      (`tests/materials/test_profiles.py`), which also covers the
      already-migrated `promising_low_ne`. `cxr-dev regen-golden` produced **no
      diff**: `material_catalog_golden.json` carries crystal `beam_uvw`, not
      beam objects, so a value-identical migration cannot move it.
- [x] **G. Completion + JSON contracts** — `complete_beam`, beam rows in
      whatever `cxr` command inventory/JSON schema tests already cover
      `material`/`profile`.
      Done 2026-08-09. `_beam_keys`/`complete_beam` in `cli/_completion.py`
      (same offline-`tomllib`, `lru_cache`, `_SAFE_TOKEN_RE` shape as
      `complete_material`/`complete_profile`), exported in `__all__`, wired to
      `cxr beam show|set|rename|delete`'s `NAME` and to `--beam` on
      `cxr profile create|set`. `create`'s `NAME` and `rename`'s `NEW_NAME` stay
      uncompleted, matching `cxr profile`. 3 tests in
      `tests/cli/test_completion.py`. No JSON-contract change was needed: the
      frozen `tests/data/cli_contract.json` and `docs/cli-reference.md` already
      covered the `cxr beam` surface from slice C, and both pass `--check`.
- [x] **H. Docs** — `docs/sweep-profiles.md`, `docs/cli-reference.md` regen,
      `docs/repo_map.md` pointer, and a short migration paragraph in
      `docs/beam-phase-space.md` pointing at the new spelling.
      Done 2026-08-09. New "Named beams" section in `docs/sweep-profiles.md`
      (storage shape, the six verbs, attach/detach, rename cascade and delete
      block, resolve-then-hash, inline spelling still decoding); the following
      "Beam block" section rewritten off the profile-only spelling. Migration
      paragraph in `docs/beam-phase-space.md` stating explicitly that no key,
      unit, exclusion, or digest changed. `cli/commands/beam.py` entry plus a
      `beam` mention in the `cli/commands/` blurb in `docs/repo_map.md`.
      `docs/cli-reference.md` needed no regen -- already current.

## Delegation

| Slice | Steps | Skills |
| --- | --- | --- |
| Catalog schema + identity | A, B | `implement-task` + `regen-golden` |
| Noun group | C | `implement-task` + `cli-ui-ux` |
| Attachment + deprecation | D, E | `implement-task` + `cli-ui-ux` |
| Migration | F | `implement-task-lite` + `regen-golden` |
| Completion, docs | G, H | `implement-task-lite` + `documentation-maintenance` |

B is the gate: C-F all assume the identity rule is settled.

## Acceptance checks

- `uv run cxr-dev verify` green.
- Every pre-existing `parameter_sha256` unchanged, including all five migrated
  profiles; renaming a beam changes no checkpoint stem.
- A profile carrying both `beam = "NAME"` and an inline `[profiles.NAME.beam]`
  table fails to load with an error naming both spellings.
- A beam built purely from `cxr beam create` flags produces the same `BeamSpec`
  as the equivalent hand-written TOML block, negative `alpha_twiss` included.
- Each retired `cxr profile` beam flag still works, warns exactly once on
  stderr naming `cxr beam`, and is listed in the generated deprecation doc.
- `docs/cli-reference.md` regenerated; no pre-existing command help, output or
  exit contract changed.
- No new physics claim: `physics-ledger-auditor` reports no orphans and no new
  ledger row.

Verified 2026-08-09 on `main` after F-H:

| check | result |
| --- | --- |
| full suite (`cxr-dev test`) | 2788 passed, 57 skipped |
| `cxr-dev test-suite core` / `cli` | 1160 + 57 skipped / 1123 passed |
| `cxr-dev lint`, `cxr-dev typecheck` | clean |
| `generate_cli_reference.py --check` | pass, no regen needed |
| `tests/cli/test_contract.py` | 139 passed |
| `cxr-dev regen-golden` | no diff |
| shipped `parameter_sha256` | all 8 bit-identical, pinned by test |
| new `Validation:` markers or ledger rows | none added |

The nine `cxr profile` beam flags remain the deprecated spelling on their
published support window (slice E); retiring them is a separate, scheduled
change, not leftover work from this task.
