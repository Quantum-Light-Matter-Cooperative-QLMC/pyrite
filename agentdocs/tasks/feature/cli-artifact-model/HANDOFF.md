# Slice 6 handoff: content-addressed CLI artifact model

Paused at user request on 2026-08-05. Work remains uncommitted and unpushed.

## Git state

- Worktree: **gone as of 2026-08-07** — was
  `<repo>/worktrees/cli-artifact-model`, which no longer exists; the
  `feature/cli-artifact-model` branch is also absent locally and on `origin`.
  Confirm what survived before resuming from this handoff. Task worktrees now
  live under `../cxr-mc-worktrees/`.
- Branch: `feature/cli-artifact-model`
- Base: `feature/cli-vocab-controls` at `7b19402`
- Slice 5 was not on `main`, so this branch intentionally starts from its
  closure commit.
- No commit, push, merge, `TODO.md` edit, or worktree cleanup performed.
- Existing slice-5 worktree was not modified.

## Implemented

### Immutable energy-grid store

- Added `src/cxr_mc/energy_grid/artifacts.py`.
- Schema: `cxr.energy-grid-artifact.v1`.
- Frozen identity fields: schema, material, sorted line rows, brem grid, sorted
  beam energies.
- Provenance, notes, timestamps, ref names, and orphan ages stay outside hash
  bytes.
- Canonical JSON, SHA-256 sharded paths, atomic hard-link publication, dedup,
  verified load, corruption/missing detection, and safe inventory.
- Added hash/byte freeze, dedup, corruption, malformed inventory, and
  per-identity-input regression tests.

### Profile refs and catalog resolution

- `profiles.NAME.energy_grid_refs` accepted and validated by material catalog.
- Selected profile refs load/verify sibling `energy-grid-artifacts` objects.
- Explicit artifact controls beam energies, line grids, and brem grid.
- Legacy `[energy_grids.*]` plus profile brem data remain read-only fallback
  when no explicit ref exists.
- Missing/corrupt/wrong-material refs fail catalog validation.
- Added explicit-ref resolution, legacy/ref equivalence, invalid-ref, and
  profile lifecycle tests.
- Profile clone/rename retain digest; deleting a ref-only profile leaves bytes
  orphaned for GC. Legacy profile-delete compatibility behavior remains.

### Canonical mutation commands and D7 aliases

- Canonical `cxr energy-grid add` creates/deduplicates artifacts and repoints
  only resolved profile.
- Hidden `energy-grid apply` alias redirects to `add` with D7 warning/removal
  metadata.
- Canonical `cxr energy-grid rm MATERIAL --energy ...` creates replacement
  artifact and repoints profile; old bytes remain recoverable until GC.
- Hidden `energy-grid line delete` alias redirects to `rm` with D7 metadata.
- Canonical/manual `line set` and `brem set` now create replacement artifacts,
  accept profile selection, preserve omitted row/step values, and leave legacy
  TOML payloads unchanged.
- Destructive preview, non-TTY/JSON rules, exact-preview revalidation, JSON
  envelope, and recovery-oriented messages added.

### Profile-scoped provenance

- New artifact-backed provenance is keyed by profile, material, band, and
  energy; standard profile falls back to old material-global records.
- Prevents a manual edit in one profile from blocking or labeling another
  profile's artifact.
- Text and JSON show paths resolve profile-scoped provenance.
- This was the final edit before pause and needs focused rerun (see below).

### Show/verify/GC

- `energy-grid show`, `line show`, and `brem show` accept profile selection and
  resolve explicit artifact refs before legacy fallback.
- Text/JSON output exposes resolved profile and artifact digest.
- Added `src/cxr_mc/energy_grid/gc.py`.
- `energy-grid verify` reports missing/corrupt profile or campaign-lock refs and
  corrupt stored objects.
- `energy-grid gc` uses profile refs plus active/archive campaign locks as
  roots, records orphan age in `.gc-metadata.json`, defaults to 14-day grace,
  supports `--prune-all`, previews exact candidates, and revalidates catalog,
  locks, inventory, metadata, and candidates before deletion.
- Existing checkpoint case-CAS and `checkpoint gc` semantics are unchanged.

### Campaign lockfile

- Added `src/cxr_mc/campaign_lock.py`, schema `cxr.campaign-lock.v1`.
- Successful completed datasets write deterministic `cxr.lock.json` beside
  directory checkpoints; legacy pickle paths use sibling `*.lock.json`.
- Payload records profile, material, dataset identity, energy-grid refs, and an
  explicit legacy-grid marker.
- Incomplete/paused runs do not write locks.
- Lock write is inside completion failure handling: a lock-write failure cannot
  claim successful completion.

### Remote artifact sync

- Code sync inventories verified local objects and uses one remote `sha256sum`
  inventory round trip.
- Matching hashes at exact canonical remote shard paths are omitted from tar;
  missing/corrupt/misplaced objects are transferred.
- Remote inventory command/path coupling and malformed output have tests.

### Documentation/generated surfaces

- Updated artifact RFC resolutions and repository map.
- Added D7 registry rows for new aliases and inherited retired flags.
- Generated CLI deprecation table once.
- Generated CLI reference and contract using an import-only CuPy stub under
  `/tmp/cxr-cupy-import-stub` because local environment lacks CuPy.
- Root energy-grid help was updated after that generation, so CLI reference and
  contract are currently stale and must be regenerated.

## Verification evidence

Green before latest profile-provenance/root-help edits:

- Artifact/catalog/add focus: `11 passed`.
- GC + campaign locks: `11 passed`.
- Remote sync + campaign locks: `6 passed`.
- Delete/alias CLI subset: `11 passed`.
- Broad changed-surface matrix: `220 passed`; eight failures were all the
  allowed baseline `ModuleNotFoundError: No module named 'cupy'` import path.
- Latest artifact/provenance/apply/remote/catalog focus: `36 passed`.
- Lint was green before the final provenance, test, registry, remote-path, and
  root-help edits.
- Generated reference and contract `--check` passed before root-help changed.

Allowed baseline failures per user: Sphinx, CuPy JIT/import, geometry typing,
and toctree issues. Do not classify unrelated new failures under that waiver.

## Immediate resume work

1. Run formatting/lint and the latest focused tests; the final
   `profile_records()` change and its new assertions have not run:

   ```bash
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test \
     tests/energy-grid/test_artifacts.py \
     tests/energy-grid/test_provenance.py \
     tests/energy-grid/test_apply.py \
     tests/energy-grid/test_artifact_gc.py \
     tests/energy-grid/test_cli.py \
     tests/cli/test_json.py \
     tests/cli/test_json_wiring.py \
     tests/cli/test_profile.py \
     tests/materials/test_material_catalog.py \
     tests/test_campaign_lock.py \
     tests/remote/test_artifact_sync.py \
     -k 'not standalone_module_entry_points_remain_available'
   ```

2. Rerun deprecation registry tests. The prior run found live but unregistered
   `energy-grid add --materials` and `energy-grid rm --json`; registry rows were
   added afterward but not verified:

   ```bash
   rtk env PYTHONPATH=/tmp/cxr-cupy-import-stub \
     UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test \
     tests/cli/test_deprecations.py tests/cli/test_reference.py
   ```

3. Regenerate all three artifacts after latest root help/registry changes, then
   run checks:

   ```bash
   rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python \
     scripts/generate_cli_deprecations.py --write docs/cli-deprecations.md
   rtk env PYTHONPATH=/tmp/cxr-cupy-import-stub \
     UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python \
     scripts/generate_cli_reference.py --write docs/cli-reference.md
   rtk env PYTHONPATH=/tmp/cxr-cupy-import-stub \
     UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python \
     scripts/freeze_cli_contract.py --write tests/data/cli_contract.json
   ```

4. Strengthen remaining acceptance evidence:

   - Add run-level lock wiring test asserting dataset identity, artifact digest,
     and legacy marker. Current scan test checks only lock existence/profile;
     standalone writer tests cover payload details.
   - Add real hidden `apply` integration test proving alias leaves legacy and
     standard scan payload unchanged. Current alias test proves warning and
     dispatch; canonical `add` integration proves immutable behavior.
   - Consider profile show/list ref observability. Energy-grid show is complete,
     but profile show/list still omit `energy_grid_refs`.

5. Run CLI suite/typecheck/verification commands, classify only the explicitly
   allowed baseline failures, run `git diff --check`, and review scoped diffs.

6. When all checks close, update this task README acceptance/checkpoint boxes and
   mark slice 6 status in `agentdocs/plans/cli-redesign-implementation-plan.md`.

## Resume session outcome (2026-08-05)

All six items above are closed. Work is still uncommitted and unpushed.

1. Lint and the focused test list run clean; the 15 failures in that list are all
   `ModuleNotFoundError: No module named 'cupy'`.
2. Deprecation/reference registry consistency tests pass (the previously
   unverified `energy-grid add --materials` and `energy-grid rm --json` rows are
   confirmed).
3. All three generated artifacts regenerated; their `--check` tests pass.
4. Acceptance evidence added:
   - `tests/test_campaign_lock.py` gained two run-level tests driving `cxr run`
     through `scan.command` and asserting lock profile, material, artifact
     digest, `legacy_energy_grid` marker, and the exact dataset identity the
     sweep ran under (both the ref and no-ref branches).
   - `tests/energy-grid/test_cli.py` gained an unmocked `apply` alias test
     running the real `add_file` path against a temp catalog and asserting the
     legacy `[energy_grids.*]` block, profile `energy_keV`, and brem override are
     byte-identical afterwards. Its catalog stub moved to
     `tests/helpers/energy_grid_catalog.py`, shared with `test_apply.py`.
   - `cxr profile show` and `cxr profile list` now expose `energy_grid_refs` in
     both text and JSON, with the legacy-fallback case labeled explicitly.
5. Typecheck fixed for this branch's code: `artifacts.py` casts, the
   `transport.py` `str | None` digest guard, and the `catalog.py` row cast. The
   `tests/cli/test_contract.py` help-path growth bound moved 130 -> 132 for the
   new artifact-store commands. See the README for the base-vs-branch suite
   comparison.
6. Task README boxes checked; slice 6 marked implemented-but-unmerged in
   `agentdocs/plans/cli-redesign-implementation-plan.md`.

## Important cautions

- Do not edit branch `TODO.md`; `main:TODO.md` is authoritative.
- Do not commit, push, merge, or remove worktrees without new user authority.
- Do not run heavy GPU/Monte Carlo work locally.
- Preserve old mutable helper functions in `apply.py` during D7 compatibility;
  canonical commands now use artifact-backed replacements.
- `/tmp/cxr-cupy-import-stub` is import-only scaffolding for CLI tree generation,
  never runtime/test evidence for GPU behavior.
