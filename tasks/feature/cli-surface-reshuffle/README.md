# CLI surface reshuffle (redesign D1-D2, output contract)

Branch: `feature/cli-surface-reshuffle`, stacked on
`feature/cli-vocab-controls`. It started at `7b6e588`; the owner-authored
vocabulary verification commit `dada9b7` was merged as `0825dfd` while this
slice was active.

Slice 3 of
[`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md).
The stacked base is deliberate: Slice 3 needs the landed D7 compatibility
substrate and the local D5/D6 work that has not yet landed on `main`. Newer
`main` simulation commits remain unrelated and are not merged here.

## Scope and order

1. Make D1 command shapes unambiguous, including an explicit app launch verb.
2. Add the top-level `job list|status|logs|attach|stop` lifecycle. Keep the
   existing `remote` and `energy-grid` spellings as hidden warning aliases.
3. Add `-R/--remote[=TARGET]` to execution verbs and normalize submission
   control to `--wait` / `--detach`; keep retired locality/lifecycle spellings.
4. Add the shared config precedence chain for current profile and remote target:
   per-call flag > `CXR_*` environment > config store > built-in default.
5. Replace automation-bound `--json` booleans with
   `-o/--output [table|json|wide]`; keep hidden `--json` warning aliases. Only
   `json` is a stable machine contract.

The RFC's content-addressed artifact model (its D3) remains Slice 6. Slice 3
only owns the surface/output work assigned by the implementation plan.

## Compatibility mappings

- `remote jobs|status|logs|stop` -> `job list|status|logs|stop`.
- `energy-grid job status|logs|attach|stop` and its flat aliases -> `job ...`.
- Implicit `app analysis|viewer [MATERIAL]` launch -> explicit `launch` leaf.
- Remote execution commands -> their owning execution verb plus `--remote`.
- `--headless`, `--no-pull`, `--detached`, and `--follow` remain accepted only
  through the D7 migration window where a direct mapping is behaviorally safe.
- `--json` remains accepted through the D7 migration window and conflicts with
  an explicit `--output`.

## Acceptance

- [ ] Canonical help exposes noun -> verb paths and hides retired spellings.
- [ ] Every retired command/flag warns once and redirects without widening a
      destructive target.
- [ ] `job` handles run, grid, recompute, and validation job records by kind.
- [ ] `--remote` preserves existing local behavior when absent and selects the
      configured or explicit target when present.
- [ ] `--wait` / `--detach` are uniform and mutually exclusive.
- [ ] `-o json` emits exactly one versioned UTF-8 object plus newline, with no
      prose, color, prompt, or progress on stdout; human formats are not frozen
      automation schemas.
- [ ] `docs/cli-reference.md`, `docs/cli-deprecations.md`, and
      `tests/data/cli_contract.json` are regenerated deliberately.
- [ ] Focused CLI/completion/remote/energy-grid tests, lint, typecheck, and
      relevant suites pass.

## Checkpoints

### Unified job lifecycle

- [x] Added canonical `job list|status|logs|attach|stop` with job-ID completion,
      exact destructive selectors, and `list --kind` normalization.
- [x] Hid and registered warning redirects for `remote jobs|status|logs|stop`,
      `energy-grid job ...`, and the older flat energy-grid job verbs.
- [x] Regenerated the deprecation/reference/Click-contract artifacts and updated
      `docs/repo_map.md`.
- [x] Focused lifecycle/contract verification: 405 passed; lint, typecheck, and
      packaging suite (196 passed) are green.

The first full CLI run reached 983 passing tests; its nine stacked-base failures
were the retired-flag assertions completed by `dada9b7`. After merging that
owner-authored base refresh, the full CLI suite passes: 1007 tests.

### Explicit app actions

- [x] Added `app analysis launch`, `app viewer launch`, and
      `app validation launch`; `export` remains the sibling action.
- [x] Kept implicit app launches as D7 warning redirects, removing the
      material-versus-subcommand ambiguity without breaking old invocations.

### Remote modifier (additive run checkpoint)

- [x] Added `-R/--remote[=TARGET]` to canonical `cxr run`. Bare selection uses
      the configured target; an explicit target is validated and restored after
      the invocation.
- [x] Added mutually exclusive `--wait` / `--detach`; remote default remains
      wait-and-pull, while local runs reject both controls.
- [x] Reused the existing remote run command callback and lifecycle rather than
      duplicating submission logic. Explicit local-only options fail with usage
      guidance when locality is remote.
- [ ] Hide/warn `remote run` only after the remaining D2a execution paths have
      canonical modifier homes, so the migration does not expose a half-retired
      namespace.

### Remote recompute modifiers

- [x] Added `-R/--remote[=TARGET]` and mutually exclusive `--wait` / `--detach`
      to `checkpoint recompute brem|line`, reusing the existing remote queue,
      viewer, and component-pull orchestration.
- [x] Kept local behavior unchanged when `--remote` is absent; reject remote-only
      controls locally and local-only/JSON controls remotely with usage errors.
- [x] Hid `remote rebrem|reline` as D7 warning redirects after their canonical
      modifier homes became complete.
- [x] Focused remote modifier tests: 86 passed; full CLI suite 1026 passed;
      lint, typecheck, reference, deprecation, and contract checks pass.

### Remote energy-grid modifier

- [x] Added `-R/--remote[=TARGET]` and mutually exclusive `--wait` / `--detach`
      to `energy-grid derive`, reusing the existing sliced grid-job submitter.
- [x] Kept local derivation unchanged; remote derivation waits and pulls the
      combined JSON by default, while `--detach` returns after submission.
- [x] Resolved all persistent inputs locally before submission, including
      `--brem-step`, and threaded them through the remote payload and metadata.
- [x] Hid `energy-grid submit` as a D7 warning redirect after canonical parity.
- [x] Focused energy-grid/deprecation/contract tests: 355 passed; full CLI suite
      1033 passed; lint, typecheck, reference, deprecation, and contract checks pass.

### Shared context precedence

- [x] Added `cxr config set|get|list` for `profile.current` and `remote.target`,
      backed by an atomic per-user TOML store.
- [x] Enforced one resolver order: per-call value > `CXR_PROFILE` /
      `CXR_REMOTE_HOST` > config store > built-in `standard` / `qlmc`.
- [x] Applied current-profile resolution to local and compatibility remote runs,
      and remote-target resolution to all subsystem SSH calls.
- [x] Corrected explicit invalid `--remote=TARGET` handling to Click usage exit 2
      and froze option/positional parsing plus host restoration.
- [x] Focused config/run/remote/check verification: 455 passed; full CLI suite
      1014 passed; lint, typecheck, reference, and frozen-contract checks pass.

### Output selector

- [x] Replaced every existing automation-bound `--json` boolean with the shared
      `-o/--output table|json|wide` selector; `table` remains the human default
      and `wide` is explicitly human/non-contractual.
- [x] Kept hidden `--json` compatibility flags with D7 warnings, conflict
      detection, registry rows, and generated deprecation documentation.
- [x] Preserved existing versioned JSON envelopes, stdout purity, partial
      failure, and resumable exit behavior under `-o json`.
- [x] Focused output/deprecation verification: 291 passed; full CLI suite 1017
      passed; lint, typecheck, reference, deprecation, and contract checks pass.

## Non-goals

- No content-addressed store, artifact hash migration, checkpoint lockfile, or
  garbage-collection grace model.
- No physics, simulation-kernel, notebook, or unrelated `main` integration.
- No compatibility removal and no push without separate authority.
