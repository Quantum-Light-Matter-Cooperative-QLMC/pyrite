# CLI surface reshuffle (redesign D1-D2, output contract)

Branch: `feature/cli-surface-reshuffle`, stacked on
`feature/cli-vocab-controls` at `7b6e588`.

Slice 3 of
[`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md).
The stacked base is deliberate: Slice 3 needs the landed D7 compatibility
substrate and the local D5/D6 work that has not yet landed on `main`. The 17
newer `main` commits are unrelated simulation work and are not merged here.

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

The full CLI suite reaches 983 passing tests after this checkpoint. Its nine
remaining failures predate this branch and come from the stacked Slice 5 base:
seven `tests/test_blaze.py` cases and one `tests/test_local_click_cli.py` blaze
case still invoke retired `--angles` while asserting empty stderr; another
local-click case invokes retired analysis `--default` with the same assertion.
The five failures introduced by the job-path warnings were corrected here.

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

## Non-goals

- No content-addressed store, artifact hash migration, checkpoint lockfile, or
  garbage-collection grace model.
- No physics, simulation-kernel, notebook, or unrelated `main` integration.
- No compatibility removal and no push without separate authority.
