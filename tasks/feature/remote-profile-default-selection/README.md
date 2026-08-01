# Profile-default material selection: `--profile` first, `-m/--material` to narrow

Branch: `feature/remote-profile-default-selection`
TODO scope: P0 #8, "Fix this: make '--profile' the default behavior,
'-m/--material', etc". Closely related to P0 #9 (see cross-reference).

## Problem

Naming a profile in `cxr remote pull` lands in a catch-22 for the `standard`
profile (implicit all-in-use membership):

```
cxr remote pull --profile standard --all --level9
  Error: pull --profile already selects the profile's materials; drop --all
cxr remote pull --profile standard --level9
  Error: profile 'standard' has no explicit material membership;
         name materials alongside --profile, or use --all
```

`--profile standard` demands `--all`; `--profile` + `--all` is rejected. Dead
end. Root cause: `standard` has **no `materials =` row** in `materials.toml` —
it is "implicit all-in-use", and `_profile_default_materials` returns an empty
list for it (`src/cxr_mc/_remote/cli.py:128`), which `pull_command` treats as an
error (`cli.py:1073-1078`).

User's framing: `--profile` should be the **default/natural** selector — naming a
profile should just resolve to its materials (its `[materials]` row if present,
else its in-use set) with no `--all` ceremony; `-m/--material` should be the
explicit **narrowing** override.

## Implementation path & likely owners

Primary: `src/cxr_mc/_remote/cli.py`

- `pull_command` (`cli.py:1048`) — the catch-22 lives at `cli.py:1064-1080`.
- `_profile_default_materials` (`cli.py:128`) — returns `None`/empty for
  implicit-membership profiles (`standard`); this empty is what forces the
  contradictory `--all`.
- `_selected_materials` (`cli.py:116`) / `_reject_all_with_values`
  (`cli.py:~481`) — shared `--all` vs explicit-material gating used across
  remote subcommands.
- Sibling commands to keep consistent: `clear_command` (`cli.py:1121`),
  `stop`/profile handlers (`cli.py:942+`), `profile_pull_command`
  (`cli.py:933`), `run` selection via `resolve_profile_materials` /
  `_start_selected` (`cli.py:787`, `cli.py:142`).
- Resolution source of truth: `CATALOG.profile_materials` and
  `scan.resolve_profile_materials` / `scan.validate_catalog_profile`.

The core fix: make `--profile NAME` with no explicit materials resolve to the
profile's full material set — for implicit-membership profiles that means the
in-use manifest (what `--all` currently loads), NOT an error. `-m/--material`
(or bare MATERIAL args) narrows within the profile. Decide the fate of `--all`
(redundant once `--profile` self-resolves; keep as a no-profile "every in-use
material" selector, or deprecate).

This is CLI-contract work: preserves documented command/help/output/exit
contracts and must regenerate `docs/cli-reference.md`.

## Folded in — P0 #9 (`run` default membership) — IN SCOPE

P0 #9 is folded into this task (user decision: combined). `cxr run standard`
currently runs ALL materials instead of the profile's `[materials]`; profiles
must default to their `[materials]` membership unless explicitly overridden.
Same root design question (profile → material-set resolution). **Design the
resolution rule once, apply it to both `pull` and `run`** (and sibling
commands). Both the `pull` catch-22 and the `run`-runs-everything bug are
symptoms of the same missing default.

## Stepwise checklist

- [ ] Write the profile→materials resolution rule (explicit `[materials]` → that
      list; implicit → in-use set; `-m/--material` narrows; `--all` per decided
      rule; positional PROFILE accepted).
- [ ] Implement in `pull_command` + `_profile_default_materials` (positional
      PROFILE + warn-and-ignore `--profile`/`--all`).
- [ ] Apply the same rule to the `run` path (P0 #9, in scope).
- [ ] Sweep sibling remote commands (`clear`, `stop`) for consistency.
- [ ] Update help text; regenerate `docs/cli-reference.md`.
- [ ] Update/extend CLI contract tests (see `scripts/freeze_cli_contract.py`,
      `tests/` remote-CLI coverage).

## Decided rules

- **Positional profile (decided).** Accept a bare positional PROFILE on `pull`,
  matching `cxr run standard` — the profile is the natural/default selector, no
  `--profile` flag required. Keep `--profile` working as an accepted alias.
- **`--all` fate (decided, agent's call).** Keep `--all` as the standalone
  "every in-use material" selector for the **no-profile** case (backward
  compatible, still useful). Once a profile self-resolves, `--all` is redundant
  with it: `--profile X --all` becomes a **warn-and-ignore no-op**, not the
  current hard error — the profile already selects its materials.
- **`-m/--material` narrows** within the selected profile.

## Open questions

- Backward compat for existing scripts using `--profile X --all`: the decided
  rule is warn-and-ignore (not error) — confirm no automation parses that exact
  error string.
- For an implicit-membership profile (`standard`), "the profile's materials" =
  the in-use manifest (what `--all` loads today). Confirm that is the intended
  default set for `standard` on both `run` and `pull` (ties to P0 #9).

## Delegation slices & required skills

- Slice A (design): resolution rule + `--all` decision + P0 #9 fold-in call,
  written up before code. Skills: `cli-ui-ux`, `repo-orientation`. Tier: lead.
- Slice B (impl): `pull` (+ `run` if folded) + sibling consistency + contract
  tests + `cli-reference.md`. Skills: `cli-ui-ux`, `regression-testing`. Tier:
  normal.

## Acceptance checks

- `cxr remote pull --profile standard --level9` resolves to the profile's
  materials and pulls — no catch-22.
- `-m/--material` narrows within a profile; documented and tested.
- `cxr-dev test`, `lint`, `typecheck` green; CLI contract tests updated.
- `docs/cli-reference.md` regenerated; help text matches behavior.
