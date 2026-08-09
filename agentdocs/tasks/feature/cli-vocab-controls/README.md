# CLI vocab controls + deprecation rollout (redesign D5–D7)

Branch: `feature/cli-vocab-controls`; original work started at `8671240`, and
the closure worktree was recreated from `main` at `ab76c6a`.
Slice 5 of [`agentdocs/plans/cli-redesign-implementation-plan.md`](../../../plans/cli-redesign-implementation-plan.md).

**Status: complete; closure audit 2026-08-05.** D5 and D6 implementation,
focused tests, generated deprecation/reference/contract artifacts, and the full
verification matrix are current. Post-slice-3 audit corrected stale fidelity
migration prose, fixed remaining preview-first destructive commands, and
removed stale canonical-flag examples.

## Ordering caveat (read first)

Slice 5 was gated on slice 3 (D1–D3) in the plan's original table. Slice 4 and
the D7 harness landed ahead of it because the
redesign RFC's own migration plan ([§4](../../../../docs/cli-redesign-rfc.md))
sequences the additive D4/D5 vocab work as phase 1, before the noun reshuffle.
The same reasoning covered the D5 flag work here. Slice 3 later landed the two
dependent pieces:

- **`-o/--output [table|json|wide]` replacing `--json`** — owned by slice 3;
  retired booleans remain hidden D7 aliases.
- **D6 `--wait` / `--detach`** — canonical remote submitters now share the D2b
  job lifecycle and mutually exclusive controls.

## Key design decision: how a retired flag spelling works

`src/cxr_mc/cli/_deprecations.py` gained `RetiredOption` + `canonical_option`.
Three designs were spiked before landing on this one; the notes matter because
the first two look correct and are not.

1. **Shared `dest` between canonical and retired option** (`click.option("--polar", "tilts")`
   plus `click.option("--tilts", "tilts")`). **Broken.** Click's parser stores
   by dest, so the retired option's callback fires even when only `--polar` was
   given, and `--polar 1 --tilts 2` is last-one-wins rather than arbitrated.
2. **Distinct dest, `expose_value=False`, eager retired callback writing the
   canonical slot.** Warns correctly, but when both spellings are given the
   retired one silently wins regardless of argv order, because Click will not
   overwrite a slot that has no recorded parameter source.
3. **Adopted:** distinct dest, `expose_value=False`, **canonical option forced
   `is_eager=True`**, retired callback checks
   `ctx.get_parameter_source(canonical_dest) is ParameterSource.COMMANDLINE`
   and raises `UsageError` on conflict, otherwise warns and writes the slot.
   `canonical_option` emits both halves from one call so the canonical option
   cannot lose the `is_eager` the check depends on.

**Landmine already hit once:** the retired option's synthesized private name
must be a valid Python identifier. `_retired_set_default_--set-default` is not
(hyphens), so Click filed it as another *option string* and silently collapsed
the option back onto the canonical dest — reintroducing failure mode 1, but only
for flags whose name contains a dash. Fixed with `re.sub(r"\W", "_", ...)`.
Any future change here needs a flag with a hyphen in its name in the test set.

Verified behaviour (`tests/cli/test_deprecations.py`): canonical alone → silent;
retired alone → warning + value flows; both → `UsageError` regardless of order;
repeatable (`multiple=True`) works; `--help` shows only the canonical.

## Implemented

- `cli/_deprecations.py`: `DeprecatedFlag`, `DEPRECATED_FLAGS` registry (21
  rows), `flag_message`, `warn_flag`, `RetiredOption`, `canonical_option`,
  `_implied_dest`, `_VALUE_KWARGS`.
- `energy_grid/_command.py`: `_derive_options` now emits `--energy`/`--polar`/
  `--azimuth`/`--material`/`--save-default` with the plurals retired (covers
  `derive` and `submit`); `apply --materials` → `--material`; `defaults --tilts/
  --azimuths/--set` → `--polar`/`--azimuth`/`--save-default`. Dest names stay
  plural on purpose — they are argv relay keys into the staged argparse
  handlers in `energy_grid/derive.py` and `energy_grid/job.py`, which are
  internal and outside D5's surface. Usage-error strings, help examples, and the
  `defaults` docstring updated to canonical spellings. `_DEFAULT_FIELD_KEYS`
  (`--clear FIELD` values) gained singular canonical names and kept the plurals.
- `cli/commands/blaze.py`: `--angles` → `--polar`; both spellings added to
  `_BlazeCommand._variadic` and `._option_names`, which match on literal flag
  text and would otherwise break the retired spelling's variadic parsing.
- `cli/commands/profile.py`: `--materials` → `--material` on `create`, `set`,
  `add`, `remove`.
- `analyze.py`, `viewer.py`: `-d/--default` → `-d/--save-default`.

## Remaining

1. **D5 leftovers — complete 2026-08-05.** Canonical
   `--thickness/--energy/--polar/--azimuth` options are repeatable on all profile
   range verbs and on canonical/compatibility material setters. Occurrences
   flatten in argv order while retaining CSV + `START:STOP:STEP` support; help
   documents repetition. Profile membership guidance and all five command
   deprecation rows now name canonical `--material`. Focused profile/material/
   deprecation tests pass; the generated command-deprecation table is current.
2. **D6 destructive contract — complete 2026-08-05.** Every live `--yes`
   option also accepts `-y`. Preview-first destructive commands share one
   TTY-only `[y/N]` confirmation helper; non-TTY/CI remains preview-only unless
   `-y/--yes` is supplied. Contract and integration tests freeze both modes.
3. **Tests — complete 2026-08-05.** `tests/cli/test_deprecations.py` now holds
   `DEPRECATED_FLAGS` to the live tree in both directions and freezes canonical
   silence, retired warnings/value flow, both conflict orders, repeatable
   values, hidden help, and a hyphenated retired flag. The registry check found
   and fixed short-option declarations naming `-d` instead of the canonical
   `--save-default`. Focused result: 76 passed; lint and typecheck pass.
4. **Docs/contracts — complete 2026-08-05.**
   `scripts/generate_cli_deprecations.py` now renders both command and option
   tables. `docs/cli-deprecations.md`, `docs/cli-reference.md`, and
   `tests/data/cli_contract.json` were regenerated after D5/D6 and the final
   canonical-help cleanup.
5. **Verify — complete 2026-08-05.** Lint and default-feature typecheck pass.
   Suites: CLI 977 passed; packaging 190 passed; apps 273 passed; core 985
   passed / 39 skipped, plus the 4 expected pre-existing
   `test_adaptive_chunk_*` failures in `tests/montecarlo/test_montecarlo.py` on CPU-only
   machines (hardcoded fp32 `_REAL_BYTES=4`). The core run's only additional
   failure was a sandbox forkserver socket denial; its focused end-to-end test
   passed outside the sandbox (1 passed, 66 deselected).
6. **Post-slice-3 closure audit — complete 2026-08-05.** Restored D6
   preview-first behavior on `profile delete` and `energy-grid line delete`,
   added fail-closed revalidation of previewed catalog state, corrected stale
   D5 help/examples and fidelity documentation, and extended the removal-window
   assertion to retired flags. Fresh focused result: 529 passed; lint,
   deprecation/reference/contract generators, and `diff --check`
   pass. Full CLI reaches 1047 passed with one unrelated failure in
   `test_mott_missing_table_logs_debug_once`; the identical failure reproduces
   on clean `main` (`transport.py` logs `Ellipsis`). Current-main typecheck and
   strict Sphinx remain blocked by the pre-existing CuPy JIT/geometry diagnostics
   and 28 autosummary/toctree/import warnings recorded by slice 3.

`TODO.md` still has no entry for this slice; task status remains branch-local.
