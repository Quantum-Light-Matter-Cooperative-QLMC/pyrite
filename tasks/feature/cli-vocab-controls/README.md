# CLI vocab controls + deprecation rollout (redesign D5–D7)

Branch: `feature/cli-vocab-controls`, off `main` at `8671240`.
Slice 5 of [`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md).

**Status: in progress, unverified.** No test run, lint, typecheck, doc
regeneration, or commit has happened on this branch yet. Everything below under
"Done" means "code written and imports cleanly", nothing stronger.

## Ordering caveat (read first)

Slice 5 is gated on slice 3 (D1–D3) in the plan's table, and **slice 3 has not
been implemented**. Slice 4 and the D7 harness landed ahead of it because the
redesign RFC's own migration plan ([§4](../../../docs/cli-redesign-rfc.md))
sequences the additive D4/D5 vocab work as phase 1, before the noun reshuffle.
The same reasoning covers the D5 flag work here. Two parts of slice 5 genuinely
do need slice 3 and are **out of scope on this branch**:

- **`-o/--output [table|json|wide]` replacing `--json`** — the plan assigns the
  `-o` contract to slice 3's row, and `--json` is still live on 23 commands.
- **D6 `--wait` / `--detach`** — needs the D2b unified `job` noun, which slice 3
  introduces. No command has either flag today.

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

Verified behaviour (scratch spikes, not yet committed as tests): canonical alone
→ silent; retired alone → warning + value flows; both → `UsageError` regardless
of order; repeatable (`multiple=True`) works; `--help` shows only the canonical.

## Done (written, not verified)

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

1. **D5 leftovers.**
   - Canonical range flags are not yet repeatable. `_range_cli_options` in
     `cli/commands/profile.py` gives `--thickness/--energy/--polar/--azimuth`
     CSV + `START:STOP:STEP` types but no `multiple=True`; D5 asks for
     "singular, repeatable + `START:STOP:STEP`". Adding it means flattening
     tuple-of-lists in `_collect_updates` and the `sweep set` / `material set`
     equivalents.
   - Existing `DEPRECATIONS` rows still name `--materials` as the canonical
     replacement (e.g. `cxr profile add NAME --materials MATERIAL,...`, 7 rows).
     These are now wrong and must be reworded to `--material`.
2. **D6 destructive contract.** `--yes` exists on 32 commands but `-y` on only
   14 — add the short spelling to the other 18. Then add the TTY `[y/N]`
   prompt: destructive commands currently dead-end with "resubmit with --yes",
   which D6 explicitly calls out. Non-TTY/CI must still require `--yes`.
   Suggest one `confirm_destructive()` helper in `cli/_core.py` next to
   `hidden_alias`.
3. **Tests.** `tests/cli/test_deprecations.py` currently holds `DEPRECATIONS`
   to the live tree in both directions; extend it to do the same for
   `DEPRECATED_FLAGS` by walking `command.params` for `RetiredOption`
   instances. Add the six behaviours from the spike list above, including a
   hyphenated flag name.
4. **Docs/contracts.** `scripts/generate_cli_deprecations.py` only reads
   `DEPRECATIONS`; teach it the flag table. Then regenerate
   `docs/cli-deprecations.md`, `docs/cli-reference.md`, and
   `tests/data/cli_contract.json` — the contract freeze **will** fail until
   regenerated, since every touched command's help text changed.
5. **Verify.** lint, typecheck, then `test-suite cli` / `packaging` / `apps` /
   `core`. Expect 4 pre-existing `test_adaptive_chunk_*` failures in
   `tests/test_montecarlo.py` on CPU-only machines (hardcoded fp32
   `_REAL_BYTES=4`); they are unrelated to this branch.

## Not started

`TODO.md` has no entry for this slice yet, and no commit exists on the branch.
