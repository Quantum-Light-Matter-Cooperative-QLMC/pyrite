# CLI-specific TODO List

## Structural redesign → see the RFC

The command-structure rethink (noun/verb ordering, remote-as-modifier, unified
job lifecycle, content-hash artifact model, controlled verb/flag vocabularies,
deprecation policy) now lives in **`docs/cli-redesign-rfc.md`**. Its §5 maps
every former punch-list item to a decision. Do structural work from the RFC, not
from here.

The items below are the ones that live *outside* the RFC — file/fix
independently.

## Bugs (fix + regression test, independent of the RFC)

1. `energy-line` / energy-grid azimuth accepts 0–360° instead of the physical
   `(90, 270)` limit. (was High-Pri #4)
2. `cxr [remote] run -p` requires BOTH a single material AND a profile —
   contradicts `run`'s own optional `-m`. Should accept profile, material, or
   both. (was High-Pri #7)
3. `cxr material` help text points to `cxr profile members`, which does not
   exist (membership is `profile set/add/remove --materials`). Stale pointer.
4. `cxr app analysis [MATERIAL] [COMMAND]` mixes an optional positional with a
   subcommand at the same level — a material named `export` collides with the
   `export` subcommand. (RFC D1 removes this structurally, but it's a live
   collision now.)

## Independent ergonomics (ship anytime; not blocked on the RFC)

1. Add `[coherent|incoherent|both]` to `cxr profile set` (and `add`). If a user
   has individually added both, auto-switch to `both` — but make that switch
   **explicit/logged**, not implicit magic. (was High-Pri #2)
2. `cxr` with no args should print help, like `-h/--help`. (was Low-Pri #1)
