# Slice 1 WIP handoff — profile-emission-modes

Branch: `feature/profile-emission-modes` (worktree
`/home/alexa/dev/cxr-mc/worktrees/profile-emission-modes`). Not pushed.

Checkpoint commit: `18552dc` (`wip: profile-emission tri-state field + identity
divergence (slice1)`).

## Status vs slice-1 checklist (task-doc steps 1 & 2)

**Step 1 — tri-state field + derived property + Settings + apply_settings +
_PROFILES: DONE.**
**Step 2 — dataset_identity emission divergence key: DONE.**

Slice is functionally complete and green. Committed as `wip:` only because the
coordinator requested a pause/handoff mid-flight, not because anything is
unfinished. No downstream (slice 2/3/4) work started.

## Files touched + exact edits

- `src/cxr_mc/results/store.py`
  - Import line: added `Literal` to `from typing import ...`; added module-level
    alias `EmissionMode = Literal["incoherent", "coherent", "both"]`.
  - `Settings` dataclass (~L67): replaced field `coherent_emission: bool = False`
    with `emission: EmissionMode = "incoherent"` plus a derived
    `@property coherent_emission` returning `self.emission in {"coherent","both"}`.
    (`Settings` is a plain `@dataclass`, not frozen; property + field coexist.)
- `src/cxr_mc/results/__init__.py`
  - Added `EmissionMode` to the `from .store import (...)` block and to `__all__`.
- `src/cxr_mc/profiles.py`
  - Import: `from .results import EmissionMode, Settings`.
  - `SweepProfile` (frozen dataclass, ~L70): replaced field
    `coherent_emission: bool = False` with `emission: EmissionMode = "incoherent"`
    plus derived `@property coherent_emission` (same rule).
  - `apply_settings` (~L84): now passes `emission=self.emission` to `replace`
    (was `coherent_emission=self.coherent_emission`).
  - `_PROFILES`: NOT edited — `full`/`survey` inherit the `"incoherent"` default,
    so no explicit change needed (matches locked decision).
  - `dataset_identity` divergence block (was ~L277-291): replaced the
    `coherent_on = settings_payload.pop("coherent_emission", False)` /
    `if coherent_on: resolved["coherent_emission"]=True` logic with
    `emission = settings_payload.pop("emission", "incoherent")` /
    `if emission != "incoherent": resolved["emission"]=emission`. Clean rename,
    NO legacy `coherent_emission=True` back-compat branch (locked decision).
- `tests/test_profiles.py`
  - Added `SweepProfile` to the profiles import.
  - New tests:
    `test_named_profiles_default_to_incoherent_emission` (back-compat default),
    `test_emission_mode_resolves_through_profile_to_settings` (parametrized over
    the 3 modes: profile.apply_settings -> Settings.emission + derived
    coherent_emission),
    `test_emission_modes_yield_three_distinct_digests_incoherent_unchanged`
    (3 distinct digests; incoherent adds no key; pins the two pre-change
    incoherent digests below).

## Bit-for-bit incoherent digest — VERIFIED, test written, PASSING

Captured with pre-change code, re-verified after the change (identical):
- `hopg` / `full` / incoherent:
  `d0bb205f2268b8cd30801b1146de8daf7e745ca70399919718519542a7c9b45c`
- `mose2` / `survey` / incoherent:
  `a6d8116bf5f0aef9b61f0b43cc4e4e4522e6e8d51167e51450d0fc3232417a7e`

Both are asserted in
`test_emission_modes_yield_three_distinct_digests_incoherent_unchanged`. The
first digest is also independently pinned by the pre-existing
`test_standard_detector_keeps_historical_payload_and_digest_bit_for_bit`, which
still passes — a second, independent proof the incoherent path is unchanged.
`coherent`/`both` each surface `resolved_parameters["emission"]` and produce
distinct digests; incoherent adds no `emission` key.

## Discoveries / downstream breakage (left for later slices)

- `tests/test_scan_coherent.py` and `src/cxr_mc/scan.py` are expected to break
  and are OWNED BY SLICE 3 — do not fix here. Specifically
  `scan.py:530 replace(settings, coherent_emission=coherent)` now raises
  `TypeError` (coherent_emission is a derived property, not a settable field),
  which is exactly the CLI-flag path slice 3 removes. `test_scan_coherent.py`
  asserts `--coherent/--incoherent` flag behavior that no longer holds.
  NOTE: I did NOT run `test_scan_coherent.py` after the change (paused before
  that step). Confirm the breakage is purely the field→property rename before
  slice 3 rewires; it should be.
- All other `coherent_emission` READERS (blaze.py:318, prune.py:96,
  profiles.py:253 build_cases call, scan.py:604/628/639, runner.py:653 via
  case dict, results/store record `coherent_emission` at store.py — actually a
  separate record field, untouched) keep working through the derived property.
- `sweep.py` `build_cases(coherent_emission=...)` signature untouched (reads the
  derived property) — as designed.

## Exact next action to resume

Slice 1 is done. If resuming slice 1 only: nothing left. Before landing, run the
full suite to see slice-3-owned fallout:
`rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test tests/test_scan_coherent.py`
— expect failures; those are slice 3's, do not fix in this branch's slice-1
scope. Otherwise proceed to slice 2 (runner dual-spectra) per the task doc.

## Commands run + results

- `scripts/dev.py test tests/test_profiles.py` -> **24 passed**.
- `scripts/dev.py lint` -> **All checks passed**.
- `scripts/dev.py typecheck` -> 4 diagnostics, ALL pre-existing unrelated GPU
  optional-import errors (`cupy`, `dpctl`, `dpnp`, `cupyx.profiler` in
  `montecarlo/_backend.py` + `runner.py`). None in changed files.
- Ad-hoc digest verification script confirmed the two incoherent digests above,
  3-distinct-digest invariant, and profile→settings resolution for all 3 modes.
