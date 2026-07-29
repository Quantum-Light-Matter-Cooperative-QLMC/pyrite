# profile-rename

`cxr profile` had no way to rename a named campaign profile without
delete+recreate (losing ranges/overrides/membership). `remove` (strip values
from a profile's grids) already shipped in `27b4643`; this branch adds the
paired `rename` verb — the second half of `TODO.md`'s ">user< CLI Work" item.
Also folded `remove-material` into `remove --materials`, mirroring the
existing `add`/`add-material` relationship (unified command preferred,
positional-args command kept as a legacy alias pointing at it).

## Implementation

- `src/cxr_mc/cli/profile.py`: `rename_command` (`cxr profile rename NAME
  NEW_NAME`), mirrors `delete_command`'s existing-profile lookup and
  `_write` dry-run/atomic-write path.
  - Blocks renaming `standard` (hardcoded default profile name assumed
    throughout `profiles.py`/`materials/catalog.py`).
  - Rejects a `NEW_NAME` that already exists, or that equals `NAME`.
  - Migrates the profile's `[energy_grids.NAME]` fallback bucket (if
    present) to `[energy_grids.NEW_NAME]` alongside the profile table, so
    the rename doesn't orphan the shared line-grid store the way a blind
    key-rename would (`delete_command` blocks deletion on this same
    referent; rename actively carries it over instead).
- `_remove_membership` (mirrors `_add_membership`): shared by unified
  `remove --materials`; `remove_material_command` keeps its own positional
  (`nargs=-1`) implementation, now docstring-pointed at the unified form.
- `tests/test_cli_profile.py`: rename+bucket-migration roundtrip, forbid
  `standard`, reject existing/self name, dry-run no-op; `remove --materials`
  removed/not-member reporting, bare-`remove` usage error.
- `docs/cli-reference.md`: regenerated (`scripts/generate_cli_reference.py
  --write docs/cli-reference.md`).

## Status

Done. Verified: `scripts/dev.py test tests/test_cli_profile.py` (34 passed),
`ruff check` on touched files clean, `docs/cli-reference.md` currency test
passing. Full `scripts/dev.py test` run has 7 pre-existing unrelated
failures (GPU-OOM retry / adaptive-chunk / click-default-snapshot tests,
confirmed failing on `main` before this branch) — not touched here.
