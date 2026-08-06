# CLI verb collapse + module fold (redesign D4, pkg P3)

## Scope

[`docs/plans/cli-redesign-implementation-plan.md`](../../../docs/plans/cli-redesign-implementation-plan.md)
slice 4: land the canonical lifecycle verbs from
[`docs/cli-redesign-rfc.md`](../../../docs/cli-redesign-rfc.md) D4, and retire
`rebrem.py`/`reline.py`/`prune.py` into the checkpoint recompute/cleanup
modules per [`docs/package-structure-rfc.md`](../../../docs/package-structure-rfc.md)
P3 instead of leaving parallel entry points behind shims.

## Verb mapping

D4's table lists `gc` as replacing `prune`, `reap`, and `clear --all`, and `rm`
as replacing `delete`, `clear`, and `prune-jobs`. Applied literally that splits
one command across two verbs by flag, so `checkpoint gc --all` would mean both
"drop obsolete records" and "delete every dataset" — the overloading D4 exists
to remove. Resolved by the intent behind the two verbs instead:

| Retired | Canonical | Why |
|---|---|---|
| `checkpoint prune` | `checkpoint gc` | reclaims records unreachable/obsolete under current profiles |
| `checkpoint clear` | `checkpoint rm` | deletes an explicit target (`MATERIAL...`, `--profile`, or `--all`) |
| `prune` (top level) | `checkpoint gc` | was already an alias of `checkpoint prune` |
| `performance prune` | `performance rm` | deletes explicitly named profile directories |
| `remote clear` | `remote rm` | explicit target |
| `remote prune` | `remote gc` | obsolete records |
| `remote reap` | `remote gc` | orphaned reservations |
| `remote performance prune` | `remote performance rm` | explicit target |

`--all` therefore keeps one meaning per verb: on `rm` it widens the explicit
target set, on `gc` it widens which profiles are scanned for obsolete records.

`remote gc` runs both reclamations that `prune` and `reap` ran separately.
Each half previews unless `--yes`, so the combined command keeps the
preview-then-confirm contract of both retired spellings. The retired spellings
stay as their own hidden commands rather than aliases of `gc`, so `cxr remote
reap --yes` still only releases reservations — an alias would have silently
widened a destructive command.

## Module fold

- `src/cxr_mc/prune.py` deleted; `prune_checkpoints` and its helpers moved
  verbatim into `src/cxr_mc/checkpoint_cleanup.py` alongside
  `clear_checkpoints`.
- `src/cxr_mc/cli/commands/cleanup.py` (new) owns the Click layer
  (`gc_command`, `rm_command`), matching the `recompute.py` /
  `cli/commands/recompute.py` split landed in the first half of this slice.
- `cli/_core.py` gained `hidden_alias(group, command, name)`, replacing the
  hand-rolled copy/rename/hide block in `_remote/cli.py`.

## Deferred, deliberately

- **`remote jobs` → `remote list`, `remote prune-jobs` → `remote rm`.** Both
  concern the job noun that D2b owns. Converting now would leave `remote list`
  meaning jobs while `remote rm`/`gc` mean checkpoints — the noun overloading
  D4 is meant to prevent. Sequenced with D2b.
- **`checkpoint add` / `checkpoint verify`.** D4 lists them with `gc`, but they
  are the content-addressed store's lifecycle verbs; they land with slice 6
  (artifact RFC phase 5), which is what gives them something to address.
- **Remote payload scripts still invoke retired spellings** (`cxr rebrem`,
  `cxr reline`, `cxr slim` from the previous slice; `cxr prune` from this one,
  in `_remote/scripts.py`). They work through the aliases but emit deprecation
  warnings into remote job logs. Out of scope here (D2a `--remote` modifier
  work owns the remote payload surface); worth a dedicated pass.

## Checklist

- [x] `checkpoint_cleanup.py` absorbs `prune.py`; `prune.py` deleted.
- [x] `cli/commands/cleanup.py` Click layer; `cli/_core.py::hidden_alias`.
- [x] `checkpoint gc`/`rm`, `performance rm`, `remote gc`/`rm`,
  `remote performance rm` canonical; retired spellings hidden + warning.
- [x] Eight new rows in `cli/_deprecations.py`; `docs/cli-deprecations.md`
  regenerated.
- [x] `docs/cli-reference.md` and `tests/data/cli_contract.json` regenerated.
- [x] `docs/repo_map.md` ownership entries for `checkpoint_cleanup.py` and
  `cli/commands/cleanup.py`; entry-point rows updated.
- [x] `tests/test_prune.py` → `tests/checkpoint/test_gc.py`,
  `tests/checkpoint/test_clear.py` → `tests/checkpoint/test_rm.py`;
  `_dev.py` suite lists updated. Alias-still-warns coverage added for
  `checkpoint clear`, `performance prune`, `remote prune`, `remote reap`.
- [x] Suites green: cli 960, packaging 189, apps 273, core 983 + 4 pre-existing
  `_adaptive_chunk` failures (CPU-only `_REAL_BYTES=4`, unrelated).

## Not from this branch

`cxr-dev format` reformatted 11 files carrying pre-existing format drift
(`_remote/presentation.py`, `_remote/viewer.py`, `cli/_dashboard.py`,
`cli/commands/profile.py`, `plots/altair_spectra.py`, `plots/spectra.py`,
`tests/test_local_dashboard.py`, `tests/plots/test_material_comparison.py`, and
others). Committed alongside this work at the owner's direction; formatting
only, no behavior change.
