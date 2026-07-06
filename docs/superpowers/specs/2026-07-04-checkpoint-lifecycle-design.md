# Checkpoint lifecycle: grid-filtered pull, local archive, remote clear

**Date:** 2026-07-04
**Status:** Implemented (2026-07-06) on `feature/checkpoint-lifecycle`
**Scope:** `dev/remote.py`, `src/cxr_mc/{slim,cli}.py`, `src/cxr_mc/results/selection.py`, a new `src/cxr_mc/archive.py`

## Problem

Each material has one monolithic checkpoint `checkpoints/<material>.pkl` shaped
`{config_name: {E0_keV: record}}`. `run.run_sweep` is resumable, so this file is
the accumulated **union of every config ever swept** for that material — as grid
definitions change over time, configs from older grids pile up and go stale.

`dev/remote.py pull` is a plain `scp` of that whole file, so pulling results for
analysis of any single run drags down the entire (mostly stale) union.
Filtering machinery already exists — `results.filter_results` (by config name),
`results.select_results` (by case value), `results.slim_results` / `cxr slim`
(drop wide-brem, downcast, + value constraints) — but it all runs **after** the
pull, so it never reduces the bytes on the wire. Records carry no run-id /
timestamp / provenance, so "which records came from which run" is not recorded.

## Goals

1. Pull only the results relevant to a run, filtering **on the box before the
   transfer** so the wire cost shrinks.
2. Keep a durable local shelf so a good pull is never lost, with a two-tier
   active / long-term model.
3. Let stale accumulation on the box be cleared deliberately and safely.

## Definition of "a run"

"The desired results for a given run" = **the current grid**: exactly the config
set that `config.material_sweep(material)` → `sweep.build_cases` produces now.
Because `config.py` is deterministic and synced to the box, the box can
reconstruct the identical config-name set with no name-list crossing the wire —
this is `filter_results` semantics, computed remotely. Stale = any config whose
name is not in the current grid.

`--quick` grids are **out of scope**: they are defined inline in `scan.py`
(`run()`), not reproducible from `material_sweep(material)` alone, and are already
tiny.

## Lifecycle model

```
  GPU box (scratch compute)                 Laptop (durable store + viz)
  checkpoints/<stem>.pkl   --pull --grid-->  checkpoints/<stem>.pkl   (active slot)
  (full accumulated union)                        │  ▲
        │                                  archive │  │ restore
        │ clear                                    ▼  │
        ▼                                   checkpoints/archive/<label>.pkl
     (deleted)                                  (long-term shelf)
```

- **Box** — the live accumulator; ephemeral. Cleared deliberately.
- **Active slot** — `checkpoints/<stem>.pkl`, exactly what `analysis.ipynb`
  loads (`MATERIAL=stem`), populated by the grid-filtered pull.
- **Long-term shelf** — `checkpoints/archive/<label>.pkl`, named snapshots the
  user chooses to keep.

---

## Component 1 — Grid-filtered remote pull

Shrinks the transfer by filtering to the current grid on the box.

### `results.slim_results(..., grid=<material_key>)` (new kwarg)

In `src/cxr_mc/results/selection.py`. When `grid` is given, rebuild the current
grid's config-name set and keep only those configs, reusing `filter_results`:

```python
from ..config import default_settings, material_sweep
from ..sweep import build_cases

def _grid_names(material):
    settings = default_settings()
    sweep = material_sweep(material)
    cases = build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)
    return {c["name"] for c in cases}
```

- Orthogonal to and composable with the existing value `constraints`,
  `drop_wide_brem`, `downcast`. Applied as the first narrowing step (by name),
  then value constraints, then per-record trimming.
- Imports of `config`/`sweep` are made **inside the function** (or guarded) to
  avoid a new import cycle: `config` already imports `results`, so a
  module-level import of `config` from `results.selection` would be circular. A
  function-local import breaks the cycle.

### `cxr slim <ckpt> --grid`

In `src/cxr_mc/slim.py`. Add a `--grid` flag. When set, infer the material from
the checkpoint stem (the basename without `.pkl`; reject a `_quick` stem with a
clear "quick grids aren't grid-filterable" error) and pass `grid=<material>`
through `slim_checkpoint` → `slim_results`. Composable with `--drop-wide-brem` /
`--downcast`. Usable standalone on either machine.

### `dev/remote.py pull <stems> --grid [--drop-wide-brem] [--downcast] [--no-sync]`

- Default (standalone `pull --grid`): call `sync_code()` first so the box
  rebuilds the grid from the same `config.py` the laptop has (closes sync
  drift); `--no-sync` skips it.
- Per stem, over ssh: `{REMOTE_UV} run --no-sync cxr slim
  {REMOTE_DIR}/checkpoints/<stem>.pkl --grid [flags] -o /tmp/<stem>.grid.pkl`,
  then `scp` `/tmp/<stem>.grid.pkl` into local **`checkpoints/<stem>.pkl`**
  (the active slot), and `rm -f` the box temp.
- Without `--grid`, `pull` keeps its current whole-file behavior.
- `scan`'s trailing pull accepts the same flags and forwards them.
- `--grid` alone is **lossless per record** — only stale configs drop;
  byte-trimming (`--drop-wide-brem` / `--downcast`) stays opt-in.

---

## Component 2 — Local active / long-term archive

Pure local file operations, so they live on the **`cxr` console script** next to
`slim` (library-side, unit-testable), not in `remote.py`. New module
`src/cxr_mc/archive.py` mirroring `slim.py`'s `add_subparser` / handler shape,
wired into `cli.py`. Shelf dir: `checkpoints/archive/`.

- **`cxr archive <stem> [label]`** — copy `checkpoints/<stem>.pkl` →
  `checkpoints/archive/<label>.pkl`. Default `label = <stem>-<YYYYMMDD>`. Atomic
  temp+replace. Refuses to overwrite an existing label without `--force`.
- **`cxr restore <label> [--as <stem>]`** — copy `archive/<label>.pkl` back to
  the active `checkpoints/<stem>.pkl`. Default stem inferred from the label
  (strip a trailing `-<YYYYMMDD>`). Refuses to overwrite an existing active slot
  without `--force`.
- **`cxr archives`** — list the shelf: label, size (MB), record count
  (`sum(len(v) for v in results.values())`).

Archive stores whatever the active slot holds at the time — typically the
grid-filtered view.

---

## Component 3 — Remote clear

Delete a material's accumulated pickles on the box.

- **`dev/remote.py clear <material> [--yes]`** — over ssh, delete
  `checkpoints/<material>.pkl` **and** `checkpoints/<material>_quick.pkl` on the
  box.
- Without `--yes`: print exactly which of the two files exist and would be
  deleted, then stop (a safe dry preview) — no deletion.
- With `--yes`: `rm -f` both, report which were removed.
- **Refuses** (before any deletion) if a live job is producing either stem —
  reuses the `_refuse_if_busy` / `_live_jobs` guard already in `remote.py`, so a
  clear cannot yank a checkpoint out from under a running sweep.
- `<material>` validated by the existing `_check_materials` alphabet guard (it is
  interpolated into a remote shell command).

---

## Error handling

- Grid pull: if `cxr slim --grid` finds the box checkpoint missing, the ssh
  command exits nonzero and `_run`/`_ssh_capture` surface it; the `scp` is not
  attempted for that stem.
- `restore` of a missing label / `archive` of a missing active slot: clear
  "no such …" message, nonzero exit.
- `clear` with no matching files on the box: reports "(nothing to clear)".
- Overwrite guards (`--force`) on `archive`/`restore` prevent silent clobber of
  a shelf snapshot or the active slot.

## Testing

- `slim_results(grid=...)`: unit test that a checkpoint holding current-grid +
  extra stale configs slims to exactly the current-grid names; composes with
  `drop_wide_brem`/`downcast`/value `constraints`; a `_quick` stem is rejected.
- `cxr slim --grid`: CLI test on a temp checkpoint (round-trips through
  `run.load_checkpoint`).
- `archive`/`restore`/`archives`: round-trip on a temp `checkpoints/` tree,
  including `--force` guard behavior and label→stem inference.
- `remote.py clear`: unit-test the dry-preview vs `--yes` branch and the
  `_refuse_if_busy` refusal by monkeypatching `_live_jobs` (no real ssh), mirror
  the existing remote-command tests if present.

## Out of scope / non-goals

- Provenance/run-id stamping of records (a heavier record-schema change).
- Filtering `--quick` grids.
- Remote long-term storage (the shelf is local by design).
- Value-slice CLI flags on `cxr slim` (`--tilt`/`--energy`): value constraints
  remain programmatic via `slim_results(**constraints)`; only `--grid` is added
  to the CLI here.

## Build order

The three components are independent and can land as separate commits/PRs:
1. Grid-filtered pull (`slim_results(grid=)` → `cxr slim --grid` → `pull --grid`).
2. Remote clear.
3. Local archive/restore.
