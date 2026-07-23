# `cxr reline` + dataset-partial pull — design

## Problem

`cxr rebrem` recomputes ONLY the bremsstrahlung portion of an existing
per-material checkpoint (rewrites `brem_wide` / `brem` / `E_grid_brem`, leaves
the expensive line `spec` untouched), and `cxr remote rebrem` submits it to the
GPU box and pulls the result. There is no symmetric command for the **line**
side: to re-run coherent line spectra with new line-grid bounds (the bespoke
per-material `E_grid_line_by_energy` derivation workflow) or a higher line `Ne`,
the only path today is a full sweep, which needlessly recomputes brem too.

Second gap: `cxr remote pull` (and the rebrem auto-pull) replace the local
`<stem>.pkl` **whole-file**. A checkpoint record holds two independent datasets
— the line spectrum and the brem background — in the same pickle. Whole-file
replace means a rebrem pull clobbers any newer local line spectrum, and vice
versa. There is no way to pull just one dataset and overwrite only that portion
of the local pickle.

## Goals

1. `cxr reline` — recompute ONLY the line `spec` of existing checkpoints (new
   line grid and/or line `Ne`), leaving brem untouched. Local, and via
   `cxr remote reline` (SLURM submit + attach dashboard + pull), mirroring
   `rebrem` exactly.
2. Dataset-partial pull: pull just the brem dataset or just the line dataset and
   overwrite **only that portion** of the local pickle, in place, leaving the
   other dataset's records intact — for one, multiple, or all materials.
3. Both `rebrem` and `reline` results individually pullable per material set.

## Non-goals

- No change to the live full sweep (`cxr scan` / `cxr remote scan`).
- No new checkpoint schema. Datasets are the existing record keys, grouped.
- No cross-material union semantics beyond what exists (`union_checkpoint`).

## Record dataset groups

A record `results[config_name][E0]` is a dict. The two datasets:

| Dataset | Record keys | `case` keys |
|---------|-------------|-------------|
| **line** | `spec`, `E_grid` | `Ne`, `E_grid_line`, `E_cut_lines_keV` |
| **brem** | `brem_wide`, `brem`, `E_grid_brem` | `Ne_brem`, `E_grid_brem`, `brem_step_eV` |

`E_grid` is the line grid and is **shared**: `brem` is `brem_wide` interpolated
onto `E_grid`. So a line-only change to `E_grid` leaves `brem` stale — the merge
and the recompute both re-interp `brem` from the retained `brem_wide` (cheap, no
brem transport). `brem_wide` on `E_grid_brem` is the authoritative brem array.

The keys defining each group live in one place — a `LINE_KEYS` / `BREM_KEYS`
constant pair (record-level and case-level) in `cxr_mc/results.py` — consumed by
both the projection (C) and the merge (D) so they cannot drift.

## A. Line recompute core — mirror of the brem side

`_spectrum_case` (runner.py) computes line + brem from transported segments. Its
line block is factored into a reusable helper, mirroring how the brem block was
factored into `_brem_wide_from_segments`:

- `runner._lines_for_case(case, E_grid) -> spec` — new, mirror of
  `_brem_for_case`. Builds tilted geometry + optional groove, transports `Ne`
  electrons at `seed` (line seed, NOT `seed + 1`), runs `mc_spectrum` with the
  case's `crystal` / `hkl_list` / `B_ang2` / `layer_radiators` / mosaic / groove
  / `layers=abs_layers`, onto `E_grid`. Reuses the EXACT live-sweep line path so
  a multilayer/mosaic/grooved record relines identically to a fresh sweep.
  `_spectrum_case`'s line block becomes a call to the shared helper (same
  refactor discipline as the brem factor-out).

- `run.repair_line_spec(results, only_stale=True, line_ne=None, line_step_eV=None,
  from_config=True, redo_all=False, save_every=0, save_cb=None, on_progress=None)`
  — mirror of `repair_brem_wide`. Per selected record:
  1. Determine the target line grid:
     - `line_step_eV` given → uniform grid at that spacing over the record's
       current `[E_grid[0], E_grid[-1]]` (explicit override).
     - else `from_config` (default) → re-derive from the material's CURRENT
       `ScanSpec` via the same `E_grid_line_by_energy[E0]` lookup `build_cases`
       uses (`sweep._line_grid_for_energy`), keyed by the record's material stem
       + `E0_keV`. This is the bespoke-per-material-grid workflow: edit bounds in
       `materials.toml`, `cxr reline MoS2`, lines re-run on the new grid.
     - else → the record's existing `E_grid` (pure `Ne` bump).
  2. `line_ne` overrides `case["Ne"]` if given.
  3. `spec = _lines_for_case(case, E_grid_new)`; write `r["spec"]`,
     `r["E_grid"] = E_grid_new`, `r["case"]["E_grid_line"]` (encoded).
  4. Re-interp brem onto the new grid: `r["brem"] = interp(E_grid_new,
     E_grid_brem, brem_wide)`. `brem_wide` / `E_grid_brem` untouched.
  - Selection (`only_stale`): skip records already at target — same grid + same
    `Ne` (and finite `spec`) — so an interrupted run is resumable. `redo_all`
    forces every record. Params persisted into `case`, so re-runs skip.
  - `save_every` / `save_cb` / `on_progress`: identical contract to
    `repair_brem_wide` (feeds `--progress-file` → remote dashboard).
  - Returns count relined; mutates in place.

- `run.reline_checkpoint(path, save_every=100, **kw)` — mirror of
  `repair_checkpoint`: load, `repair_line_spec`, atomic re-pickle in place +
  sidecar manifest, resumable.

The material stem needed for the config lookup is the checkpoint stem, threaded
in from `reline_checkpoint` (rebrem doesn't need it; reline does).

## B. `cxr reline` CLI

New module `cxr_mc/reline.py`, structural mirror of `rebrem.py`:

- `reline_checkpoints(materials=None, checkpoint_dir="checkpoints", line_ne=None,
  line_step_eV=None, redo_all=False, save_every=100, progress_file=None)` —
  mirror of `rebrem_checkpoints` (same `--all` = every non-`.slim.pkl`, same
  single-material `progress_file` guard, same `_write_progress_record` wiring).
- `add_subparser(sub)` registers `reline`:
  - `material` (nargs `*`), `-a/--all`
  - `--line-ne` (new line electron count)
  - `--line-step` (explicit uniform line-grid spacing [eV]; default: rebuild from
    the material's current `E_grid_line_by_energy` config)
  - `--redo-all`, `--checkpoint-dir`, hidden `--progress-file`, `--save-every`

Registered alongside `rebrem` wherever subcommands are wired.

```
cxr reline MoS2                     # re-run lines on current-config grid
cxr reline MoS2 --line-ne 40000     # bump line Ne, same grid
cxr reline --all --line-step 5      # explicit 5 eV line grid, every checkpoint
```

## C. Dataset projection (box-side, for transfer)

`results.project_dataset(results, dataset)` (`dataset` in `{"line","brem"}`) —
returns a new nested results dict where each record carries ONLY that dataset's
record keys (nesting `config_name -> E0` preserved for matching). `spec`/`brem`
arrays that don't belong to the dataset are dropped, shrinking the wire payload
(a brem-only projection is tiny; a line-only projection is the big `spec`).

`cxr slim` gains `--brem-only` / `--line-only` (mutually exclusive, and with the
existing trims), delegating to `project_dataset`.

## D. Local dataset merge

`results.merge_dataset(local, incoming, dataset, force=False) -> (n_merged,
n_skipped)`:

- For each `(config_name, E0)` in `incoming`:
  - present in `local` → overwrite ONLY that dataset's record keys (+ the
    dataset's `case` sub-keys). For `dataset="line"` whose incoming `E_grid`
    differs from local, re-interp local `brem` from the retained local
    `brem_wide` onto the new `E_grid` (keeps brem consistent without a brem
    array in the incoming projection).
  - absent from `local` → **skip + warn** by default (report the count);
    `force=True` inserts the incoming record as-is (caller is told a partial
    record will be half-populated for the non-pulled dataset).
- Pure in-memory; caller owns load/save/atomicity.

## E. Dataset-partial pull

`lifecycle.pull(stems, ..., dataset=None, force=False)`:

- `dataset is None` → unchanged whole-file / slim pull.
- `dataset` set → for each stem:
  1. box runs `cxr slim <ckpt> --{dataset}-only -o <tmp>` (sync first, so the box
     projects with the same key groups),
  2. scp `<tmp>` local, `rm` the box tmp,
  3. **archive the local active slot first** (`archive.archive_checkpoint`) so
     `cxr restore` undoes the merge,
  4. load local + incoming, `merge_dataset(local, incoming, dataset, force)`,
     atomic re-pickle + manifest,
  5. print `merged N / skipped M -> checkpoints/<stem>.pkl`.
- The whole-file digest fast-path does not apply to a partial merge (local file
  legitimately differs); skipped for `dataset` pulls.

## F. Remote reline queue

Mirror the rebrem remote path exactly:
- `scripts._reline_queue_script` / `_reline_queue_metadata` (payload runs
  `cxr reline <mat> --progress-file ...` with the line flags).
- `lifecycle.start_reline_queue(materials, line_ne=, line_step_eV=, redo_all=,
  no_sync=, dry_run=)` — reserves the same stems (must not race a scan/rebrem of
  the same material), submits, prints the attach/status/pull panel.
- `_cli_reline(args)` in `_remote/cli.py` — `_selected_materials` (one/multiple/
  `--all`), submit, attach, `state._completed_materials`, then
  `lifecycle.pull(completed, dataset="line")`.
- `_build_remote_parser` registers `reline` mirroring `rebrem`'s flags
  (`--line-ne`, `--line-step`, `--redo-all`, `--all`, `--no-sync`, `--dry-run`).

## G. rebrem auto-pull + `cxr remote pull` flags

- `_cli_rebrem`'s trailing pull switches to `lifecycle.pull(completed,
  dataset="brem")` — a rebrem now merges only brem into the local pickle,
  preserving any local line spectrum.
- `cxr remote pull` gains `--brem-only` / `--line-only` (mutually exclusive; map
  to `dataset=`) and `-f/--force`, so any prior `rebrem`/`reline` result is
  individually re-pullable per material set:

```
cxr remote pull MoS2 W --brem-only     # merge only brem for two materials
cxr remote pull --all --line-only      # merge only line spec, every material
```

## Testing

- `test_run.py`: `repair_line_spec` skips at-target records; reline changes
  `spec`/`E_grid` and re-interps `brem` but leaves `brem_wide`/`E_grid_brem`
  byte-identical; `--line-step` vs from-config grid selection; resumability.
- `runner`: `_lines_for_case` delegates a multilayer/mosaic/grooved case through
  the same path as a live sweep (monkeypatch `mc_spectrum`, assert args), mirror
  of `test_repair_brem_wide_delegates_stacked_case_to_runner`.
- `results`: `project_dataset` drops the other dataset's keys; `merge_dataset`
  overwrites only the target keys, skips unmatched (and inserts under `force`),
  and re-interps brem on a changed line grid.
- `test_remote.py`: `_cli_reline` submits + pulls `dataset="line"`; rebrem
  auto-pull now `dataset="brem"`; `pull(dataset=...)` archives before merge and
  writes a merged (not whole-file-replaced) pickle. Dry-run script snapshots.
- `test_slim.py`: `--brem-only` / `--line-only` projection round-trips.

## Symmetry table

| Concern | brem (exists) | line (new) |
|---------|---------------|------------|
| per-case recompute | `_brem_for_case` | `_lines_for_case` |
| in-place repair | `repair_brem_wide` | `repair_line_spec` |
| checkpoint driver | `repair_checkpoint` | `reline_checkpoint` |
| CLI | `cxr rebrem` | `cxr reline` |
| remote queue | `start_rebrem_queue` | `start_reline_queue` |
| remote CLI | `_cli_rebrem` | `_cli_reline` |
| pull dataset | `dataset="brem"` | `dataset="line"` |
