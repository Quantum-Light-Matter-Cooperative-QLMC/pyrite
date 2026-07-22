# `cxr line-grid` command group — design

Date: 2026-07-22
Branch: `feature/add-40-60kev-energies`
Status: approved (design), pending implementation plan

## Problem

Deriving per-material line-grid bounds and applying them to
`src/cxr_mc/data/materials.toml` is currently a three-part manual chore:

1. `scripts/line_grid_bounds_job.py` — submit/manage a sliced SLURM job on qlmc
   that runs the derivation.
2. `scripts/analyze_line_grid_bounds.py` — the derivation itself; emits a combined
   JSON `{material: {line_rows: [...], brem: {stop_eV, raw_eV, step_eV}}}`.
3. **No tool** for step three: a human reads that JSON and hand-edits
   `E_grid_line_by_energy` + `E_grid_brem` blocks in `materials.toml`. The
   provisional 40/60 keV placeholder rows (and their `# pending rerun` comments)
   are evidence of this manual, error-prone step.

There is also no CLI path to set or override a single material's grid, or to
record *why* a grid was chosen. Overrides today mean hand-editing the TOML.

## Goals

- Both scripts become first-class `cxr` subcommands under a new `cxr line-grid`
  group, parallel to `cxr remote`.
- End-to-end flow with **zero manual config-file editing**: submit → apply.
- Specify particular materials and/or beam energies. An energy not yet in the
  profile (e.g. 40, 60 keV before they existed) is **added** — to both the
  profile `energy_keV` values array (making it a real supported sweep energy)
  and the per-material grid tables.
- Manual grid overrides via CLI (not file editing), with optional notes.
- Job management (`status`/`attach`/`logs`/`stop`) shows the **same** output as
  `cxr remote status [-v/-vv]` etc. — same functions, same verbosity.

## Non-goals

- No change to the derivation physics or the coverage/margin/spacing math.
- No change to the SLURM slicing / staging / resume contract in `remote.py`.
- No blocking "submit-and-wait" mega-command; SLURM slices run long, so submit
  and apply stay separate (apply can auto-pull the result).

## Command surface

New top-level group `cxr line-grid`, registered in `cli.py` alongside the
others via a `line_grid.add_subparser(sub)`:

```
cxr line-grid derive   [--materials …] [--energies …] [geometry flags] [--set-default] [derivation flags]
cxr line-grid submit   [--materials …] [--energies …] [geometry flags] [--set-default] [--slice-minutes N] [--no-sync] [--dry-run]
cxr line-grid status   [jobid] [-v|-vv]
cxr line-grid attach   [jobid]
cxr line-grid logs     [jobid] [-f]
cxr line-grid stop     [jobid]
cxr line-grid apply    [json] [--materials …] [--pull] [--force] [--dry-run]
cxr line-grid set      <material> --energy E --stop S [--num N] [--start S0] [--note "…"]
cxr line-grid set-brem <material> --stop S [--step ST] [--note "…"]
cxr line-grid defaults [--set <flags>]        # show or update persistent defaults
cxr line-grid show     [material]
cxr line-grid regen-golden [--check]          # rebuild the independent catalog golden
```

Geometry flags (on `derive`/`submit`): `--tilts d1,d2,…` (diagnostic polar
tilts, deg), `--azimuths a1,a2,…` (azimuths, deg), `--thickness A1,A2,…`
(crystal thickness/thicknesses, Å). Each overrides the corresponding default
for **that run only**. Adding `--set-default` also persists the run's
geometry/energies/materials as the new standing defaults (see below).

- `derive` — run the derivation **locally** (thin wrapper over the current
  `analyze_line_grid_bounds.main` logic). Same flags: `--top-k`, `--coarse-ne`,
  `--refine-ne`, `--max-workers`, `--coarse-engine`, `--grid-stop`,
  `--grid-step`, `--brem-grid-stop`, `--json-out`, `--max-minutes`.
- `submit` — stage + submit the sliced SLURM job on qlmc via the existing
  `remote.py` contract (current `line_grid_bounds_job.start`). Reuses
  `remote.sync_code`, `_stage_job_script`, `_submit_staged_job`, `_new_jobid`,
  `_slurm_batch_script`.
- `status`/`attach`/`logs`/`stop` — **delegate directly** to `remote.job_status`,
  `remote.attach`, `remote._cli_logs`-equivalent, `remote._stop_jobid`. `status`
  takes `-v` as an `action="count"` and passes `detail=verbose` to
  `remote.job_status`, exactly as `cxr remote status` does — output is identical.
- `apply` — merge derived bounds into `materials.toml` (see below).
- `set` / `set-brem` — manual override of one material's line row / brem grid.
- `show` — print a material's current resolved grids with `source`/`note`.

### Streamlined end-to-end (no manual editing)

```
cxr line-grid submit --materials hopg,diamond --energies 40,60
cxr line-grid status -vv           # watch, identical to cxr remote status -vv
cxr line-grid apply --pull         # scp combined JSON back + merge into materials.toml
```

`apply --pull` reuses remote scp (as `cxr remote pull` does) to fetch the
combined `json_out` from qlmc, so the JSON is never touched by hand either.

## Module layout

Both scripts move **fully** into `src/cxr_mc/`, matching the repo convention
that every subcommand impl lives there (`remote.py`, `scan.py`, `export.py`,
`line_grid_bounds.py`); `scripts/` keeps only `dev.py`. New `line_grid` package:

```
src/cxr_mc/line_grid/
  __init__.py    # add_subparser + _cli_* handlers (the cxr line-grid group)
  derive.py      # derivation, from scripts/analyze_line_grid_bounds.py
  job.py         # remote submit + SLURM job-script builder, from scripts/line_grid_bounds_job.py
  apply.py       # materials.toml write-back; set / set-brem / show
  defaults.py    # line_grid_defaults.toml read/write
  golden.py      # regen-golden (independent serializer)
```

`__init__.py` delegates job management to `remote.py` (status/attach/logs/stop)
and calls `derive`/`job`/`apply` for the rest. The existing math helpers in
`src/cxr_mc/line_grid_bounds.py` (`coverage_energy`, `margined_stop`,
`spacing_num`) fold into the package as `line_grid/bounds.py` (rename; update its
importers).

**Scripts deleted**: `scripts/line_grid_bounds_job.py` and
`scripts/analyze_line_grid_bounds.py` are removed. Two consequences to handle:

1. **Remote SLURM payload** (`job.py`'s `_slice_payload`) currently runs
   `uv run --no-sync python scripts/analyze_line_grid_bounds.py …` **by path on
   qlmc**. It switches to invoking the installed module —
   `uv run --no-sync python -m cxr_mc.line_grid.derive …` (same repo is synced to
   qlmc, so the module is importable). The derive module keeps a
   `python -m cxr_mc.line_grid.derive` entry (`__main__` / `main(argv)`) with the
   identical CLI flags the payload passes.
2. **Tests**: `tests/test_line_grid_bounds_job.py` and
   `tests/test_analyze_line_grid_bounds.py` update their imports from
   `scripts...` to `cxr_mc.line_grid.job` / `cxr_mc.line_grid.derive`; behaviour
   assertions unchanged. Rename the test files to match if that follows repo
   convention.

`cxr line-grid derive` (local) and `python -m cxr_mc.line_grid.derive` (remote
payload) share the same `derive.main`, so local and remote runs stay identical.

## `apply` write-back — surgical block regeneration

No new dependency (only `tomllib`, read-only, exists today). The tool **owns**
three managed regions of `materials.toml` and regenerates only those, byte-for-
byte preserving everything else (comments, unrelated blocks, ordering):

1. Per-material `E_grid_line_by_energy = [ … ]` block.
2. Per-material `E_grid_brem = { arange = { … } }` line.
3. `[profiles.standard] energy_keV = { values = […] }` — newly requested
   energies inserted in sorted numeric order.

Mechanism: locate `[materials.<key>]` (and `[profiles.standard]`) section spans
by text scan, find the managed assignment within, replace its span with freshly
emitted text. Emitted rows keep the existing **one-inline-table-per-line**
format for clean diffs:

```toml
  { energy_keV = 40.0, grid = { linspace = { start = 10.0, stop = 3000.0, num = 998, endpoint = true } }, source = "derived job 458 (2026-07-22)" },
```

Provisional hand-comments (40/60 placeholders) inside a regenerated block are
replaced by the real derived rows — intended cleanup.

Rows are computed from the combined JSON exactly as today's manual translation
does: `stop_eV`, `num`, `start_eV` from each `line_rows` entry; brem
`stop`/`step` from the `brem` descriptor. `start`/`endpoint` conventions match
the current TOML (`start=10.0` at ≤60 keV, `start=50.0` above, `endpoint=true`).

`--dry-run` prints the unified diff without writing. After writing, `apply`
re-loads the catalog to validate (fail loudly if the regenerated TOML doesn't
parse).

## Notes + provenance (schema addition)

To make CLI-managed overrides and notes first-class (not comments that die on
regeneration), each managed grid entry carries optional structured metadata:

- Line row allowed keys extend from `{energy_keV, grid}` to
  `{energy_keV, grid, source, note}` in
  `src/cxr_mc/materials/_catalog_decode.py::_line_grids_by_energy`
  (line 167 `errors.keys(...)`).
- The brem grid tolerates optional `source`/`note` alongside its `arange`/
  `linspace` payload (via the `_grid` allowed-keys check,
  `_catalog_decode.py:109`).
- **Both fields are parsed then ignored by physics** — they never reach a
  `ScanSpec` grid array. They exist for provenance/display only.

Provenance stamping:

- `apply` derived rows: `source = "derived job <slurm_id> (<date>)"`.
- `set` / `set-brem`: `source = "manual"`, plus the user's `--note` string.

### Sticky manual overrides

`apply` will **not** overwrite any row/grid whose `source = "manual"` unless
`--force` is passed. This protects hand-tuned grids from being clobbered by a
later rerun. `apply` reports which rows it skipped for this reason; `show` lists
each material's `source`/`note` so overrides are auditable.

## Diagnostic geometry & thickness overrides + persistent defaults

The derivation samples a geometry × thickness set to find each material's
worst-case coverage. Today these are effectively hardcoded:

- **Thickness**: pinned to `DIAGNOSTIC_THICKNESS_ANG = 1e7` Å (1 mm, thickest
  slab = high-energy worst case) in `analyze_line_grid_bounds.py`.
- **Angles**: `_geometry_plan` reads the reference material's profile
  `tilt_deg` / `tilt_azim_deg` arrays (quantized) — the full tilt × azimuth
  product.

New behaviour:

- `--tilts`, `--azimuths`, `--thickness` override each per run. `_geometry_plan`
  and `_build_case` gain explicit tilt/azimuth/thickness parameters; when a flag
  is unset the value falls back to the persistent default, which itself defaults
  to today's behaviour (angles from the profile scan, thickness `1e7`). This
  preserves current output when no flag is given.
- `--thickness` accepts a comma list; each thickness becomes an extra axis in the
  diagnostic geometry product (the worst coverage across all sampled thicknesses
  drives the bound). A single value reproduces today's single-slab scan.

### Persistent defaults file

Defaults live in a **tool-owned** `src/cxr_mc/data/line_grid_defaults.toml`,
fully owned by the `line-grid` tool — not part of the material catalog, so no
catalog schema change and no golden-fixture churn. It holds:

```toml
# managed by `cxr line-grid defaults --set` / `--set-default`; edit via CLI
tilts = []            # [] = use each material's profile tilt_deg (today's default)
azimuths = []         # [] = use profile tilt_azim_deg
thickness_ang = [1.0e7]
brem_step_ev = 25.0   # default E_grid_brem step; --step overrides per-run/-set
energies = [30, 40, 50, 60, 100, 150, 200, 250, 300]
materials = ["hopg", "diamond", "wse2", "mose2"]
```

`set-brem --step` overrides the brem grid step for one write; unset it uses
`brem_step_ev` from this file. The default is changeable via
`cxr line-grid defaults --set --brem-step …` (or `--set-default` on a run that
passed `--brem-step`), same partial-merge/atomic-write path as the other
defaults.

- Runs read this file for their defaults (replacing the current module-level
  `DEFAULT_*` constants). Missing file → built-in fallbacks equal to today's
  values.
- `cxr line-grid defaults` prints the current defaults.
- `cxr line-grid defaults --set --tilts … --thickness …` (or `--set-default` on a
  `derive`/`submit` run) writes them. Written via the same atomic-write helper;
  a stamped `# last set: <date>` comment records provenance.
- `--set-default` persists **only the flags supplied on that run** merged over the
  existing file (a run that sets just `--thickness` leaves `tilts`/materials
  untouched).

## `set` / `show`

- `cxr line-grid set hopg --energy 60 --stop 3800 --num 1264 --note "widened for
  detector X tail"` — regenerates just that one line row via the same write-back
  path, stamped `source="manual"`. `--num`/`--start` optional; if `--num`
  omitted, computed from `start`/`stop` via the existing `spacing_num` at the
  standard 3 eV spacing.
- `cxr line-grid set-brem hopg --stop 140000 --note "…"` — manual brem override.
- `cxr line-grid show hopg` — prints the resolved line rows + brem grid with
  their `source`/`note`, plus a `(manual)` / `(derived …)` tag per row.

## Golden config regeneration

`tests/data/material_catalog_golden.json` is the **independent** oracle for
`test_packaged_catalog_matches_independent_serialized_golden` — it captures each
material's resolved grids (energies, tilts, `E_grid_line_by_energy`, brem, …) as
fingerprints. `apply` / `set` / `set-brem` change real bound values, so the
resolved grids move and the golden must be refreshed. Today that refresh is
manual; this adds a command.

`cxr line-grid regen-golden` rebuilds the golden and writes it. `--check`
regenerates in memory and diffs against the checked-in file, exiting nonzero on
drift (for CI / pre-commit) without writing.

**Independence requirement**: the golden is only meaningful if it is *not* a
straight dump of the runtime `CATALOG`. The regenerator must serialize by
parsing `materials.toml` directly (raw `tomllib`) and computing the grid
fingerprints through its own path — never by calling the resolver under test.
The implementation reuses the existing independent serializer if one is found;
otherwise it adds one under `scripts/`. (Implementation plan resolves which.)

Wiring: `cxr line-grid apply` accepts `--regen-golden` to invoke it right after
a successful write; without the flag, `apply` prints a one-line reminder that the
golden is now stale. A matching `scripts/dev.py regen-golden` alias keeps it in
the dev-tooling surface next to `lint`/`format`/`verify`.

## Testing

- `apply` merge/write-back: golden-in / golden-out TOML fixtures — derived JSON +
  starting TOML → expected TOML; assert other blocks untouched, new energy
  inserted into profile array, sticky-manual skip, `--force` override, `--dry-run`
  emits diff without writing, regenerated TOML re-parses.
- `set` / `set-brem`: single-row regen, `source="manual"` + note stamped,
  `--num` auto-compute path.
- Schema tolerance: catalog parses `source`/`note` on line rows and brem grid,
  ignores them in resolved `ScanSpec`. Regenerate
  `tests/data/material_catalog_golden.json` and refresh
  `tests/test_material_catalog.py` expectations.
- Status parity: `cxr line-grid status [-v/-vv]` calls `remote.job_status` with
  the same `detail` — assert delegation (mock `job_status`, check `detail` arg).
- Geometry overrides: `--tilts/--azimuths/--thickness` reach `_geometry_plan` /
  `_build_case`; unset flags reproduce today's profile-angle + 1e7-thickness scan
  (regression); multi-thickness expands the geometry product.
- Defaults file: `defaults --set` / `--set-default` round-trip (partial merge,
  atomic write, missing-file fallbacks equal today's constants, `brem_step_ev`);
  `derive`/`submit`/`set-brem` read defaults from it.
- `regen-golden`: rebuild reproduces the current checked-in golden byte-for-byte
  from an unchanged catalog; `--check` exits nonzero on injected drift and does
  not write; regenerator does not import the runtime resolver.
- Keep existing `tests/test_line_grid_bounds_job.py` /
  `tests/test_analyze_line_grid_bounds.py` green through the thin shims.

## Risks / mitigations

- **Text-surgery brittleness**: block boundaries located by naive scan could
  mis-parse. Mitigation: after every write, re-load the catalog and fail if it
  doesn't parse or a targeted material's resolved grid doesn't match intent.
- **Golden churn**: schema addition forces golden regen. Contained, expected.
- **Remote invocation switch**: the SLURM payload moves from a path-invoked
  script to `python -m cxr_mc.line_grid.derive`. Mitigation: keep the module's
  CLI flags byte-identical to the old script's; a `--dry-run` submit asserts the
  generated payload command in tests before any live qlmc run.
