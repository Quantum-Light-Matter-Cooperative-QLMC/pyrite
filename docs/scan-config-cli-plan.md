# Plan: CLI scan-parameter reconfiguration, per-material overrides, multi-config checkpoints

Status: planned (decisions locked 2026-07-26). Owner: Alex.

## Decisions (made up front)

1. **Multi-config checkpoints — Option A, per-set stems.** Each named
   parameter-set writes its own pickle: `checkpoints/<material>@<set>.pkl`.
   The catalog-default (or persisted-override) grid keeps `<material>.pkl`.
   Generalizes the existing `_quick` → `<material>_quick.pkl` pattern; no
   pickle schema change; `archive`/`restore`/`union`/`slim` already operate
   per stem. Rejected: nested `{set: {config: {E0: record}}}` pickle (schema
   migration across results/plots/run/slim/archive/union + both marimo apps)
   and config-name fingerprints (opaque names, fingerprint-aware selection
   everywhere).
2. **Persisted overrides — root-level TOML.** `cxr-overrides.toml` at the
   repo root (gitignored), managed by a new `cxr config` subcommand. The
   packaged `data/materials.toml` and the immutable `CATALOG` are never
   touched.
3. **Session overrides — dedicated flags + generic `--set`.** Common knobs
   get explicit `cxr scan` flags; everything else goes through repeated
   `--set key=value` using the same grid grammar as the TOML.

## Motivating gap

Scan grids are projections of the immutable `CATALOG`
(`config.material_sweep`). The CLI exposes only `--quick`, `--n-families`,
`--beam-uvw`. Worse, the checkpoint store is `{config_name: {E0: record}}`
and the config name (`"{label} {thickness} pol=.. az=.."`) does NOT encode
energy grids, `n_families`, `beam_uvw`, or electron counts — so running two
different parameter-sets into one pickle silently resume-skips or mixes
records. Per-set stems close that hole.

## Override model

Precedence (low → high):

```
packaged materials.toml (CATALOG)
  < cxr-overrides.toml [defaults]          # broad reconfiguration
  < cxr-overrides.toml [materials.<key>]   # per-material custom set
  < cxr scan session flags / --set         # this invocation only
```

`cxr-overrides.toml` schema (grid grammar identical to `materials.toml`:
`values` / `arange` / `linspace` / `logspace`):

```toml
schema_version = 1

[defaults]                       # applies to every material
tilt_deg = { linspace = { start = 0.0, stop = 89.0, num = 19, endpoint = true } }
n_families = 6

[materials.mose2]                # custom-set material
thickness_ang = { values = [1e4, 5e4] }
energy_keV = { values = [50.0, 100.0] }
beam_uvw = [0, 0, 2]
```

Allowed keys: the `ScanSpec` grid fields (`thickness_ang`, `energy_keV`,
`tilt_deg`, `tilt_azim_deg`, `E_grid_line`, `E_grid_line_by_energy`,
`E_grid_brem`) plus the scalar `Sweep` knobs already CLI-adjacent
(`n_families`, `beam_uvw`, `theta_obs_deg`, `mosaic`, `mosaic_fwhm_deg`) and
the `Settings` counts (`n_electrons`, `n_electrons_brem`,
`beam_current_na`). Unknown keys or material names fail loud at load, reusing
the catalog's grid validators (factor the `_GRID_KINDS` resolution out of
`materials/catalog.py` into a shared helper rather than duplicating it).

File location resolution: `./cxr-overrides.toml`, overridable via
`CXR_OVERRIDES` env var and `--overrides-file`. Add to `.gitignore`.

## New/changed CLI surface

### `cxr config` (new subcommand, `src/cxr_mc/user_config.py`)

- `cxr config set [-m MATERIAL] KEY VALUE` — persist an override
  (`[defaults]` without `-m`, `[materials.<key>]` with). VALUE accepts the
  compact grammar below.
- `cxr config get [-m MATERIAL] [KEY]` — show one value or the whole table.
- `cxr config unset [-m MATERIAL] KEY` / `cxr config clear [-m MATERIAL]`.
- `cxr config list` — full effective-override dump.
- `cxr config materials` — **the "which materials are custom-set" listing**:
  one row per `[materials.<key>]` table with its overridden keys.
- `cxr config diff [MATERIAL]` — effective grid vs pristine catalog grid.

Writes are atomic (temp + `os.replace`, matching `_checkpoint_save`), file
round-trips through `tomllib` for validation before replace.

### `cxr scan` (extended)

- Dedicated flags: `--thickness`, `--energies`, `--tilts`, `--azimuths`
  (comma-separated values or `linspace:start:stop:num` /
  `arange:start:stop:step` / `logspace:start:stop:num`).
- Generic escape hatch: `--set KEY=VALUE` (repeatable), same grammar; covers
  every allowed key above without new argparse per field.
- `--preset NAME` — names the parameter-set; checkpoint goes to
  `checkpoints/<material>@NAME.pkl`.
- `--no-overrides` — ignore `cxr-overrides.toml` for this run (pristine
  catalog grid).
- `--overrides-file PATH`.

Stem rule: no session grid changes → `<material>.pkl` (persisted overrides
count as "the defaults" — that is their purpose). Any session-flag/`--set`
grid change without `--preset` → write `<material>@adhoc.pkl` and print a
warning naming the stem; with `--preset` → `<material>@<preset>.pkl`.
`--quick` keeps `_quick` behavior unchanged. Preset names validated
`[a-z0-9-]+` to keep stems filesystem- and `archive`-label-safe.

### Provenance sidecar

Every scan writes `<stem>.meta.json` next to the pickle: resolved sweep
fields (grids as lists), settings counts, override sources used
(catalog/defaults/material/session), timestamp, git rev. No pickle schema
change; consumers that iterate `results.values()` are untouched.
`cxr checkpoints` and `cxr config diff` read it.

### `cxr checkpoints` (new, small)

List active checkpoint stems: material, set name, record count, and grid
summary from the sidecar (or "no meta" for legacy pickles).

## Consumer awareness (per-set stems)

- `run.py` — no change: `scan.py` already passes an explicit
  `checkpoint_path`.
- `analyze.material_menu` — also glob `<material>@*.pkl`; menu label
  `"{label} ({set})"`; `load_checkpoint` gains stem passthrough (it already
  takes an arbitrary stem string).
- `slim`, `archive`, `restore`, `union` — already stem-based; verify with
  tests that `@` stems round-trip (archive label inference, union same-
  material check via `case["crystal"]` still works since the crystal is
  unchanged across sets).
- `remote` — thread the new scan flags through submit command construction
  and pull the `@`-stem checkpoint + sidecar back (final slice).
- Marimo `scan_app.py` — out of scope beyond continuing to work; it uses
  `material_sweep`, so persisted overrides apply automatically (a follow-up
  can add preset UI).

## Implementation slices (each independently verifiable)

1. **Override layer.** `user_config.py`: file discovery, parse, validation
   (shared grid-resolver factored from `materials/catalog.py`), and
   `apply_overrides(material) -> (sweep_overrides, settings_overrides)`.
   Wire into `config.material_sweep` / `default_settings` behind an
   `overrides=True` toggle. Tests: grammar, precedence, unknown-key
   rejection, no-file no-op.
2. **`cxr config` subcommand.** All verbs incl. `materials` and `diff`;
   atomic writes. Tests: CLI round-trip via `main([...])`, list/materials
   output.
3. **`cxr scan` session flags + preset stems + sidecar.** Flag parsing to
   overrides dict, stem rule, `meta.json` writer. Tests: stem derivation
   matrix (default/adhoc/preset/quick), flag→Sweep equivalence with
   programmatic `material_sweep(**overrides)`.
4. **Consumers + `cxr checkpoints`.** analyze menu `@`-stems, archive/slim/
   union `@`-stem tests, new listing command.
5. **Remote passthrough + docs.** Thread flags/stems through `remote`;
   update `docs/repo_map.md` entries (new `user_config.py`, changed CLI
   dispatch list), README CLI table, TODO.md sync; regenerate repo-map
   inventory.

Verification per slice: `scripts/dev.py test` (targeted files), then full
`verify` at the end. No physics touched → no validation-ledger entries; but
`E_grid_*` overrides pass through the same positivity/finiteness validators
as the catalog so a bad user grid cannot reach `mc_spectrum`.

## Explicit non-goals

- No nested-pickle schema change; legacy checkpoints load unchanged.
- No editing of packaged `materials.toml` at runtime.
- No marimo preset-selection UI (follow-up).
- No cross-set merged analysis view (load two stores manually or use
  `union` semantics later).
