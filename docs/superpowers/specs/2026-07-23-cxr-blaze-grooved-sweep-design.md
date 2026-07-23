# `cxr blaze` — grooved-crystal sweep driver

**Date:** 2026-07-23
**Status:** approved (design)
**Depends on:** blazed-groove-geometry feature (`montecarlo.groove`, `Sweep.groove_spacing_ang`, `sweep._reject_invalid_groove_geometry`) — already implemented and ledgered.

## Goal

A dedicated CLI subcommand that runs a Monte-Carlo CXR sweep over a crystal with a
**blazed sawtooth entrance face** (grooves), writing to a checkpoint kept separate
from the crystal's flat-face (`cxr scan`) checkpoint. Groove face angles are set
automatically by the existing geometry code (`montecarlo.groove.blazed_groove_spec`);
the driver only supplies groove spacing and the scan grid.

Motivating command:

```
cxr blaze hopg --spacing 2e-6 --energy 30 --angles 25 45
```

adds a HOPG crystal with 2 µm-spaced grooves, scanned over polar angles 25° and 45°.

## Non-goals

- No new physics. The groove kernel, its derivation docstring, and its validation
  ledger row already exist. This is CLI + sweep plumbing only.
- Grooves remain **v1 single-slab**: laterally infinite, no substrate/stack, no
  finite footprint, `theta_obs = 90°`, `tilt_azim = 180°`, `0 < tilt < 90°`. These
  are enforced by `sweep._reject_invalid_groove_geometry`; the driver constructs
  every Sweep to satisfy them and does not relax them.
- Groove spacing stays **scalar per Sweep** (not a sweepable `Sweep` field). Per-energy
  spacing is achieved by building one Sweep per (energy, spacing) pair, not by making
  the field a sequence.

## CLI surface

New module `src/cxr_mc/blaze.py`, mirroring `scan.py`'s structure
(`add_subparser(sub)` → `_build_parser(ap)` → `run(args)`), registered in
`cli.py` alongside the other subcommands.

```
cxr blaze <material> --energy E [E ...] --spacing S [S ...]
                     [--angles A [A ...]]
                     [--workers N] [--checkpoint-dir DIR]
                     [--max-minutes M] [--progress-file PATH] [--no-progress]
```

| Arg | Required | Meaning |
|-----|----------|---------|
| `material` | yes | catalog crystal key (positional, like `cxr scan`) |
| `--energy` | **yes** | beam energies in keV (one or more) |
| `--spacing` | **yes** | groove spacings in **meters** (one, or one per energy) |
| `--angles` | no | polar tilt_deg values. Omitted → std catalog `tilt_deg` grid |
| `--workers` | no | forwarded to `run_sweep` (auto; 0 = serial) |
| `--checkpoint-dir` | no | default `checkpoints` |
| `--max-minutes` | no | soft wall-clock budget (exit 75 if work remains), as `cxr scan` |
| `--progress-file` / `--no-progress` | no | same hidden hooks as `cxr scan` |

Both `--energy` and `--spacing` are required (deliberate: a grooved run is
meaningless without a spacing, and the user wants energy pinned explicitly rather
than defaulted).

### Spacing units

`--spacing` is in **meters** and converted to angstroms (×1e10) before reaching
`Sweep.groove_spacing_ang`. `2e-6` m → `2e4` Å → 2 µm, matching the motivating
command and the existing test fixtures (`spacing_ang=2.0e4`).

### Energy ↔ spacing pairing

- **Equal lengths** → zipped element-wise: `--energy 30 50 --spacing 2e-6 3e-6`
  yields the pairs (30 keV, 2 µm) and (50 keV, 3 µm).
- **Single spacing** → broadcast to every energy: `--energy 30 50 --spacing 2e-6`
  yields (30 keV, 2 µm) and (50 keV, 2 µm).
- **Any other length mismatch** → `SystemExit` with a clear message (e.g. 3 energies
  and 2 spacings).

## Sweep construction

For each (energy, spacing) pair, build one groove Sweep via
`config.material_sweep(material, **overrides)` with overrides forcing the groove
geometry:

```python
material_sweep(
    material,
    theta_obs_deg=90.0,
    energy_keV=[energy_keV],
    groove_spacing_ang=spacing_ang,
    tilt_deg=(angles if angles is not None else <std catalog tilt_deg>),
    tilt_azim_deg=180.0,
    crystal_width_mm=None,
    crystal_height_mm=None,
    substrate=None,
    stack=None,
)
```

`material_sweep` already pulls the material's `E_grid_line` / `E_grid_line_by_energy`
/ `E_grid_brem` and thickness grid from the catalog, so those are inherited
unchanged. When `--angles` is omitted the std catalog `tilt_deg` grid is reused; if
that grid contains a groove-illegal angle (0 or 90°) `build_cases` raises loudly and
the user supplies `--angles`.

`build_cases` is called for each pair; the resulting case lists are concatenated
into one flat list. Penetration-watchdog gating (`config.gate_cases_by_penetration`)
is applied to the concatenated list exactly as `cxr scan` does, and its summary
printed.

## Checkpoint

Written to an explicit path `checkpoints/{material}_grooved.pkl` (stem
`{material}_grooved`), passed to `run_sweep(checkpoint_path=...)` — the same
mechanism `cxr scan --quick` uses to avoid clobbering the real checkpoint. The
flat-face `checkpoints/{material}.pkl` is never touched. Resume filtering is per
`(name, E0_keV)`, so re-invoking picks up where a budget-limited run left off. The
sidecar `{material}_grooved.meta.json` manifest is written by `run_sweep` as usual.

## `build_cases` name change

Case `name` currently encodes thickness, polar tilt, azimuth, stack, footprint —
**not** groove spacing. Two runs at the same (energy, tilt) but different spacings
would therefore collide on `(name, E0_keV)` inside one checkpoint, and resume
filtering would silently drop the second.

Fix: in `build_cases`, when `sweep.groove_spacing_ang is not None`, append a groove
tag to `name`:

```
... pol=25 az=180 groove=2um
```

(spacing rendered in µm via `groove_spacing_ang / 1e4`, `:g` format). Guarded on
`groove_spacing_ang is not None`, so **ungrooved case names are byte-for-byte
unchanged** — no existing checkpoint or golden fixture is disturbed. This makes
different spacings coexist as distinct records in one `{material}_grooved.pkl`,
satisfying "different groove spacings for different energies within the same
checkpoint" (and also same-energy/different-spacing, should the user want it).

## Testing

`tests/test_blaze.py`:

1. **Arg parsing** — `--energy` and `--spacing` both required (missing either →
   `SystemExit`); meters→Å conversion (`2e-6` → `groove_spacing_ang == 2e4`).
2. **Pairing** — equal-length zip pairs energies with spacings; single spacing
   broadcasts; length mismatch (≠1, ≠N) raises.
3. **Forced geometry** — every built case has `tilt_azim_deg == 180`,
   `crystal_width_mm is None`, `crystal_height_mm is None`,
   `theta_obs_rad == deg2rad(90)`, and a `groove_spacing_ang` key; no `abs_layers`.
4. **Angles** — `--angles 25 45` → tilts {25, 45}; omitted → std catalog grid.
5. **Checkpoint stem** — driver targets `{material}_grooved.pkl`, not `{material}.pkl`
   (assert via a monkeypatched `run_sweep` capturing `checkpoint_path`).
6. **Name encodes spacing** — grooved case names contain `groove=<µm>um`; two pairs
   with different spacings produce disjoint names → coexist in one checkpoint.
7. **Regression** — an ungrooved `build_cases` name is unchanged (guard holds).

Existing `montecarlo`/`groove`/`sweep` behavior is exercised by their own suites;
`test_blaze.py` covers only the new driver + the name change.

## Files touched

- `src/cxr_mc/blaze.py` — new driver module.
- `src/cxr_mc/cli.py` — register `blaze.add_subparser(sub)` + docstring line.
- `src/cxr_mc/sweep.py` — groove tag in `build_cases` name (guarded).
- `tests/test_blaze.py` — new.
- `docs/repo_map.md` — pointer to the new module (repo convention).
