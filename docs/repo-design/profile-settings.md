# Profile settings reference

Every key a `[profiles.NAME]` table accepts, with its editing path. "TOML" means
no CLI setter exists: edit `profiles/NAME.toml` in the catalog directory and run
`pyrite material validate CATALOG_DIR` (catalog commands also validate on load). The
workflow guide is [Sweep profiles](../guides/sweep-profiles.md); the full schema
is [Materials catalog schema](materials-catalog-schema.md).

All keys below except `materials` membership bookkeeping are result-affecting:
they enter the resolved sweep and therefore dataset identity, so a change gives
new checkpoint stems and earlier results stay under their old identity.

## Ranges and membership

`pyrite profile show NAME` presents Setting / Value tables grouped by editing
command: sweep/membership/emission, beam, detectors, filters, numerics
(sampling/convergence/transport), precision, line-grid, energy grids and material
overrides. Numerics name their value source; adaptive
sampling delegates `line-trials` and `brem-trials` to the precision policy.
Long values wrap in the default table output; `--output wide` keeps each value
on one line. `--output json` retains the stable machine payload.

| TOML key | Type / unit | Setter | Reset / notes |
|---|---|---|---|
| `materials` | material keys; absent = all | `profile set\|add\|remove --material` | `profile set --all-materials` |
| `thickness_ang` | grid, Angstrom | `profile set\|add\|remove --thickness` | exactly one of `thickness_ang`/`thickness_layers` |
| `thickness_layers` | grid, layer count | TOML | stacks only |
| `energy_keV` | grid, keV | `profile set\|add\|remove --energy` | |
| `tilt_deg` | grid, deg | `profile set\|add\|remove --polar` | |
| `tilt_azim_deg` | grid, deg | `profile set\|add\|remove --azimuth` | |
| `E_grid_brem` | grid, eV | TOML (`pyrite-dev energy-grid brem set` stores per-material artifacts instead) | uniform grids keep only `step` |
| `E_grid_line` | grid, eV | TOML (`pyrite-dev energy-grid line` maintains artifacts) | ignored while a line-grid policy is set |
| `energy_grid_refs` | material -> artifact digest | `pyrite material energy-grid derive`, `pyrite-dev energy-grid add` | ignored while a line-grid policy is set |
| `overrides.MATERIAL.*` | any range/grid key above, plus counts | none: `pyrite material set` was removed in 0.6.0 (#359); use a single-material profile | Remove existing override keys by editing the TOML. Bundled stack rows (`thickness_layers`, sapphire thickness) stay until materials and physical objects are separated |

## Calculation numerics

`pyrite profile numerics show|set|reset NAME` owns these; `show` reports
explicit value, effective value and source (`profile`, `default`, `built-in`)
in human output. The JSON payload retains its existing source spellings.

| TOML key | Setter flag | Notes |
|---|---|---|
| `n_electrons`, `n_electrons_brem` | `--line-trials`, `--brem-trials` | single-value grids; `profile set\|add -l/-b` edit them as sweepable grids; old electron options warn and are scheduled for removal in 0.8.0 |
| `n_families`, `max_reflections` | `--reflection-families`, `--maximum-reflections` | |
| `mosaic_nodes`, `mosaic_route` | `--mosaic-nodes`, `--mosaic-route` | |
| `straggling`, `energy_model`, `max_dE_frac` | `--straggling`, `--energy-model`, `--maximum-fractional-energy-loss` | also accepted by `profile create\|set` for compatibility |
| `inelastic_model`, `inelastic_cutoff_eV`, `secondary_threshold_eV` | `--inelastic-model`, `--inelastic-cutoff-ev`, `--secondary-threshold-ev` | |
| `elastic_model`, `bremsstrahlung_model` | `--elastic-model`, `--bremsstrahlung-model` | |
| `radiative_model`, `radiative_cutoff_eV` | `--radiative-model`, `--radiative-cutoff-ev` | |
| `pair_production_model`, `positron_transport` | `--pair-production-model`, `--positron-transport` | |
| `atomic_electron_deflection` | `--atomic-electron-deflection` | |

## Line-grid policy

`pyrite profile line-grid show|set|reset NAME` owns `[profiles.NAME.line_grid_policy]`.

| TOML key | Values (default first) | Flag |
|---|---|---|
| `bandwidth` | `kinematic-ceiling`, `resonance-population` | `--bandwidth` |
| `resolution` | `sinc-nyquist`, `resonance-local` | `--resolution` |
| `quadrature` | `node`, `bin-mean` | `--quadrature` |
| `windows` | `false`, `true` | TOML; selector edits keep it |
| `max_points` | integer >= 2 (600000) | TOML; selector edits keep it |

`resonance-local` requires `resonance-population` and `bin-mean`. The editor
also refuses a policy without `windows = true` on `coherent`/`both` emission,
and `bin-mean` with a positive `max_dE_frac`. `coverage-0.95` labels stored energy-grid artifacts and
is not a profile value.

Not profile keys (per-call `Sweep.line_grid_policy` in the API, or
environment): per-observable `rtol` (`PYRITE_ENERGY_GRID_RTOL`,
`PYRITE_ENERGY_GRID_RTOL_<OBSERVABLE>`; built-in 1e-3 intrinsic source, 1e-2
detected counts), `max_spacing_eV` (`PYRITE_ENERGY_GRID_MAX_SPACING_EV`; 3 eV),
`backend_safety_ulps` (`PYRITE_ENERGY_GRID_ULPS`; 8), and detailed window maps
(API only; the profile `windows` boolean takes the resolver defaults).
`max_points` also reads `PYRITE_ENERGY_GRID_MAX_POINTS`. Bandwidth truncation
(1e-4) and the local halo limit (1e-4) are built-in.

## Instruments and emission

| TOML key | Setter | Notes |
|---|---|---|
| `beam` | `profile set --beam NAME`; `profile remove --beam` | named `[beams.NAME]` edited with `pyrite beam`; inline tables are TOML |
| `detector` | `profile set --detector NAME` | named `[detectors.NAME]` edited with `pyrite detector` |
| `detectors.ID` | TOML | collection; cannot be combined with `detector`/`physical_detector` |
| `physical_detector` (+ `scorer`, `response`, `acquisition`) | `profile physical-detector show\|set\|reset` | |
| `filters` | `profile filter add\|set\|rm\|list\|show` | order is part of identity |
| `emission` | `profile set --emission`; `profile add\|remove --coherent/--incoherent` | absent = `incoherent` |
| `temporal_profile` | `profile set --temporal-profile/--no-temporal-profile` | absent = off |

## Creation and cloning

`profile create NAME` copies the packaged `standard` ranges, bremsstrahlung
grid, membership and member-material overrides; it copies no instrument,
numerics or line-grid policy. `profile create NAME --from SOURCE` copies every
SOURCE key except `overrides`, including numerics and `line_grid_policy`, and
lists the inherited sections.
