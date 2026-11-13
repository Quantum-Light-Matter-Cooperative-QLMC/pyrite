# Profile settings reference

Every key a `[profiles.NAME]` table accepts, with its editing path. "TOML" means
no CLI setter exists: edit `profiles/NAME.toml` in the catalog directory and run
`pyrite material validate CATALOG_DIR` (catalog commands also validate on load). The
workflow guide is [Configure a campaign](../guides/configuration-cookbook.md); the full schema
is [Materials catalog schema](materials-catalog-schema.md).

All keys below except `materials` membership bookkeeping are result-affecting:
they enter the resolved sweep and therefore dataset identity, so a change gives
new checkpoint stems and earlier results stay under their old identity.

## Ranges and membership

`pyrite profile show NAME` presents SETTING / VALUE tables grouped by editing
command: sweep/membership/emission, named presets, beam, detectors, filters,
numerics (`profile numerics`: sampling, reflections and mosaic, integration,
channeling grid), physics (`profile physics`: models and cutoffs, emission,
channeling), adaptive sampling, cxr-grid, energy grids and material overrides.
Numerics and physics rows add a SOURCE column (see
[CLI output vocabulary](cli/cli-output.md)). While adaptive sampling is
active, `cxr-histories` and `brem-histories` read `unused (adaptive)`.
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
| `E_grid_brem` | grid, eV | TOML (`pyrite-dev energy-grid brem set` stores per-material artifacts instead) | absent = built-in 25 eV uniform spacing; uniform bounds resolve per case |
| `E_grid_cxr` | grid, eV | TOML (`pyrite-dev energy-grid line` maintains artifacts) | ignored while a line-grid policy is set |
| `energy_grid_refs` | material -> artifact digest | `pyrite numerics energy-grid derive`, `pyrite-dev energy-grid add` | line rows ignored while a line-grid policy is set; brem still applies |
| `overrides.MATERIAL.*` | any range/grid key above, plus counts | TOML; prefer a single-material profile | Remove existing override keys by editing the TOML. Bundled stack rows (`thickness_layers`, sapphire thickness) stay until materials and physical objects are separated |

Catalog writes use `cxr_histories`, `brem_histories`, `E_grid_cxr` and
`cxr_grid_policy`. The compatibility spellings `E_grid_line` and
`line_grid_policy` still decode to the same fields; declaring both spellings of
one field is an error. The removed budget keys `cxr_trials`, `brem_trials`,
`n_electrons` and `n_electrons_brem` are load errors naming the replacement.
Resolved run metadata retains its identity vocabulary (`Ne`, `Ne_brem`), so
existing checkpoints and results still load.

## Calculation numerics and physics

Profile-scoped settings have two disjoint owners:

- `pyrite profile numerics show|set|reset|validate PROFILE` owns sampling,
  reflection/mosaic, integration and channeling-grid controls, plus the
  `adaptive-sampling` and `cxr-grid` subgroups.
- `pyrite profile physics show|set|reset|validate PROFILE` owns physical
  models, straggling, cutoffs, emission and channeling physics.

`pyrite numerics` and `pyrite physics` manage reusable named presets
(`list|show|create|set|reset|validate|rename|delete|adopt`) that a profile references with
`profile set --numerics|--physics`. `show` prints SETTING / VALUE / SOURCE,
where SOURCE is `default`, `preset:NAME`, `profile` or `run`. The sampling
group leads with `mode: fixed (...)` or `mode: adaptive (...)`. The JSON
payload keeps its existing source spellings; `cxr.profile.numerics.show` now
carries numerics rows only and adds a `preset` key (physics rows moved to
`pyrite.profile.physics.show`).

The 0.31 spellings still work for two minor releases, warn on stderr and keep
their exit codes: `pyrite numerics|physics show|set|reset|validate PROFILE`,
`pyrite numerics|physics preset ...`, `pyrite numerics adaptive-sampling|cxr-grid`,
and physics flags on `profile numerics set|reset`. See
[CLI deprecations](cli/cli-deprecations.md).

Numerics (`profile numerics set`):

| TOML key | Setter flag | Notes |
|---|---|---|
| `cxr_histories`, `brem_histories` | `--cxr-histories`, `--brem-histories` | fixed counts; ignored while adaptive sampling is active. `profile set\|add -l/-b` edit them as sweepable grids |
| `n_families`, `max_reflections` | `--reflection-families`, `--maximum-reflections` | |
| `mosaic_nodes`, `mosaic_route` | `--mosaic-nodes`, `--mosaic-route` | |
| `energy_model`, `max_dE_frac` | `--energy-model`, `--maximum-fractional-energy-loss` | also accepted by `profile create\|set` for compatibility |

Physics (`profile physics set`):

| TOML key | Setter flag | Notes |
|---|---|---|
| `straggling` | `--straggling` | also accepted by `profile create\|set` for compatibility |
| `inelastic_model`, `inelastic_cutoff_eV`, `secondary_threshold_eV` | `--inelastic-model`, `--inelastic-cutoff-ev`, `--secondary-threshold-ev` | |
| `elastic_model`, `bremsstrahlung_model` | `--elastic-model`, `--bremsstrahlung-model` | |
| `radiative_model`, `radiative_cutoff_eV` | `--radiative-model`, `--radiative-cutoff-ev` | |
| `pair_production_model`, `positron_transport` | `--pair-production-model`, `--positron-transport` | |
| `atomic_electron_deflection` | `--atomic-electron-deflection` | |
| `transport_mode`, `mixed_c1`, `mixed_c2`, `mixed_max_step_um` | `--transport-mode`, `--mixed-c1`, `--mixed-c2`, `--mixed-maximum-step-um` | parameters apply only under `mixed`; runs that score CXR lines refuse `mixed` (#990) |

The built-in defaults are `bremsstrahlung_model="bremslib"` and
`radiative_model="bremslib-soft-hard"` with `radiative_cutoff_eV=1000`. Run
preflight admits the coupled default before checkpoint access: it requires the
BremsLib continuum source and an ungrooved target. Unsupported configurations
fail with guidance; `pyrite run --allow-physics-fallback` permits uncoupled
transport and records that substitution in `physics_fallbacks` alongside any
continuum-source fallback. The same policy applies to remote submission and to
legacy `radiative_model="auto"`. Without the flag every coupled request is
strict.

To choose uncoupled transport deliberately, set `radiative_model="uncoupled"`
in the profile or a physics preset (`--radiative-model uncoupled`); it takes no
`radiative_cutoff_eV`. Pair `bremsstrahlung_model="eedl"` with it. This scores radiation after transport without
debiting radiative energy loss from the electron trajectory; it is a model
choice, not a substitute with a guaranteed accuracy bound. Assess its limits
against the [coupled transport model](../physics/radiation-physics/hard-bremsstrahlung-events.md).
Existing checkpoint identities retain their historical interpretation;
resolving a new run does not reinterpret stored `auto` or absent fields.

## Line-grid policy

`pyrite profile numerics cxr-grid show|set|reset NAME` owns `[profiles.NAME.cxr_grid_policy]`.

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
| `detector` | `profile set --detector NAME` | named `[detectors.NAME]` edited with `pyrite detector`; use `reference` for ideal Timepix3 aperture and raw spectra |
| `detectors.ID` | `profile detector list\|show\|add\|set\|reset\|rm`; `migrate` converts legacy storage | collection; cannot be combined with `detector`/`physical_detector` |
| `physical_detector` (+ `scorer`, `response`, `acquisition`) | `profile detector show\|set\|reset NAME physical`; `rm NAME physical` | |
| `filters` | `profile filter add\|set\|rm\|list\|show` | order is part of identity |
| `emission` | `profile set --emission`; `profile add\|remove --coherent/--incoherent` | absent = `incoherent` |
| `temporal_profile` | `profile set --temporal-profile/--no-temporal-profile` | absent = off |

Scalar detector fields must be explicit. Omitted geometry is a
usage error; attach `reference` to preserve the equivalent scalar acceptance
and checkpoint identity. This bundled object integrates the 14.08 mm Timepix3
aperture at 400 mm as one ideal scalar detector (90°, 2.016603235051288° full
polar span, 0.0012390399999999998 sr), exposing raw spectra without pixel/count
acquisition.
External complete catalogs need their own copy of the object.
A `kind = "point"` detector (bundled `point`; `detector set NAME --point` or
`profile detector set PROFILE ID --point`) is explicit without acceptance
fields: it is an ideal on-axis (or `observation_angle_deg`) point observer with
zero acceptance, as admitted axial fibre cases require (#624).

## Creation and cloning

`profile create NAME` copies the packaged `standard` ranges, membership and
member-material overrides. An omitted `E_grid_brem` uses the built-in default.
Creation copies no instrument (pass `--detector reference` for the ideal
aperture), numerics or line-grid policy. `profile create NAME --from SOURCE`
copies every SOURCE key except `overrides`, including numerics and
`cxr_grid_policy`, and lists the inherited sections.

Detector collections are edited with `profile detector list|show|add|set|reset|rm`.
Named scalar/pixel objects belong to `pyrite detector`; profile-local edits of a
reference require `set NAME ID --inline-copy`. Legacy `default` and `physical`
IDs retain their storage until `profile detector add NAME ID --detector REF`
converts it into a collection, or explicit `profile detector migrate NAME` is
requested. Conversion previews the TOML and checkpoint variant consequences
before confirmation; `--yes` applies it in scripts and `--dry-run` previews it.
Each profile run scores every attached ID into its own dataset, with counting
observations stored alongside the corresponding detector's results. The single
`profile set --detector` shortcut updates one collection entry and rejects
ambiguous collections. See [detector contracts](../reference/campaign-configuration.md#detector-collections-filters-and-counting-observations) for
membership, optional-section resets, migration, and inspection JSON.

## Pixel acquisition reporting bins

Within a detector's `acquisition` table, choose exactly one axis form:
`measured_edges_eV`, uniform `measured_min_eV`/`measured_max_eV`/
`measured_bin_width_eV`, or `reporting_bins = "native"`. The native form needs
only `exposure_s`; optional `hit_threshold_eV`, `mode` and `seed` behave as for
explicit edges. `profile detector set PROFILE ID --reporting-bins native` and
`detector create|set NAME --reporting-bins native` replace prior axis settings.
Concrete native edges resolve per complete observation before identity or
storage, with at most 65,536 reporting bins. See the
[detector contracts](../reference/campaign-configuration.md#detector-collections-filters-and-counting-observations).
