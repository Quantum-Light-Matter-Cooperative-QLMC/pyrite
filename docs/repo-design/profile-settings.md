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
| `transition_radiation` | `--transition-radiation` | `none` (default), `history`, `bunch` or experimental `two-beam` |
| `[channeling] model` | `--channeling-model` | `none` (default) or `planar` |
| `cxr_source` | `--cxr-source` | `mc` (default) or `none`: skip characteristic-line scoring (reflection selection, line grid, line passes) for continuum-only runs; refuses coherent emission, temporal profiles, line roulette, channeling and transition radiation (#1327). Digests change only when `none` |
| `transport_mode`, `mixed_c1`, `mixed_c2`, `mixed_max_step_um` | `--transport-mode`, `--mixed-c1`, `--mixed-c2`, `--mixed-maximum-step-um` | parameters apply only under `mixed`; runs that score CXR lines refuse `mixed` (#990) |

### Turning a setting off

Every optional setting turns off according to its type ("off follows type",
#1242):

| Setting type | On | Off |
|---|---|---|
| Boolean feature (`straggling`, `positron_transport`, `temporal_profile`, `gdf_shape_only`) | `--x` | `--no-x`; TOML `false` |
| Non-boolean optional setting (`secondary_threshold_eV`, `pair_production_model`, `transition_radiation`, `[channeling] model`, `atomic_electron_deflection`) | a value | `none`; TOML `"none"` |
| Any field | — | `reset FIELD` reverts to the attached preset or built-in default; it never means off |

TOML has no null, so an explicit `none` is stored: it overrides a preset that
enables the setting, whereas an absent key inherits the preset. `none`, the
deprecated alias `off`, and an absent key with an off default resolve to the
same case identity and digests. In Python, `Physics` takes `None` or `"none"`
for `pair_production_model` and `secondary_threshold_eV`, `"none"` for
`transition_radiation`, and `simulate(..., channeling=None)` for channeling.
`off` still works in the CLI, TOML and `Physics` until 0.38.0 and warns
([CLI deprecations](cli/cli-deprecations.md)).

Coupled settings turn off together: pair production requires the secondary
threshold, and positron transport requires pair production. Turning off one
half alone is a usage error (exit 2) that names the corrected command.

#### Turn off particle cascades

Secondary and pair cascades run under the profile's history-count mode like
any incoherent case; adaptive sampling counts each primary with all its
descendants. To turn cascades off, including ones enabled by an attached
physics preset:

```console
$ pyrite profile physics set PROFILE --pair-production-model none --secondary-threshold-ev none
```

Add `--no-positron-transport` when positron transport is on.

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
| `bandwidth` | `resonance-population`, `kinematic-ceiling` | `--bandwidth` |
| `resolution` | `sinc-oversampled`, `sinc-nyquist`, `resonance-local` | `--resolution` |
| `quadrature` | `node`, `bin-mean` | `--quadrature` |
| `windows` | `false`, `true` | TOML; selector edits keep it |
| `max_points` | integer >= 2 (600000 `node`, 1000000 `bin-mean`) | TOML; selector edits keep it |

Any stored selector makes every case resolve its line grid from its own
trajectories, ahead of explicit `E_grid_line` and stored energy-grid rows.

### When to pick what

- **Bandwidth** sets the cost. `resonance-population` stops where the case's
  own lines leave at most 1e-4 of their yield above the axis, and audits that
  after transport. `kinematic-ceiling` is the closed-form upper bound: always
  safe, no audit, but far wider at high beam energy.
- **Resolution** sets which observable the axis resolves. `W` is the measured
  sinc-Nyquist step (first-zero width of the narrow lines), capped at 3 eV.

  | `resolution` | Step | Resolves |
  |---|---|---|
  | `sinc-oversampled` | `W/8` (`node`); $(3/\pi)\sqrt{0.01}\,W \approx W/10.5$ (`bin-mean`) | peak height and FWHM of isolated lines to ~1% |
  | `sinc-nyquist` | `W` | line area only; 8-10x fewer points |
  | `resonance-local` | fine near measured resonances, 3 eV elsewhere | yield on axes too wide for a uniform fine step |

- **Quadrature** sets what each point means. `node` samples the line profile
  (spectral density); a step coarser than a line can alias its yield. `bin-mean`
  stores the exact mean over each bin: yield is exact at any step, but peaks
  are lowered and widened, as in a binned detector. It is incoherent-only and
  refuses a positive `max_dE_frac`.
- **Windows** (`windows = true`) refine fine windows around kinematic lines,
  absorption edges and characteristic lines on a `sinc-nyquist` backbone. They
  are the only automatic route for coherent emission.

### Resolution and fallback rules

The table is pinned by `tests/energy-grid/test_line_grid_selector_table.py`.

| `bandwidth` | `resolution` | `quadrature` | `windows` | Axis | Fallback³ |
|---|---|---|---|---|---|
| unset, `kinematic-ceiling` | unset, `sinc-nyquist` | any | on | `kinematic-ceiling` / `sinc-nyquist` | none (already `kinematic-ceiling`) |
| unset, `resonance-population` | `resonance-local` | `bin-mean` | off | `resonance-population` / `resonance-local` | refuses |
| unset | unset, `sinc-oversampled`, `sinc-nyquist` | any | off | default¹ / as chosen² | widens toward `kinematic-ceiling` |
| `resonance-population` | unset, `sinc-oversampled`, `sinc-nyquist` | any | off | `resonance-population` / as chosen² | refuses |
| `kinematic-ceiling` | unset, `sinc-oversampled`, `sinc-nyquist` | any | off | `kinematic-ceiling` / as chosen² | none (already `kinematic-ceiling`) |

Every other combination is refused: windows with `resonance-population`,
`sinc-oversampled` or `resonance-local`; `resonance-local` with `node` or
`kinematic-ceiling`.

1. The default bandwidth is `resonance-population`, or `kinematic-ceiling` for
   coherent emission or a positive `max_dE_frac`, where its audit cannot run.
2. Unset resolution is `sinc-oversampled`. It never depends on the bandwidth.
3. Fallback has one rule: only a defaulted bandwidth moves. If the measured axis
   cannot be audited (channeling, transition-radiation and mixed-transport
   cases), the case uses `kinematic-ceiling`. If the audit refuses after
   transport, the same trajectories are rescored, with a warning, on the first
   wider stop within the share (at most `kinematic-ceiling`). An explicit
   bandwidth, including the one `resonance-local` implies, refuses instead.
   Resolution, quadrature and budget never change.

A `sinc-oversampled` axis over `max_points` is refined locally instead (#1257):
its step around every measured line narrower than 8 `max_spacing_eV`,
`max_spacing_eV` elsewhere, with every spacing join outside each line's halo
(yield within 2e-4 per line under `node`; exact under `bin-mean`). The grid
record's `local_axis` gives the route, uniform count and reason. Beyond every
line's halo the backbone may be graded up to 10 eV, which is above
`max_spacing_eV` (#1271): steps grow by 10 % of the distance from the nearest
protected edge. This applies only where it saves nodes, and never on channeling,
transition-radiation or mixed-transport cases. `local_axis.grading` records it.
Only a local
axis that also exceeds `max_points` refuses, naming `--resolution sinc-nyquist`
(and, on `resonance-population`, `resonance-local`) as opt-outs; nothing
coarsens automatically. See the
[join derivation](../validation/beam-transport/line-grid-node-local-refinement.md).
`sinc-shape` (#1230) is a
deprecated alias of `sinc-oversampled` with `bin-mean` for one release. The
editor also refuses a policy without `windows = true` on `coherent`/`both`
emission. `coverage-0.95` labels stored energy-grid artifacts and is not a
profile value. Derivations: [shape step](../validation/beam-transport/line-grid-sinc-shape-spacing.md),
[`W/8` convergence](../validation/beam-transport/line-grid-sinc-convergence.md),
[measured bandwidth](../validation/beam-transport/line-grid-resonance-bandwidth.md);
decision record: [ADR 0023](../adr/0023-line-grid-selector-rules.md).

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
