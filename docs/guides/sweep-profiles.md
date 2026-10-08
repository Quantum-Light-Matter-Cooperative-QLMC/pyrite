# Sweep fidelity and dataset identity

`pyrite run` resolves every sweep at `full` fidelity: automatic case-local line grids, 300 line electrons, 150 bremsstrahlung electrons, and complete configured reflection sets.

`--fidelity {full,survey}` is deprecated in 0.4.0 and will be removed in 0.6.0 (see [CLI deprecations](../repo-design/cli/cli-deprecations.md)). During the window it still works and warns once on stderr. `--fidelity full` is the default and can simply be omitted. The provisional `survey` preset (60/30 electrons, at most two beam energies, three thicknesses, five polar tilts, two azimuths, two reflection families with at most four resolved reflections, and explicit photon grids cropped to their central 70% then sampled at one-quarter density) is retired with no built-in replacement. Use `--quick` for a smoke test, or a user-defined catalog profile with explicit electron counts and narrower grids for a reduced campaign (see the [configuration cookbook](configuration-cookbook.md)).

Existing `--survey` checkpoints keep their identity: they remain readable, and `pyrite remote pull MATERIAL@PROFILE --hash PREFIX` selects one when several variants share a profile. Full-fidelity stems and digests are unchanged. `profile` names catalog `[profiles.*]` campaigns only; see [ADR-0005](../adr/0005-energy-grid-schema-decisions.md) for the decision record.

The supported high-level Python API has no fidelity shorthand. Construct a `Scene`, `Sweep`, and `Numerics` explicitly as described in the [Python API workflow](python-api-workflow.md). The internal `pyrite.campaign.config.default_settings` and `material_sweep` helpers remain campaign compatibility surfaces rather than the supported library entry point.

`pyrite material energy-grid derive`, locally or with `--remote`, is an optional optimization upstream of this choice: it measures line and bremsstrahlung bounds without a fidelity setting and installs those bounds for `--profile NAME` (or the configured current profile). Without installed bounds, runs use automatic line-grid resolution.

## Calculation numerics

Use the nested profile workflow to inspect every result-affecting calculation control and the source of its effective value:

```bash
pyrite profile numerics show standard
pyrite profile numerics show standard -o json
```

The output groups sampling counts, reflection/mosaic convergence, and transport integration controls. Each field reports its explicit profile value, effective value, and source (`profile`, `fidelity`, or `built-in`). Resolution follows:

```text
per-run override > explicit profile value > fidelity preset > built-in
```

Set controls without editing TOML directly:

```bash
pyrite profile numerics set standard \
  --line-electrons 500 --bremsstrahlung-electrons 200 \
  --reflection-families 6 --maximum-reflections 12 \
  --mosaic-route mc --mosaic-nodes 7 --yes

pyrite profile numerics set standard \
  --energy-model midpoint --maximum-fractional-energy-loss 0.02 --straggling --yes
```

`--dry-run` prints the exact TOML diff without writing. Reset named fields, or omit all field names to reset every explicit numeric to fidelity/built-in resolution:

```bash
pyrite profile numerics reset standard reflection-families mosaic-route --yes
pyrite profile numerics reset standard --yes
```

Existing `profile create|set --ne-line/--ne-brem`, `--straggling`, `--energy-model`, and `--max-de-frac` spellings remain compatible. Worker, chunk, backend, core, and other execution-only tuning are intentionally absent: they affect runtime, not calculation results or checkpoint identity.

`pyrite profile create NAME` starts from the packaged `standard` sweep's ranges, inline bremsstrahlung grid, and membership. It uses only materials present in the selected catalog. Packaged per-material overrides, such as stack layer counts, are copied only for the new profile's member materials. A non-member resolves against `standard`, so its row would never be read. It does not copy the selected catalog's mutable `standard` profile or attach a beam, scalar detector, physical detector, filters, emission policy, transport numerics, or line-grid policy. Absent emission resolves to incoherent and absent straggling resolves to off. `pyrite profile show NAME` marks code defaults with `(default)` and shows an absent detector as `none`.

`pyrite profile create NAME --from SOURCE` explicitly clones SOURCE's ranges, membership, beam, scalar and physical detectors, filters, emission, transport numerics, line-grid policy, and adaptive precision policy. Its output lists any inherited instrument and physics sections. Per-material overrides remain local to SOURCE and are not cloned. Review the cloned profile before running it, especially if SOURCE is a modified `standard`.

## Adaptive electron counts

Profiles run adaptive electron counts by default. Each incoherent case transports electrons in blocks and stops once the relative standard error of each watched yield meets the target and the heavy-tail guards pass. If the target is not met by the maximum count, the run stops there and is flagged statistics-limited. One count serves both line and bremsstrahlung transport.

A profile with neither fixed counts nor a `precision` table uses the default policy: target relative standard error 0.05 on line and bremsstrahlung yields, 200 to 20,000 electrons, blocks of 100, and the default guards. A profile keeps fixed counts, and `pyrite profile show` and `pyrite profile precision show` say why, when it sets `n_electrons`/`n_electrons_brem` (directly or in a material override) or when it cannot run adaptive: coherent or `both` emission, particle cascades (secondaries, pair production, positrons), a GDF beam, or a grooved entrance face. To opt a profile out, give it fixed counts with `pyrite profile numerics set NAME --line-electrons N --bremsstrahlung-electrons N`.

The default (#361) changed every affected dataset identity once, including `standard`'s, so runs from before it do not resume; their checkpoints stay on disk under the old stems. On a subset of the `standard` sweep (hopg and MoS2, 48 cases, 30–300 keV, 100 nm–1 mm), the default converged every case at 400–2,100 electrons. A fixed count of 300 left about half the cases above 5 %, and a single fixed count that met 5 % everywhere needed 1,500–2,000 electrons per case.

To set an explicit policy:

```bash
pyrite profile numerics reset hopg_scan line-electrons bremsstrahlung-electrons
pyrite profile precision set hopg_scan --target-rse 0.05 \
  --min-electrons 400 --max-electrons 20000 --block-electrons 100
pyrite profile precision set hopg_scan --observable line --observable brem
pyrite profile precision show hopg_scan
pyrite profile precision reset hopg_scan            # back to the default policy
```

The policy is stored as `[profiles.NAME.precision]`. A new policy needs `--target-rse`, `--min-electrons`, `--max-electrons` and `--block-electrons`. The minimum, maximum and optional `--pilot-electrons` must be multiples of the block size. Optional guards default to `--max-electron-share 0.05`, `--min-effective-electrons 100`, `--stability-blocks 3` and `--stability-fraction 0.5`. `--band-ev START,STOP` fixes the monitor band; it defaults to the case's line band. `--batch-means-band-ev` adds reported per-bin errors and is never used to stop.

A profile with an explicit `precision` table cannot also set `n_electrons`/`n_electrons_brem` grids or coherent emission; the catalog loader rejects either combination. An explicit policy also refuses physical detectors (`material simulate`), GDF beams, grooves, and particle cascades, rather than falling back. Adaptive runs use only the per-electron or CUDA transport cores. An adaptive run never uses the canonical `<material>` checkpoint stem.

The Python API takes the same policy as `Numerics(precision=Precision(...))`; see the [API example](../api.md#scene-simulation). Configured `Settings`/`Sweep` campaigns can set `Settings(precision=policy)` and lower it with `api.build_configured_cases`.

The dataset and case identities include the requested policy. Realized counts and statistics are stored separately as `adaptive_sampling` in the result and in every checkpoint component, so a realized count never changes the identity used for resume or CAS reuse. Electron-count sweep grids cannot be combined with adaptive precision. Use [statistical methods](../computation/statistical-methods.md#adaptive-electron-counts) to choose a minimum that accounts for skewed contributions and rare electrons.

## Bremsstrahlung grid

A profile's `E_grid_brem` sets the continuum grid for every case. For a uniform `arange` grid only `step` takes effect at full fidelity. Each case raises `start` to the medium's photon-continuum floor, snapped onto the `step` lattice, and replaces `stop` with the beam energy plus one step. A per-material uniform override therefore changes only the step. A nonuniform grid, such as the geometric continuum from `pyrite-dev energy-grid brem set --spacing geometric`, is used as stored and is a real per-material limit.

Earlier releases stored a derived uniform `E_grid_brem` override for each material in the packaged profiles and copied them into new profiles. Those rows are gone. Case grids and case content keys are unchanged. The dataset identity hashes the declared sweep grid, though, so affected materials get a new `parameter_sha256`. Resuming an older checkpoint for one of them reports `checkpoint dataset identity mismatch`: archive it and rerun. Uniform `E_grid_brem` rows in existing user profiles under `~/.pyrite/catalog/profiles/` are harmless. Deleting them has the same identity effect.

## Line-grid policy

By default each case's line axis spans a closed-form kinematic bandwidth at the measured sinc spacing. That bandwidth is capped at the beam's kinetic energy and at the end of the Chantler coupling data (about 966 keV for every element), above which no line can emit in this model. At MeV beam energies that axis needs millions of nodes. A profile can instead name measured policies in `[profiles.NAME.line_grid_policy]`:

| Selector | Values (default first) | Meaning |
|---|---|---|
| `bandwidth` | `kinematic-ceiling`, `resonance-population` | closed-form upper edge, or the edge measured from the case's own resonance population (capped by the closed form) |
| `resolution` | `sinc-nyquist`, `resonance-local` | uniform measured sinc spacing, or fine spacing only where narrow lines resonate on a 3 eV backbone |
| `quadrature` | `node`, `bin-mean` | point samples of the line profile, or bin means: exact integrated yield, smoothed peak height and width |

`resonance-local` requires `resonance-population` and `bin-mean`. Edit the table with the `line-grid` group rather than TOML; impossible combinations are usage errors (exit 2) and nothing is written:

```bash
pyrite profile line-grid show high_energy
pyrite profile line-grid show high_energy -o json
pyrite profile line-grid set my_profile --bandwidth resonance-population \
  --resolution resonance-local --quadrature bin-mean --dry-run
pyrite profile line-grid set my_profile --bandwidth resonance-population \
  --resolution resonance-local --quadrature bin-mean
pyrite profile line-grid reset my_profile resolution quadrature
pyrite profile line-grid reset my_profile
```

`show` lists each selector's explicit value, effective value and source, and which grid source the profile's cases use. `pyrite profile show NAME` prints a one-line summary, and its JSON payload carries `line_grid_policy` (`null` when unset). Editing `standard` prompts unless `--yes`.

Grid-source precedence per case, highest first:

1. A profile policy. Any stored selector, even one equal to its default, makes every case resolve its line grid automatically from its own trajectories. Explicit `E_grid_line`, `energy_grid_refs` and stored rows are then ignored.
2. Without a policy: an explicit `E_grid_line` (profile or material override).
3. A stored per-energy row from `energy_grid_refs` or the legacy `[energy_grids.MATERIAL]` store. `PYRITE_ENERGY_GRID_*` environment values outrank this layer, not layer 2.
4. Automatic resolution with the default selectors for energies a stored mapping does not cover; the crystal's built-in `E_grid` where no mapping exists.

Automatic resolution does not support coherent emission (#117), so the editor refuses a policy on a `coherent` or `both` profile. `bin-mean` also refuses a positive `max_dE_frac`. Coherent feature windows are #350.

The table applies to every case of the profile and joins its dataset identity: editing it gives new checkpoints, and earlier results stay under their old identity. `profile create --from SOURCE` clones it; a fresh `profile create` does not set one. `high_energy` uses the full measured policy. A measured-bandwidth case records its upper-edge truncation audit and the line yield's per-electron relative standard error in `line_grid_resolved`. When a few electrons carry the yield (relative standard error above 0.1, typical at 5 MeV where rare electrons scatter into the detector's radiation cone), the case still runs, is flagged `statistics_limited`, and raises `LineYieldStatisticsWarning`.

Tolerances, maximum spacing, point budget, backend ULPs and feature windows are not profile keys. Set them per call through `Sweep.line_grid_policy` or with `PYRITE_ENERGY_GRID_*`; see the [profile settings reference](../repo-design/profile-settings.md) for names and built-in values.

## Identity and storage

Every profile-aware scan resolves settings and complete `Sweep` first, converts them to JSON-compatible values, and hashes that payload with SHA-256. Component checkpoint `meta.json` stores profile, hash, and exact resolved parameter payload under `dataset_identity`.

Electron counts, reflection limits, mosaic quadrature, and transport integration settings are part of that resolved identity. Distinct effective numerics cannot resume into one checkpoint; execution-only tuning remains identity-neutral.

Canonical, unmodified `full` runs retain `checkpoints/<material>/` for compatibility. Survey runs and explicitly overridden full runs use `checkpoints/<material>@<label>-<12-char-hash>/`. Older `<material>--<fidelity>-<digest>` stems remain readable. `--quick` retains its historical `<material>_quick` stem but also records resolved identity. Thus variants cannot silently resume into each other.

## Named beams

A beam is a catalog object of its own, a top-level `[beams.<name>]` table alongside `[materials.*]` and `[profiles.*]`, and a profile attaches one by name:

```toml
[beams.gaussian_200fs]
label = "200 fs Gaussian bunch, 0.1 mm spot"
transverse_fwhm_mm = 0.1
rep_rate_hz = 5000.0
bunch_charge_pc = 1.0

[beams.gaussian_200fs.longitudinal]
kind = "gaussian"
envelope_rms_fs = 200.0

[profiles.my_scan]
beam = "gaussian_200fs"

[detectors.eds]
label = "SEM EDS geometry"
observation_angle_deg = 119.0
polar_acceptance_deg = 16.6
solid_angle_sr = 0.066

[profiles.validation]
detector = "eds"

[profiles.filter_demo.physical_detector]
distance_mm = 400.0
polar_deg = 90.0

[[profiles.filter_demo.filters]]
name = "half_filter"
material = "silicon"
thickness_mm = 0.1
size_mm = [7.04, 14.08]
distance_mm = 200.0
polar_deg = 90.0
offset_mm = [3.52, 0.0]
```

`pyrite beam list|show|create|set|rename|delete` manages the objects; `pyrite profile set <profile> --beam NAME` attaches one and `pyrite profile remove <profile> --beam` detaches it. `rename` rewrites every referencing profile, and `delete` is blocked while any profile still points at the beam, so a reference is never orphaned.

The reference resolves to *values* before hashing, and `label` is stripped, so `parameter_sha256` -- and therefore every checkpoint stem -- depends on the beam the profile runs, not on what that beam is called. Two profiles sharing a beam share checkpoints; renaming a beam moves nothing.

An inline `[profiles.<name>.beam]` table still decodes and means exactly the same thing as a reference, but the CLI never writes it. A profile carrying both spellings fails to load. `--beam NAME` is the only way `pyrite profile` touches beam phase space.

Detector geometry can be declared in `[profiles.NAME.detectors.ID]` tables. Each ID is a lowercase letter followed by lowercase letters, digits, `_`, or `-`. A table may contain scalar acceptance (`observation_angle_deg`, `polar_acceptance_deg`, `solid_angle_sr`) or pixel geometry (`distance_mm`, pose, grid, and optional `scorer`, `response`, and `acquisition`). Pixel acceptance is derived from its geometry; scalar acceptance fields cannot be mixed into the same pixel table. A reference to a named `[detectors.NAME]` object can be written under `[profiles.NAME.detectors]` as `ID = "NAME"`; named objects can also contain pixel geometry. `pyrite profile show NAME` lists every ID and its resolved acceptance.

`pyrite detector list|show|create|set|rename|delete` still manages named scalar objects, and `pyrite profile set NAME --detector REF` still writes the legacy single-detector reference. Use catalog TOML for a collection or a named pixel object. The legacy `detector` and `physical_detector` profile entries remain loadable, but a profile cannot combine either with a `detectors` collection.

### Bundled examples and implicit defaults

The bundled `default` beam (200 fs, 5 kHz, 1 pC) and `default` detector (90 degrees) are examples, not a description of your beamline or detector, and the bundled `standard` profile is a 128-line example sweep. PyRITE still falls back to them when nothing names a choice, and warns on stderr:

- `pyrite run` and `pyrite remote start` without `PROFILE`, `PYRITE_PROFILE`, or a saved `profile.current` run `standard` and warn. Select a profile with `pyrite run NAME` or keep one with `pyrite config set profile.current NAME`.
- A run from a user-selected catalog (see [External catalogs](external-catalog.md)) whose profile names no `beam` uses the built-in example beam and warns; one that names no `detector` uses the code-default scalar detector and warns. Attach your own with `pyrite profile set NAME --beam BEAM --detector DETECTOR`. Profiles run from the bundled catalog are examples themselves and do not warn.

A profile that omits `detector` and `physical_detector` resolves the code-default scalar detector and no physical detector; it does not inherit geometry or a counting observation from `standard`. Attach the intended detector explicitly to use other geometry. The `standard` profile itself and profiles with explicit geometry keep their resolved identities.

The [deprecation schedule](../repo-design/cli/cli-deprecations.md#implicit-defaults) gives the release from which each fallback becomes an error. Only the fallback changes: `standard` keeps its name and contents, and its checkpoints keep their stems.

## Beam block

A beam table -- named or inline -- decodes into `BeamSpec`. The nested `longitudinal` and `transverse` sub-tables carry the bunch and phase-space policies; `docs/physics/beam-transport/beam-phase-space.md` is the reference for every key, its units, and the mutual exclusions between them. `pyrite beam create` / `pyrite beam set` write the same keys from `--emittance`, `--twiss-beta`, `--twiss-alpha`, `--energy-spread`, and the legacy `--transverse-fwhm-mm`. They are the only CLI surface that writes them: `pyrite profile` has no inline beam flags.

Both sub-tables join `parameter_sha256` only when they diverge from the inert defaults, so a profile that never sets them hashes exactly as it did before the keys existed and resumes into its existing checkpoints.

## Detector collections, filters, and counting observations

A profile can mix scalar and pixel detectors. For example:

```toml
[profiles.survey.detectors.eds]
observation_angle_deg = 70.0
polar_acceptance_deg = 10.0
solid_angle_sr = 0.05

[profiles.survey.detectors.camera]
distance_mm = 400.0
polar_deg = 110.0
shape = [256, 256]
pitch_mm = [0.055, 0.055]

[profiles.survey.detectors.camera.response]
kind = "ideal"

[profiles.survey.detectors.camera.acquisition]
exposure_s = 1.0
measured_min_eV = 0.0
measured_max_eV = 20000.0
measured_bin_width_eV = 400.0
hit_threshold_eV = 500.0
```

The pixel pose uses `distance_mm`, `polar_deg`, `azimuth_deg`, `roll_deg`, and `offset_mm`; the grid uses `shape` and `pitch_mm` (default 256 by 256 at 0.055 mm). Optional `scorer`, `response`, and `acquisition` belong to that detector ID. An acquisition produces a factorized counting observation. Profile filters remain shared across the collection.

The older `[profiles.NAME.physical_detector]` table remains editable with `pyrite profile physical-detector show|set|reset`; creating one requires `--distance-mm`. Each `[[profiles.NAME.filters]]` table is one `FilterPlate`, edited in declared order with `pyrite profile filter add|set|rm|list|show`.

```bash
pyrite profile physical-detector set filter_demo --distance-mm 400 --polar-deg 60 \
  --shape 256 256 --angular-shape 5 5
pyrite profile filter add filter_demo --name half_filter --material silicon \
  --thickness-mm 0.1 --size-mm 7.04 14.08 --distance-mm 200
pyrite profile physical-detector set filter_demo --exposure-s 1 \
  --measured-range-ev 0 20000 --measured-bin-width-ev 400 --hit-threshold-ev 500
```

One `pyrite run` invocation executes every detector ID for each selected material. Each detector gets a distinct checkpoint stem containing its ID and parameter digest. With an acquisition, that detector also writes a factorized observation under `observations/<stem>/`, beside its checkpoint. The scalar acceptance used for a pixel detector's source case comes from its geometry, including when there is no acquisition. `profile show` prints that derived acceptance.

Legacy `detector` becomes ID `default`; legacy `physical_detector` becomes ID `physical`. If both are present, a run executes both, and each gets an ID-qualified checkpoint stem. A profile with neither has one implicit scalar `default` detector. A profile with only one legacy detector keeps its checkpoint identity where its resolved geometry is unchanged.

What an edit costs on the next run: pose and pixel grid change the projection and therefore the dataset; angular shape and filters re-evaluate observations on new transport while cached scalar records are kept; response, acquisition, and beam normalization only rescore stored observations, with no transport. Remote transfer and garbage collection of observation stores are not implemented.

`pyrite run NAME -m MATERIAL --ephemeral` runs one local in-memory scene on a pixel detector, using its scorer and, when present, its acquisition. It selects the sole pixel detector automatically; use `--detector ID` when several are present. Thickness, energy, polar, azimuth, and electron-count grids must each resolve to one value. The command prints a compact line/background and pixel-grid summary; `-o wide` prints one tab-separated line, and `-o json` emits the stable `cxr.material.simulate` v1 envelope, including total counts for a counting observation. The same scene resolver, API call, payload, and array writer also serve `pyrite material simulate MATERIAL --profile NAME`, which remains supported.

Use `--output-file PATH.npz` to write the complete factorized spatial arrays, including per-tile intrinsic lines, pixel-to-tile mapping, solid angles, filter path lengths, and line attenuation coefficients. Background remains a 1D summary in the payload. The writer uses the exact requested path, even without a suffix, and refuses to overwrite an existing file. Ephemeral mode never reads or writes campaign checkpoints, the shared per-case checkpoint cache, or observation stores. Only the explicitly requested output artifact is written. `--no-cache` on an ordinary run suppresses shared-cache I/O but still writes campaign checkpoints.

`--detector` and `--output-file` require `--ephemeral`. Ephemeral mode requires an explicit `-m MATERIAL` and accepts normal catalog/profile selection and output/progress controls. It rejects remote/preset workflows, checkpoint/cache controls, trajectories, profiling, quick-grid changes, worker/reflection overrides, wall-clock/resume controls, and command-line beam/GDF overrides before simulation. Put scene inputs and workload counts in the singleton profile. Usage errors exit 2 and runtime failures exit 1; JSON failures use the same simulation envelope, and diagnostics stay on stderr. Ordinary run defaults and its run-summary JSON schema are unchanged.

When comparing line totals, compare the same observation and resolved grids. Ordinary checkpoint spectra are scalar source spectra at the detector's projected angle; an ephemeral pixel spectrum is a solid-angle-weighted average over angular tiles after finite-filter attenuation. Automatic pixel line grids resolve jointly over the tile directions, whereas scalar grids resolve along one direction. The two totals need not agree even with identical electron seeds and counts. A matched 200-electron lowering regression checks seed, counts, geometry, and explicit-grid equality; a deterministic directional-output integration test checks exact payload equality between the two single-scene commands. The historical 1.85e-6 versus 1.76e-6 comparison did not preserve its scratch catalog, seeds, resolved grids, or full outputs, so the exact difference cannot be attributed retrospectively. Use explicit grids and matched transport controls to isolate grid effects, and compare the same factorized observation to test command parity.

The bundled `emittance_demo` beam is the worked example: a Courant-Snyder waist (`alpha_twiss_x = 0`) on the crystal entrance face at 0.1 mm·mrad normalized emittance and a 0.05 m beta function, plus a 0.1% energy spread, run by `hopg_emittance_demo` over hopg at 30 and 100 keV. Because the stored emittance is normalized, that one beam is the same physical beam at both energies -- 0.12 mm and 2.4 mrad RMS at 30 keV, shrinking as `1/sqrt(beta*gamma)` at 100 keV. A `transverse` table clears the spot FWHM that a beam otherwise defaults to; the two spellings are mutually exclusive, and specifying both is an error rather than a precedence rule. The beam's `longitudinal` table is the longitudinal counterpart; lab campaigns that scan microbunch structure keep those beams in their external catalog.

Archive and restore copy the complete component directory, including identity metadata. Archive merge rejects two identity-bearing datasets whose resolved parameter hashes differ. Legacy checkpoints without identity remain readable and merge-compatible; their provenance cannot be reconstructed retroactively.

For native GPT time-output electron beams, see [GDF beam import](gpt-gdf-beams.md),including named beam configuration, run overrides, and normalization.
