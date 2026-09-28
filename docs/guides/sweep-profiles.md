# Sweep fidelity and dataset identity

`pyrite run` accepts two named, independently resolved fidelity policies:

- `full` preserves production behavior: automatic case-local line grids, 300 line electrons, 150 bremsstrahlung electrons, and complete configured reflection sets.
- `survey` is provisional. It uses 60/30 electrons, at most two beam energies, three thicknesses, five polar tilts, two azimuths, two reflection families (at most four resolved reflections), and any explicit photon grids cropped to their central 70% then sampled at one-quarter density. Automatic line grids remain case-local under either fidelity.

Run a survey with:

```bash
pyrite run standard -m mose2 --fidelity survey
```

`--profile full|survey` is not a compatibility spelling: `profile` is reserved for catalog `[profiles.*]` campaigns. Fidelity (`--fidelity`) selects the grid-reduction policy; catalog profiles select scan-parameter ranges. See [ADR-0005](../adr/0005-energy-grid-schema-decisions.md) for the decision record.

The supported high-level Python API has no fidelity shorthand. Construct a `Scene`, `Sweep`, and `Numerics` explicitly as described in the [Python API workflow](python-api-workflow.md). The internal `pyrite.campaign.config.default_settings` and `material_sweep` helpers remain campaign compatibility surfaces rather than the supported library entry point.

`pyrite material energy-grid derive`, locally or with `--remote`, is an optional optimization upstream of this choice: it measures line and bremsstrahlung bounds without a fidelity setting and installs those bounds for `--profile NAME` (or the configured current profile). Survey fidelity reduces an installed explicit grid together with other sweep axes; otherwise both fidelities use automatic line-grid resolution.

## Calculation numerics

Use the nested profile workflow to inspect every result-affecting calculation control and the source of its effective value:

```bash
pyrite profile numerics show standard --fidelity full
pyrite profile numerics show standard --fidelity survey -o json
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

## Line-grid policy

By default each case's line axis spans a closed-form kinematic bandwidth at the measured sinc spacing. At MeV beam energies that axis needs millions of nodes. A profile can instead name measured policies:

```text
[line_grid_policy]
bandwidth = "resonance-population"  # stop from the case's own line population
resolution = "resonance-local"      # fine spacing only where narrow lines resonate
```

The table applies to every case of the profile and joins its dataset identity. `high_energy` uses it. A measured-bandwidth case records its upper-edge truncation audit and the line yield's per-electron relative standard error in `line_grid_resolved`. When a few electrons carry the yield (relative standard error above 0.1, typical at 5 MeV where rare electrons scatter into the detector's radiation cone), the case still runs, is flagged `statistics_limited`, and raises `LineYieldStatisticsWarning`.

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

The older inline `[profiles.<name>.beam]` table still decodes and means exactly the same thing, but nothing writes it any more and all bundled profiles have been converted. A profile carrying both spellings fails to load. The nine `pyrite profile create` / `pyrite profile set` flags that used to write that block have been removed; `--beam NAME` is the only way `pyrite profile` touches beam phase space.

Detector geometry uses the same named-object pattern shown above. Use `pyrite detector list|show|create|set|rename|delete` to manage these objects and `pyrite profile set <profile> --detector NAME` to attach one. Rename updates every reference; delete names and refuses surviving referents. The name and label never enter checkpoint identity, but resolved geometry does. The older inline `[profiles.<name>.detector]` table still decodes and means exactly the same thing, but nothing writes it any more. The three `pyrite profile create` / `pyrite profile set` flags that used to write that block have been removed; `--detector NAME` is the only way `pyrite profile` touches detector geometry. Response models and detector energy bins are runtime objects and are not serialized here. Named and inline geometry leave the scalar detector response-free; configure a physical counting observation or supply an explicit runtime response to model measured detector effects.

## Beam block

A beam table -- named or inline -- decodes into `BeamSpec`. The nested `longitudinal` and `transverse` sub-tables carry the bunch and phase-space policies; `docs/physics/beam-transport/beam-phase-space.md` is the reference for every key, its units, and the mutual exclusions between them. `pyrite beam create` / `pyrite beam set` write the same keys from `--emittance`, `--twiss-beta`, `--twiss-alpha`, `--energy-spread`, and the legacy `--transverse-fwhm-mm`. They are the only CLI surface that writes them: `pyrite profile` has no inline beam flags.

Both sub-tables join `parameter_sha256` only when they diverge from the inert defaults, so a profile that never sets them hashes exactly as it did before the keys existed and resumes into its existing checkpoints.

## Physical detector, filters, and counting observations

A `[profiles.NAME.physical_detector]` table places one planar pixel detector: pose (`distance_mm`, `polar_deg`, `azimuth_deg`, `roll_deg`, `offset_mm`) and pixel grid (`shape`, `pitch_mm`; omitted, one 256 by 256 Timepix3-style chip at 0.055 mm). Optional nested tables add the angular `scorer` (`angular_shape`, nearest-tile reconstruction), the detector `response` (`ideal` or uncalibrated `timepix3`), and an `acquisition` (exposure, reporting-bin edges, post-response hit threshold, expected or seeded Poisson counts). A profile without its own table inherits `standard`'s.

Edit it with `pyrite profile physical-detector show|set|reset`; `set` changes only the fields given and gives an inheriting profile its own copy first, so `standard` never changes. Each `[[profiles.NAME.filters]]` table is one `FilterPlate`, edited in declared order with `pyrite profile filter add|set|rm|list|show`. For example:

```bash
pyrite profile physical-detector set filter_demo --distance-mm 400 --polar-deg 60 \
  --shape 256 256 --angular-shape 5 5
pyrite profile filter add filter_demo --name half_filter --material silicon \
  --thickness-mm 0.1 --size-mm 7.04 14.08 --distance-mm 200
pyrite profile physical-detector set filter_demo --exposure-s 1 \
  --measured-range-ev 0 20000 --measured-bin-width-ev 400 --hit-threshold-ev 500
```

With an acquisition the profile is a **counting observation**, and two things change for `pyrite run`:

- The sweep's scalar observation angle, polar acceptance, and solid angle come from the physical detector's projection, so the intrinsic checkpoint and the observation describe the same transport. Scalar detector overrides on such a profile are rejected, and `pyrite profile show` marks the scalar detector as superseded.
- Every case also stores a factorized observation under `observations/<stem>/`, beside the `checkpoints/` root, evaluated on the same transport as the scalar record (see [Python API workflow](python-api-workflow.md) for reading one back).

What an edit costs on the next run: pose and pixel grid change the projection and therefore the dataset; angular shape and filters re-evaluate observations on new transport while cached scalar records are kept; response, acquisition, and beam normalization only rescore stored observations, with no transport. Remote transfer and garbage collection of observation stores are not implemented yet.

`pyrite material simulate MATERIAL --profile NAME` runs one in-memory scene on the physical detector, using the profile's scorer and, when present, its acquisition. It requires singleton thickness, energy, polar, and azimuth grids, prints a compact line/background and pixel-grid summary (`-o json` is the stable envelope and adds total counts for a counting observation; `-o wide` is one tab-separated line), writes the complete factorized spatial arrays with `--output-file PATH.npz`, and never creates a checkpoint or observation store.

The bundled `emittance_demo` beam is the worked example: a Courant-Snyder waist (`alpha_twiss_x = 0`) on the crystal entrance face at 0.1 mm·mrad normalized emittance and a 0.05 m beta function, plus a 0.1% energy spread, run by `hopg_emittance_demo` over hopg at 30 and 100 keV. Because the stored emittance is normalized, that one beam is the same physical beam at both energies -- 0.12 mm and 2.4 mrad RMS at 30 keV, shrinking as `1/sqrt(beta*gamma)` at 100 keV. A `transverse` table clears the spot FWHM that a beam otherwise defaults to; the two spellings are mutually exclusive, and specifying both is an error rather than a precedence rule. The beam's `longitudinal` table is the longitudinal counterpart; lab campaigns that scan microbunch structure keep those beams in their external catalog.

Archive and restore copy the complete component directory, including identity metadata. Archive merge rejects two identity-bearing datasets whose resolved parameter hashes differ. Legacy checkpoints without identity remain readable and merge-compatible; their provenance cannot be reconstructed retroactively.

For native GPT time-output electron beams, see [GDF beam import](gpt-gdf-beams.md),including named beam configuration, run overrides, and normalization.
