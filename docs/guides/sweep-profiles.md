# Sweep fidelity and dataset identity

`pyrite run` accepts two named, independently resolved fidelity policies:

- `full` preserves production behavior: catalog grids, 300 line electrons,
  150 bremsstrahlung electrons, and complete configured reflection sets.
- `survey` is provisional. It uses 60/30 electrons, at most two beam energies,
  three thicknesses, five polar tilts, two azimuths, two reflection families
  (at most four resolved reflections), and photon grids cropped to their
  central 70% then sampled at one-quarter density.

Run a survey with:

```bash
pyrite run standard -m mose2 --fidelity survey
```

`--profile full|survey` is not a compatibility spelling: `profile` is reserved
for catalog `[profiles.*]` campaigns. Fidelity (`--fidelity`) selects the
grid-reduction policy; catalog profiles select scan-parameter ranges. See
[ADR-0005](../adr/0005-energy-grid-schema-decisions.md) for the decision record.

The supported high-level Python API has no fidelity shorthand. Construct a
`Scene`, `Sweep`, and `Numerics` explicitly as described in the [Python API
workflow](python-api-workflow.md). The internal
`pyrite.campaign.config.default_settings` and `material_sweep` helpers remain
campaign compatibility surfaces rather than the supported library entry point.

`pyrite material energy-grid derive`, locally or with `--remote`, is upstream
of this choice: it measures catalog-ready line and bremsstrahlung bounds
without a fidelity setting and installs those full bounds for `--profile NAME`
(or the configured current profile). Later
`pyrite run --fidelity survey` reduces the stored photon grids together with
other sweep axes; `full` uses them unchanged.

## Calculation numerics

Use the nested profile workflow to inspect every result-affecting calculation
control and the source of its effective value:

```bash
pyrite profile numerics show standard --fidelity full
pyrite profile numerics show standard --fidelity survey -o json
```

The output groups sampling counts, reflection/mosaic convergence, and transport
integration controls. Each field reports its explicit profile value, effective
value, and source (`profile`, `fidelity`, or `built-in`). Resolution follows:

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

`--dry-run` prints the exact TOML diff without writing. Reset named fields, or
omit all field names to reset every explicit numeric to fidelity/built-in
resolution:

```bash
pyrite profile numerics reset standard reflection-families mosaic-route --yes
pyrite profile numerics reset standard --yes
```

Existing `profile create|set --ne-line/--ne-brem`, `--straggling`,
`--energy-model`, and `--max-de-frac` spellings remain compatible. Worker,
chunk, backend, core, and other execution-only tuning are intentionally absent:
they affect runtime, not calculation results or checkpoint identity.

## Identity and storage

Every profile-aware scan resolves settings and complete `Sweep` first, converts
them to JSON-compatible values, and hashes that payload with SHA-256. Component
checkpoint `meta.json` stores profile, hash, and exact resolved parameter
payload under `dataset_identity`.

Electron counts, reflection limits, mosaic quadrature, and transport integration
settings are part of that resolved identity. Distinct effective numerics cannot
resume into one checkpoint; execution-only tuning remains identity-neutral.

Canonical, unmodified `full` runs retain `checkpoints/<material>/` for
compatibility. Survey runs and explicitly overridden full runs use
`checkpoints/<material>@<label>-<12-char-hash>/`. Older
`<material>--<fidelity>-<digest>` stems remain readable. `--quick` retains its
historical `<material>_quick` stem but also records resolved identity. Thus
variants cannot silently resume into each other.

## Named beams

A beam is a catalog object of its own, a top-level `[beams.<name>]` table
alongside `[materials.*]` and `[profiles.*]`, and a profile attaches one by name:

```toml
[beams.gaussian_200fs]
label = "200 fs Gaussian bunch, 0.1 mm spot"
transverse_fwhm_mm = 0.1
rep_rate_hz = 5000.0
bunch_charge_pc = 1.0

[beams.gaussian_200fs.longitudinal]
kind = "gaussian"
envelope_rms_fs = 200.0

[profiles.hopg_hbn_gaussian_200fs]
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

`pyrite beam list|show|create|set|rename|delete` manages the objects;
`pyrite profile set <profile> --beam NAME` attaches one and
`pyrite profile remove <profile> --beam` detaches it. `rename` rewrites every
referencing profile, and `delete` is blocked while any profile still points at
the beam, so a reference is never orphaned.

The reference resolves to *values* before hashing, and `label` is stripped, so
`parameter_sha256` -- and therefore every checkpoint stem -- depends on the beam
the profile runs, not on what that beam is called. Two profiles sharing a beam
share checkpoints; renaming a beam moves nothing.

The older inline `[profiles.<name>.beam]` table still decodes and means exactly
the same thing, but nothing writes it any more and all bundled profiles have
been converted. A profile carrying both spellings fails to load.

Detector geometry uses the same named-object pattern shown above. Use
`pyrite detector list|show|create|set|rename|delete` to manage these
objects and `pyrite profile set <profile> --detector NAME` to attach one.
Rename updates every reference; delete names and refuses surviving referents.
The name and label never enter checkpoint identity, but resolved geometry does.
Legacy inline `[profiles.<name>.detector]` tables and the three equivalent
profile flags remain readable during their deprecation window. Response models
and detector energy bins are runtime objects and are not serialized here.
Named and inline geometry nevertheless preserve the catalog's existing default
`Timepix3` response; this named-object workflow does not make that response
configurable.

## Beam block

A beam table -- named or inline -- decodes into `BeamSpec`. The nested
`longitudinal` and `transverse` sub-tables carry the bunch and phase-space
policies; `docs/physics/beam-transport/beam-phase-space.md` is the reference for every key, its units,
and the mutual exclusions between them. `pyrite beam create` / `pyrite beam set` write
the same keys from `--emittance`, `--twiss-beta`, `--twiss-alpha`,
`--energy-spread`, and the legacy `--transverse-fwhm-mm`. The nine equivalent
`pyrite profile create` / `pyrite profile set` flags still work and still write an
inline block, but each warns once naming `pyrite beam`.

Both sub-tables join `parameter_sha256` only when they diverge from the inert
defaults, so a profile that never sets them hashes exactly as it did before the
keys existed and resumes into its existing checkpoints.

## Finite filters and one physical detector

Finite downstream filter plates are profile-local observation settings. They
are not part of `pyrite run`, `Sweep`, or checkpoint identity. Use
`pyrite material simulate` for a single in-memory scene on a physical pixel
detector; it follows the validated planar pixel-ray path and records a separate
observation digest.

Each `[[profiles.NAME.filters]]` table describes one `FilterPlate`. Its pose
uses the same source-centred observation fields as the Python API. A
`[profiles.NAME.physical_detector]` table is required by `material simulate`;
it supplies the detector pose and pixel geometry. Omitting `shape` and
`pitch_mm` selects a 256 by 256 Timepix3-style grid at 0.055 mm pitch.

Use `pyrite profile filter add|rm|list|show` to edit and inspect plates. The
`add` command validates the plate with the same public object used by the
Python API. It can also create or deliberately replace the physical-detector
table with `--detector-distance-mm` and the prefixed pose flags; `--shape` and
`--pitch-mm` require that distance. For example:

```bash
pyrite profile filter add filter_demo --name half_filter --material silicon \
  --thickness-mm 0.1 --size-mm 7.04 14.08 --distance-mm 200 \
  --detector-distance-mm 400
```

`pyrite material simulate MATERIAL --profile NAME` requires singleton
thickness, energy, polar, and azimuth profile grids, because it runs exactly
one scene. It prints a compact line/background and pixel-grid summary;
`-o json` emits its stable envelope and `-o wide` emits one tab-separated
summary line. Pass `--output-file PATH.npz` to write the complete factorized
spatial arrays. The command never creates a checkpoint.

The bundled `emittance_demo` beam is the worked example: a Courant-Snyder waist
(`alpha_twiss_x = 0`) on the crystal entrance face at 0.1 mm·mrad normalized
emittance and a 0.05 m beta function, plus a 0.1% energy spread, run by
`hopg_emittance_demo` over hopg at 30 and 100 keV. Because the stored emittance
is normalized, that one beam is the same physical beam at both energies -- 0.12
mm and 2.4 mrad RMS at 30 keV, shrinking as `1/sqrt(beta*gamma)` at 100 keV. A
`transverse` table clears the spot FWHM that a beam otherwise defaults to; the
two spellings are mutually exclusive, and specifying both is an error rather
than a precedence rule. The `compressed_microbunch` beam, run by
`hopg_hbn_compressed_microbunch`, is the longitudinal counterpart.

Archive and restore copy the complete component directory, including identity
metadata. Archive merge rejects two identity-bearing datasets whose resolved
parameter hashes differ. Legacy checkpoints without identity remain readable
and merge-compatible; their provenance cannot be reconstructed retroactively.
