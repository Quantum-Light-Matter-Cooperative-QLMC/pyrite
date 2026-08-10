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

Python callers use `default_settings("survey")` and
`material_sweep("mose2", fidelity="survey")`. `full` remains default for both.
Explicit `material_sweep` overrides apply after profile resolution.

`pyrite energy-grid` is upstream of this choice. `derive`, locally or with
`--remote`, measures catalog-ready line and bremsstrahlung bounds without a
fidelity setting, and `apply` stores those full bounds. Later
`pyrite run --fidelity survey` reduces the stored photon grids together with
other sweep axes; `full` uses them unchanged.

## Identity and storage

Every profile-aware scan resolves settings and complete `Sweep` first, converts
them to JSON-compatible values, and hashes that payload with SHA-256. Component
checkpoint `meta.json` stores profile, hash, and exact resolved parameter
payload under `dataset_identity`.

Canonical, unmodified `full` runs retain `checkpoints/<material>/` for
compatibility. Survey runs and explicitly overridden full runs use
`checkpoints/<material>--<variant>-<12-char-hash>/`. `--quick` retains its
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
