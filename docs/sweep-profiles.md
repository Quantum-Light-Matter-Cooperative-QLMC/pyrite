# Sweep fidelity and dataset identity

`cxr run` accepts two named, independently resolved fidelity policies:

- `full` preserves production behavior: catalog grids, 300 line electrons,
  150 bremsstrahlung electrons, and complete configured reflection sets.
- `survey` is provisional. It uses 60/30 electrons, at most two beam energies,
  three thicknesses, five polar tilts, two azimuths, two reflection families
  (at most four resolved reflections), and photon grids cropped to their
  central 70% then sampled at one-quarter density.

Run a survey with:

```bash
cxr run standard -m mose2 --fidelity survey
```

`--profile full|survey` is not a compatibility spelling: `profile` is reserved
for catalog `[profiles.*]` campaigns (see
`docs/cli-energy-grid-sweep-rework-plan.md`). Fidelity (`--fidelity`) selects
the grid-reduction policy; catalog profiles select scan-parameter ranges.

Python callers use `default_settings("survey")` and
`material_sweep("mose2", fidelity="survey")`. `full` remains default for both.
Explicit `material_sweep` overrides apply after profile resolution.

`cxr energy-grid` is upstream of this choice. `derive`, locally or with
`--remote`, measures catalog-ready line and bremsstrahlung bounds without a
fidelity setting, and `apply` stores those full bounds. Later
`cxr run --fidelity survey` reduces the stored photon grids together with
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

## Beam block

A catalog profile's `[profiles.<name>.beam]` table decodes into `BeamSpec`. The
nested `beam.longitudinal` and `beam.transverse` sub-tables carry the bunch and
phase-space policies; `docs/beam-phase-space.md` is the reference for every key,
its units, and the mutual exclusions between them. `cxr profile create` /
`cxr profile edit` write the same keys from `--emittance`, `--twiss-beta`,
`--twiss-alpha`, `--energy-spread`, and the legacy `--transverse-fwhm-mm`.

Both sub-tables join `parameter_sha256` only when they diverge from the inert
defaults, so a profile that never sets them hashes exactly as it did before the
keys existed and resumes into its existing checkpoints.

The bundled `hopg_emittance_demo` is the worked example: a Courant-Snyder waist
(`alpha_twiss_x = 0`) on the crystal entrance face at 0.1 mm·mrad normalized
emittance and a 0.05 m beta function, plus a 0.1% energy spread, over hopg at 30
and 100 keV. Because the stored emittance is normalized, that one block is the
same physical beam at both energies -- 0.12 mm and 2.4 mrad RMS at 30 keV,
shrinking as `1/sqrt(beta*gamma)` at 100 keV. A `beam.transverse` table clears
the spot FWHM that a profile beam otherwise defaults to; the two spellings are
mutually exclusive, and specifying both is an error rather than a precedence
rule. `hopg_hbn_compressed_microbunch` is the longitudinal counterpart.

Archive and restore copy the complete component directory, including identity
metadata. Archive merge rejects two identity-bearing datasets whose resolved
parameter hashes differ. Legacy checkpoints without identity remain readable
and merge-compatible; their provenance cannot be reconstructed retroactively.
