# Sweep profiles and dataset identity

`cxr scan` accepts two named, independently resolved policies:

- `full` preserves production behavior: catalog grids, 300 line electrons,
  150 bremsstrahlung electrons, and complete configured reflection sets.
- `survey` is provisional. It uses 60/30 electrons, at most two beam energies,
  three thicknesses, five polar tilts, two azimuths, two reflection families
  (at most four resolved reflections), and photon grids cropped to their
  central 70% then sampled at one-quarter density.

Run a survey with:

```bash
cxr scan mose2 --profile survey
```

Python callers use `default_settings("survey")` and
`material_sweep("mose2", profile="survey")`. `full` remains default for both.
Explicit `material_sweep` overrides apply after profile resolution.

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

Archive and restore copy the complete component directory, including identity
metadata. Archive merge rejects two identity-bearing datasets whose resolved
parameter hashes differ. Legacy checkpoints without identity remain readable
and merge-compatible; their provenance cannot be reconstructed retroactively.
