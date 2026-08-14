# Troubleshooting

## A command uses the wrong profile, host, or workspace

Run `pyrite config list`; it reports the winning source for every value. A
command option beats `PYRITE_*`, which beats legacy `CXR_*`, the persistent
store, and the built-in default. Remove or change the higher-precedence value.

## The catalog does not load

Use `pyrite material validate [PATH]`. Errors are grouped by TOML location.
Common causes are an unknown key, missing required root table, invalid grid
descriptor, profile membership naming an unknown material, beam references to
an unknown beam, CIF paths outside packaged data, or elements without transport
support. See the [catalog schema](../repo-design/materials-catalog-schema.md).

## A run asks for an energy grid

The selected profile/material lacks a verified immutable line-grid reference.
Select the profile through `PYRITE_PROFILE` or persistent configuration, run
`pyrite material energy-grid derive`, add the returned JSON to that profile with
`pyrite-dev energy-grid add --profile NAME`, then run `pyrite-dev energy-grid
verify`. Do not paste
survey-cropped bounds into the catalog; fidelity reduction happens later.

## A run does not resume the checkpoint I expected

Compare the dataset identities with `pyrite checkpoint list`. Fidelity,
profile-resolved values, overrides, beam/detector values, emission mode, and
crystallography can change the digest and stem. This is intentional isolation,
not a cache miss bug. Legacy checkpoints without identity are
provenance-incomplete.

## Analysis cannot find output

Confirm the effective `workspace.root`, any `PYRITE_HOME`/`CXR_HOME`, and any
`--checkpoint-dir` used for the run. Then use `pyrite checkpoint list` from the
same context. Analysis needs component checkpoint data; exported HTML alone is
not an input dataset.

## Only one spectral component is present

Interrupted runs and component recomputation can leave line or brem alone. Use
`pyrite checkpoint recompute line|brem MATERIAL` for the missing component.
Check identity before merging or restoring data; incompatible identities are
correctly rejected.

## The GPU is unavailable or runs out of memory

Inspect `pyrite config setup --help` and the active backend environment. CPU is the
safe base installation. Do not solve memory pressure by changing scientific
inputs invisibly: use documented resource/chunk controls, record them for
performance work, and route heavy GPU sweeps through the [cluster
workflow](running-on-a-cluster.md).

## Remote execution fails

Verify `pyrite config get remote.target`, then inspect `pyrite job list` and
`pyrite job status JOB_ID`. Use `job logs` for scheduler/runtime failures and
`remote` resource commands for connectivity/setup. Detached submission returns
before results exist; pull or wait for completion before analysis.

## Results look plausible but differ unexpectedly

Compare full identities, selected coordinates, fidelity, emission mode,
detector response, solid-angle scaling, and catalog revision. View source
components before detector convolution. Then consult the [validation
ledger](../validation/physics-validation-ledger.md); execution success is not
physics validation.
