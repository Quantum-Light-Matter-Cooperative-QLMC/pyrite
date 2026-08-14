# Working with results

PyRITE stores simulation output as component checkpoints, then lets analysis
and export commands consume those checkpoints without rerunning transport.

## Checkpoint layout and identity

Canonical full runs use `checkpoints/<material>/`. Survey runs and modified
profiles use identity-qualified directories so incompatible parameter sets do
not silently resume into one another. Each dataset records its resolved input
payload and hash in component metadata.

Active datasets are directories beneath the effective `checkpoints/` root and
are offered by the analysis app's dataset selector. `pyrite checkpoint list`
does not list them; it lists labels in the long-term archive shelf:

```bash
uv run pyrite checkpoint list
```

The [sweep-profile guide](sweep-profiles.md) explains how fidelity and profile
resolution affect dataset identity. The
[checkpoint store design](../repo-design/storage/checkpoint-case-store.md)
documents the internal persistence model.

## Analyze or export

Launch the checkpoint-driven analysis app for a material:

```bash
uv run pyrite app analysis launch hopg
```

For non-interactive use, inspect `pyrite app analysis export --help`. The generated
[CLI reference](../repo-design/cli/cli-reference.md) is authoritative for
accepted arguments and output formats.

Current component paths retain a `.pkl` suffix for compatibility, but new
writes contain versioned HDF5 data. Legacy plain, gzip, and zstd pickle
generations remain readable and migrate to HDF5 on the next normal save. Use
the [result schema](../repo-design/storage/result-schema.md) for independent
inspection; do not infer the encoding from the suffix.

## Preserve or reduce data

Use checkpoint commands instead of manually editing checkpoint directories:

```bash
uv run pyrite checkpoint --help
uv run pyrite checkpoint archive --help
uv run pyrite checkpoint slim --help
```

`gc` removes cases that no longer match current profile resolution; `rm`
targets selected datasets. Both provide previews and confirmation controls.
Review their help before destructive use.

Remote runs can be pulled into the same local checkpoint layout. See
[Running on a cluster](running-on-a-cluster.md) for submission and transfer.
