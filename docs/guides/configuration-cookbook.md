# Configuration cookbook

Use these recipes to configure a run without editing checkpoint data. Commands
show intent; consult the [CLI reference](../repo-design/cli/cli-reference.md)
for every option.

## Select defaults for this machine

```bash
pyrite config set profile.current standard
pyrite config set remote.target qlmc
pyrite config set workspace.root /path/to/pyrite-workspace
pyrite config list
```

For a temporary override, use a command option or environment variable. It
wins over the persistent store and does not mutate it:

```bash
PYRITE_PROFILE=sub_100keV pyrite run -m hopg --fidelity survey
PYRITE_HOME=/scratch/my-run pyrite checkpoint list  # archive shelf in that workspace
```

## Create and inspect a campaign

```bash
pyrite profile create my-survey
pyrite profile set my-survey --material hopg,hbn
pyrite profile show my-survey
pyrite material show hopg --profile my-survey
```

Set shared ranges on the profile, then use `pyrite material set MATERIAL
--profile NAME` for one-material overrides. Inspect the effective material
after every edit; the displayed result, not the TOML fragment alone, is what a
run hashes.

## Reuse a named beam

```bash
pyrite beam create bench --transverse-fwhm-mm 0.1
pyrite profile set my-survey --beam bench
pyrite beam show bench
```

Renaming a beam updates profile references and leaves identities unchanged
because resolved values, not names, are hashed. Deletion is blocked while a
profile references the beam.

## Change one material safely

```bash
pyrite material show hopg --profile my-survey
pyrite material set hopg --profile my-survey --help
pyrite material show hopg --profile my-survey
```

After changing a profile or material, existing variant checkpoints are not
silently reused as the new dataset. Preview stale data with:

```bash
pyrite checkpoint gc --profile my-survey
```

Add `--yes` only after checking the exact selection.

## Derive missing photon grids

If `run` reports missing line bounds:

```bash
PYRITE_PROFILE=my-survey pyrite material energy-grid derive --material hopg
pyrite-dev energy-grid add PATH_FROM_DERIVE --profile my-survey --material hopg
pyrite-dev energy-grid verify
```

Derivation can be remote for heavy work. The stored artifact contains full
bounds; `--fidelity survey` reduces them later at run resolution.

## Diagnose precedence

Start with `pyrite config list`, then inspect shell variables and the resolved
profile/material. Empty configuration environment variables are errors, not a
request to fall through. See [Configuration and profile
resolution](../repo-design/configuration-resolution.md) for the complete chain.
