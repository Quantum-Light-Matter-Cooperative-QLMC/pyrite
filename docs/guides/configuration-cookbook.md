# Configuration cookbook

Use these recipes to configure a run without editing checkpoint data. Commands show intent; consult the [CLI reference](../repo-design/cli/cli-reference.md) for every option.

## Select defaults for this machine

```bash
pyrite config set profile.current standard
pyrite config set remote.target my-cluster
pyrite config set workspace.root /path/to/pyrite-workspace
pyrite config list
```

For a temporary override, use a command option or environment variable. It wins over the persistent store and does not mutate it:

```bash
PYRITE_PROFILE=sub_100keV pyrite run -m hopg --quick
PYRITE_HOME=/scratch/my-run pyrite checkpoint list  # archive shelf in that workspace
```

For the complete inventory, accepted formats, defaults, and operational controls, see the [environment-variable reference](../repo-design/configuration-resolution.md#environment-variable-reference).

## Create and inspect a campaign

```bash
pyrite profile create my-survey
pyrite profile set my-survey --material hopg,hbn
pyrite profile show my-survey
pyrite material show hopg --profile my-survey
```

Set ranges on the profile. When one material needs different ranges, give it its own profile (`pyrite profile create NAME --from PROFILE --material MATERIAL`) rather than a per-material override: overrides silently diverge one material from the profile it appears to follow. `pyrite material set` was removed in 0.6.0 (issue #359); remove existing `[profiles.NAME.overrides.MATERIAL]` range keys by editing the catalog TOML. Inspect the effective material after every edit; `pyrite material show` marks each range `inherited` or `overridden`, and the displayed result, not the TOML fragment alone, is what a run hashes.

## Reuse a named beam

```bash
pyrite beam create bench --transverse-fwhm-mm 0.1
pyrite profile set my-survey --beam bench
pyrite beam show bench
```

Renaming a beam updates profile references and leaves identities unchanged because resolved values, not names, are hashed. Deletion is blocked while a profile references the beam.

## Change one material safely

```bash
pyrite material show hopg --profile my-survey
pyrite profile create hopg-thick --from my-survey --material hopg
pyrite profile set hopg-thick --thickness 50000 --dry-run
pyrite material show hopg --profile hopg-thick
```

After changing a profile or material, existing variant checkpoints are not silently reused as the new dataset. Preview stale data with:

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

Derivation can be remote for heavy work. The stored artifact contains full bounds; do not paste reduced bounds into the catalog.

## Diagnose precedence

Start with `pyrite config list`, then inspect shell variables and the resolved profile/material. Empty configuration environment variables are errors, not a request to fall through. See [Configuration and profile resolution](../repo-design/configuration-resolution.md) for the complete chain and environment-variable reference.

For native GPT time-output electron beams, see [GDF beam import](gpt-gdf-beams.md),including named beam configuration, run overrides, and normalization.
