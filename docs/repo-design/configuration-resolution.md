# Configuration and profile resolution

PyRITE has three distinct configuration layers. Keeping them separate prevents
surprising runs and makes dataset identity reproducible.

1. **CLI context** selects the current campaign profile, remote target, and
   workspace root.
2. **Catalog profiles** select campaign grids, membership, beam, detector, and
   emission policy from `materials.toml`.
3. **Fidelity presets** (`full` or `survey`) set workload and reduce resolved
   grids. Fidelity is not a catalog profile.

## CLI context precedence

Each context value resolves independently in this order:

1. an option supplied for this command;
2. a `PYRITE_*` environment variable;
3. its legacy `CXR_*` environment variable;
4. the persistent user config store;
5. the built-in default.

| key | environment | built-in |
|---|---|---|
| `profile.current` | `PYRITE_PROFILE`, `CXR_PROFILE` | `standard` |
| `remote.target` | `PYRITE_REMOTE_HOST`, `CXR_REMOTE_HOST` | `qlmc` |
| `workspace.root` | `PYRITE_HOME`, `CXR_HOME` | current directory |

`pyrite config list` shows both effective values and their sources. `config
set` writes atomically to Click's platform-specific user configuration
directory. A legacy store is read only when the current store does not exist;
the next write targets the current store.

The workspace resolver uses an explicit command path first, then the effective
`workspace.root`. Packaged catalog and CIF data remain package-relative and are
never redirected into the workspace.

## Catalog-profile resolution

`pyrite run [PROFILE]` uses the positional profile when present; otherwise it
uses `profile.current`. The selected `[profiles.NAME]` row supplies shared scan
values. `[profiles.NAME.overrides.MATERIAL]` replaces values for one material.
An explicit membership list limits the campaign; an absent list means every
configured material. `-m/--material` narrows that resolved membership and does
not create another profile.

Named beam references resolve to beam values before hashing. Detector settings
inherit the selected profile's block, then the `standard` detector block, then
the built-in detector defaults. An explicit emission policy overrides the
fidelity preset's default, and an explicit `xray_dispersion`
(`vacuum`/`refractive`) overrides its photon dispersion model the same way.
Energy-grid references are verified and resolved
for the selected profile before a material sweep is built.

## Run resolution order

For each material, the run path performs the following conceptual sequence:

```text
CLI context -> catalog profile -> material override -> named beam/detector
            -> resolved detector energy bins -> fidelity preset
            -> explicit run overrides -> cases and identity
```

All values are resolved before the dataset identity is computed. Changing a
resolved physics or workload input therefore selects a different dataset;
changing only a display label does not. See [Dataset identity and
storage](storage/dataset-identity-and-storage.md).

## Inspect before compute

```bash
pyrite config list
pyrite profile show standard
pyrite beam show default
pyrite material show hopg --profile standard
```

Use [Configuration cookbook](../guides/configuration-cookbook.md) for common
edits and the generated [CLI reference](cli/cli-reference.md) for exact option
contracts.
