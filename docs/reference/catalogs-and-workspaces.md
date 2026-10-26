# Catalogs and workspaces

Use the [configuration guide](../guides/configuration-cookbook.md) for a
runnable setup exercise. This reference owns catalog selection, editable object
storage and workspace layout. Configuration precedence and environment formats
are maintained in [configuration resolution](../repo-design/configuration-resolution.md).

## User objects or a complete catalog

With no complete external catalog selected, user additions from `catalog.user`
(or `PYRITE_USER_CATALOG`, defaulting to the platform configuration directory)
are read over the bundled catalog. The layer holds profiles, beams, detectors,
crystals, media, materials, energy grids, and physics/numerics presets. Bundled
names are read-only and cannot be reused, including case-only differences.
Create editable copies, for example:

```bash
pyrite beam create lab-beam --from example
pyrite detector create clone lab-detector --from reference
pyrite profile create lab-scan --from quickstart
pyrite profile set lab-scan --beam lab-beam --detector lab-detector
```

Object `list` commands report `bundled`, `user` or `catalog` sources in text and
JSON. New definitions use `<table>/<name>.toml` in the writable layer. Remote
sync transfers the layer as `user-catalog/`. For material composition and managed
CIF creation, use [Adding materials](../guides/adding-materials.md).

A selected external catalog is **complete**: it replaces the bundle and disables
the user overlay. All object edits write that catalog in place. Keep laboratory
profiles, instrument objects, filters, samples and their CIFs under separate
version control. Remote submissions stage the selected catalog and run its copy.

## Copy a complete catalog

This installed-package recipe copies the bundled definitions and CIFs into a
new `catalog/` directory in the current directory; it refuses an existing target.
Run it in the laboratory directory you want to maintain:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
import shutil

source = files("pyrite") / "data"
destination = Path("catalog")
shutil.copytree(source / "catalog", destination)
shutil.copytree(source / "cifs", destination / "cifs")
PY
pyrite material validate catalog
pyrite --catalog catalog profile show quickstart
```

Use `--catalog PATH` for one command, `PYRITE_CATALOG` for a shell session, or
`pyrite config set catalog.path PATH` for the persistent choice, in that order
of precedence. An invalid selected path fails instead of falling back. The
selection covers profile, material, beam, detector, filter, grid and run commands.

A directory catalog has `catalog.toml`, `<table>/<name>.toml`, optional `cifs/`,
and `energy-grid-artifacts/`. Custom crystal CIF paths use `cifs/FILE.cif` and
must stay inside that directory, including after symlink resolution. Existing
packaged CIF references still work if no local file of that name exists.
Immutable energy-grid artifacts belong to the selected catalog root. See the
[full schema](../repo-design/materials-catalog-schema.md).

A complete-catalog profile must name a beam. Every runnable profile must supply
explicit scalar detector acceptance or pixel geometry; `--detector reference`
selects the ideal Timepix3 scalar aperture when that object exists in the
catalog. A copied demo already declares its instrument. Inspect with
`pyrite profile show NAME` and `pyrite profile ranges show NAME MATERIAL`.

## Output home

Generated outputs use `<workspace>/pyrite-output/`: `checkpoints/`,
`observations/`, `trajectories/`, `performance/`, `results/`, `figures/`, and
`cache/`. A command's explicit directory option wins for that output kind.
Otherwise workspace selection uses `PYRITE_HOME`, then saved `workspace.root`,
then the nearest ancestor of cwd containing `pyrite-output/`, then cwd.
Set a fixed home for installed-package use with:

```bash
pyrite config set workspace.root /scratch/lab-campaign
pyrite checkpoint list
```

`cache/` contains regenerable reproductions, analysis intermediates, photon-grid
resolutions and viewer renders; it is safe to delete. Checkpoints, observations,
trajectories and exports remain outside it. Remote pulls resolve destinations
from the local workspace. See [dataset identity and storage](../repo-design/storage/dataset-identity-and-storage.md)
for identity-qualified stems, manifests and archive behavior. Moving unchanged
catalog definitions does not change their resolved scientific inputs.

On the first CLI output-path resolution, populated legacy `checkpoints/`,
`observations/`, `performance-profiles/` and `results/` move into the new tree,
with a notice on stderr. Cross-filesystem moves copy and verify before removing
the source. If both locations exist, PyRITE warns and uses the new location
without merging. Help and imports do not move files; symlinked legacy directories
are left alone.

## Input tables in a workspace

An explicit workspace also selects generated cross-section tables and fetched
reference inputs:

| Input | Location under the workspace |
| --- | --- |
| Generated tables | `xsgen/tables/` |
| SBETHE reference data | `xsgen/reference-data/sbethe/sdbase/` |
| EEDL and EADL | `datasets/` |

New writes use the workspace. Without an explicit workspace, the per-user data
directory is the selected input tier. An explicit workspace does not search the
older per-user tables directory. `pyrite tables migrate --dry-run` previews
missing tables that can be copied from that directory; migration never deletes
or overwrites a table. Fetched SBETHE and EEDL/EADL inputs are not copied
automatically: fetch them into the workspace with `pyrite tables fetch`, or copy
them explicitly. Packaged catalog/CIF data remain package-relative.
