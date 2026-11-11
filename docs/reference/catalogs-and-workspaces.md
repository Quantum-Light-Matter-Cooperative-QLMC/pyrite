# Catalogs and workspaces

Use the [configuration guide](../guides/configuration-cookbook.md) for a
runnable setup exercise. This reference owns catalog selection, editable object
storage and workspace layout. Configuration precedence and environment formats
are maintained in [configuration resolution](../repo-design/configuration-resolution.md).

## User objects or a complete catalog

With no complete external catalog selected, user additions from `catalog.user`
(or `PYRITE_USER_CATALOG`, defaulting to the platform configuration directory)
are read over the bundled catalog. The layer holds profiles, beams, detectors,
crystals, media, materials, energy grids, physics/numerics presets, reusable
filter plates, targets, gratings and tabulated `curves/`. Bundled names are read-only and cannot be reused, including case-only differences.
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

## Private instrument catalogs

Instrument data that must stay private, such as digitized datasheet curves or
mount geometry under a nondisclosure agreement, lives in the user layer, kept
as a checkout of a private laboratory repository. Nothing in it is copied into
PyRITE.

```bash
git clone git@example.org:lab/instrument-catalog.git ~/lab/instrument-catalog
pyrite config set catalog.user ~/lab/instrument-catalog   # or PYRITE_USER_CATALOG
```

The checkout uses the user-layer layout:

```text
instrument-catalog/
  curves/lab-ccd-qe.csv        # tabulated curve with provenance header
  curves/lab-ff-efficiency.toml
  gratings/lab-ff.toml         # efficiency_curve = "lab-ff-efficiency"
  detectors/lab-ccd.toml       # response.kind = "alexs", qe_curve = "lab-ccd-qe"
```

Mark curves from restricted sources with `confidential = true`. Use object
names that differ from bundled ones. `pyrite detector list` reports these
objects as `user`, and `pyrite material validate` rejects bad curve units,
non-increasing abscissae, out-of-range values and paths that leave `curves/`.
`pyrite remote` ships the layer, including `curves/`, as `user-catalog/`, so
keep the remote box's access to that data in mind. A complete external catalog
(`--catalog`) can hold the same `curves/` and `gratings/` entries instead. The
file formats are in the [schema reference](../repo-design/materials-catalog-schema.md#instrument-curves-and-gratings).

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

A directory catalog has `catalog.toml`, `<table>/<name>.toml`, optional `cifs/`
and `curves/`, and `energy-grid-artifacts/`. Custom crystal CIF paths use `cifs/FILE.cif` and
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
