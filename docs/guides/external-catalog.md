# External catalogs and workspace data

PyRITE ships a complete example catalog in its package. Keep laboratory profiles,
instrument geometry, filters, and custom samples in a separately versioned catalog
directory. The selected catalog is complete: PyRITE does not merge it with the
bundled catalog. A profile's filter plates and physical pixel detector stay in
that profile's TOML file.

To start from the bundled definitions, copy the catalog and its CIF files outside
the PyRITE checkout, then edit the copy:

```bash
mkdir -p ~/pyrite-lab/catalog
cp -a src/pyrite/data/catalog/. ~/pyrite-lab/catalog/
cp -a src/pyrite/data/cifs ~/pyrite-lab/catalog/
pyrite material validate ~/pyrite-lab/catalog
pyrite config set catalog.path ~/pyrite-lab/catalog
pyrite config get catalog.path
```

The directory contains `catalog.toml`, `<table>/<name>.toml` object files, an
optional `cifs/` directory, and `energy-grid-artifacts/`. Custom crystal CIF
paths use `cifs/<file>.cif` and must stay inside that directory. Existing
packaged CIF references continue to work when the catalog copy has no local
file of that name. Energy-grid artifacts live under the selected catalog root.

`--catalog PATH` selects a complete catalog for one command; `PYRITE_CATALOG`
selects one for a shell session; `catalog.path` in the user configuration is the
persistent choice. Their precedence is command option, environment variable,
saved setting, bundled default. For example:

```bash
pyrite --catalog ~/pyrite-lab/catalog profile list
pyrite profile filter list my_scan
pyrite run my_scan -m my_sample
```

The same selection controls profile, material, beam, detector, filter, energy-grid,
and run commands. Edits write the selected catalog. An invalid selected path
fails instead of falling back to bundled definitions. Remote submissions stage
the selected catalog in the remote checkout and run against that copy.

The bundled `default` beam and detector are examples. Give each lab profile its
own with `pyrite profile set NAME --beam BEAM --detector DETECTOR`: a run from a
selected catalog whose profile names neither falls back to the bundled examples
and warns, and that fallback is scheduled for removal (see
[Bundled examples and implicit defaults](sweep-profiles.md#bundled-examples-and-implicit-defaults)).

Keep generated output separate from the catalog. Set an external workspace for
checkpoints, observations, and generated cross-section data:

```bash
pyrite config set workspace.root ~/pyrite-lab/workspace
```

With an explicit workspace, generated tables use `workspace/xsgen/tables/`,
fetched SBETHE reference data use
`workspace/xsgen/reference-data/sbethe/sdbase/`, and the fetched EEDL and EADL
files use `workspace/datasets/`. New writes go to the workspace. Tables in the
older per-user data directory (`~/.local/share/pyrite/xsgen/tables` on Linux)
are no longer searched as of PyRITE 0.5.0. `pyrite tables migrate` (try
`--dry-run` first) copies every table the workspace lacks from there into it;
it never deletes or overwrites anything, so remove the old copy yourself
afterwards. With no explicit workspace, the per-user data directory is the
selected tier. Fetched SBETHE data
and the EEDL/EADL datasets are not copied automatically; rerun
`pyrite tables fetch` in the workspace, or copy them.

Version the catalog and its CIFs separately from PyRITE. Checkpoint identity
depends on resolved simulation inputs rather than the local catalog path, so
moving unchanged definitions does not require rerunning completed cases.
