# Reuse downstream filter plates

A named filter is a *definition*, not a physical instance. It uses the same
finite attenuation-only `FilterPlate` geometry as an inline profile filter.

```bash
pyrite filter create si-window --material silicon --thickness-mm 0.1 \
  --size-mm 20 20 --distance-mm 50
pyrite filter attach my-profile si-window --instance-name window-a
pyrite filter attach another-profile si-window --distance-mm 65
pyrite filter show si-window
```

In a directory catalog, `filters/si-window.toml` contains:

```toml
material = "silicon"   # a crystal or homogeneous medium key in your catalog
thickness_mm = 0.1
size_mm = [20, 20]
distance_mm = 50
```

A profile carries *ordered placements*:

```toml
[[profiles.my-profile.filters]]
ref = "si-window"
name = "window-a"
distance_mm = 65        # optional override of the default pose
```

`pyrite.geometry.load_filter("si-window")` returns a frozen `FilterPlate`;
use `dataclasses.replace` to customize it in the Python API. Preset names
and descriptions never enter physical observation identity. The plate's
resolved dimensions, material, pose and ordered position still do.

`pyrite filter rename OLD NEW` updates referencing placements; `delete`
refuses while a profile refers to the definition. `--dry-run` on mutations
shows the catalog diff before any changes are written. Bundled objects cannot
be overwritten through the regular CLI.

Downstream plates only attenuate photons under the existing model; they are
not arbitrary electron-interacting bodies. This does not create a mesh/CSG
navigator or enable fluorescence/secondary generation inside a filter.
