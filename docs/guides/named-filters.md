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

## Reuse an ordered set of plates

```bash
pyrite filter create-sequence window-pair si-window kapton-window
pyrite filter attach my-profile window-pair
pyrite profile filter list my-profile
```

`create-sequence` copies the member plates, so later edits to `si-window` do
not change `window-pair`. A sequence expands in place in beam order, and each
plate keeps its own absolute pose; a placement takes no overrides. To change
one plate for a single profile, run `pyrite filter detach my-profile window-pair`
(or its index) first. This writes equivalent inline plates and leaves observation
identity unchanged. In Python, use `pyrite.geometry.load_filters("window-pair")`.

## Reuse a target

```bash
pyrite target create thin-si --material silicon --thickness-ang 20000 --tilt-deg 30
pyrite target create si-on-oxide --layer silicon 20000 --layer sio2 30000
pyrite target attach my-profile thin-si --dry-run
pyrite target attach my-profile thin-si
```

The entrance material must already be a member of the profile; `attach` never
adds members. A target attaches to that material and owns all of its
geometry; profile sweeps of thickness and angle no longer apply to it. Beam
energy, physics, numerics and grids stay with the profile. `attach` and `detach`
print the effective geometry before and after; `target detach my-profile silicon`
restores the profile geometry. In Python, `pyrite.geometry.load_target("thin-si")`
returns a `Slab` or `Stack`. See the
[schema](../repo-design/materials-catalog-schema.md#named-targets).
