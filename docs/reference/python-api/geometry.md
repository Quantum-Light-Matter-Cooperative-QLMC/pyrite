# Geometry and targets

Construct targets, crystalline cuts, solid bodies, footprints, and filters. See [scene recipes](../python-scene-recipes.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.geometry.Slab
   pyrite.geometry.Stack
   pyrite.geometry.SubstanceRef
   pyrite.geometry.RectangularBody
   pyrite.geometry.Filter
   pyrite.geometry.Layer
   pyrite.geometry.Footprint
   pyrite.geometry.BlazedGrooves
   pyrite.geometry.CrystalCut
```

## Shared bodies and explicit roles

`RectangularBody` describes a solid; `Target` and `Filter` assign its scene role.
For quantity construction, import `Q_` from `pyrite.units`:

```python
from pyrite.geometry import RectangularBody, SubstanceRef
from pyrite.units import Q_

body = RectangularBody(
    SubstanceRef("crystal", "silicon"),
    size=Q_([5, 6], "mm"),
    thickness=Q_(100, "um"),
)
body.size.to("cm")  # detached two-component quantity
body.thickness.to("um")  # scalar quantity
thicker = body.with_thickness(Q_(200, "um"))
wider = body.with_size(Q_([7, 8], "mm"))
```

Semantic inputs require Pint lengths; size has exactly two components and
thickness is scalar. Read-back uses mm. Numeric `size_mm` and `thickness_mm`
fields, positional arguments, sweep paths and serialization remain supported.
Each semantic keyword conflicts with its corresponding numeric spelling;
`dataclasses.replace` uses the numeric fields.

See [body construction and limits](../../reference/python-scene-recipes.md#reuse-bodies-as-targets-and-filters).

## Crystal cuts

`CrystalCut` sets the lattice orientation relative to a flat slab's entrance
face. Target tilt moves face and lattice together. See the
[cut convention and limits](../../reference/python-scene-recipes.md#crystal-cuts).

## Groove spacing quantities

`BlazedGrooves(spacing=Q_(2, "um"))` requires a scalar length quantity.
`grooves.spacing` returns a fresh quantity in angstroms; use
`grooves.with_spacing(Q_(1, "um"))` for replacement. Import `Q_` from
`pyrite.units`. Legacy positional `spacing_ang` values, numeric records and
geometry lowering retain their existing units and behavior. Supplying both
spellings raises `TypeError`; `dataclasses.replace` uses `spacing_ang`.
