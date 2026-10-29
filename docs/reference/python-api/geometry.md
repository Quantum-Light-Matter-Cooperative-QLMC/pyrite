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

Keyword-only `roll`, `miscut` and `miscut_azimuth` take explicit scalar
plane-angle quantities, for example
`CrystalCut(surface_hkl=(1, 0, 1), miscut=Q_(0.1, "rad"))`. Each conflicts with
its `*_deg` spelling, reads back as a fresh degree quantity and is replaced by
`with_roll`, `with_miscut` or `with_miscut_azimuth`; the `0 <= miscut < 90°`
limit is unchanged. Nested paths such as `target.cut.miscut` work in sweeps and
optimization studies.

## Layer rotation quantities

`Layer("silicon", 2000, azimuth=Q_(0.125, "turn"))` sets the layer's rotation
relative to the entrance film to 45 degrees. `layer.azimuth` returns a fresh
scalar quantity in degrees; `layer.with_azimuth(Q_(-90, "degree"))` replaces it.
Signed angles and multiple turns retain their configured values. Inputs require
explicit finite plane-angle quantities: bare numbers, dimensionless quantities
and solid angles are refused. Matching `azimuth` and `azimuth_deg` keywords
conflict even when they specify the default zero angle.

The positional/numeric `azimuth_deg` field, `dataclasses.replace`, layer records
and lowering retain degree magnitudes. Indexed sweep and optimization paths
can target `target.layers[1].azimuth` through the owning replacement method.
See [quantity parameter paths](../optimization-studies.md#quantity-parameter-paths).

## Target tilt quantities

`Slab`, `Stack` and the body-role `Target` accept keyword-only `tilt` (polar) and `tilt_azimuth`
plane-angle quantities (scalar for `Target`, scalar or one-dimensional sweep
otherwise):

```python
from pyrite.geometry import Slab
from pyrite.units import Q_

slab = Slab("hopg", 2000, footprint=None, tilt=Q_([0.5, 0.7], "rad"), tilt_azimuth=Q_(180, "deg"))
slab.tilt  # fresh degree quantity, shape (2,)
steeper = slab.with_tilt(Q_(60, "degree"))
```

Read-back returns fresh degree quantities in configured order, without wrapping
signs or turns. Bare numbers, dimensionless quantities and solid angles are
refused; each keyword conflicts with its numeric spelling even at the default.
The zero-tilt and 90° azimuth guards apply unchanged. `Stack.on_substrate`
forwards both keywords. Numeric `tilt_deg`/`tilt_azim_deg` fields, positional
arguments, `dataclasses.replace`, cases and lowering keep degree magnitudes;
sweep and optimization paths can target `target.tilt` and `target.tilt_azimuth`.

## Groove spacing quantities

`BlazedGrooves(spacing=Q_(2, "um"))` requires a scalar length quantity.
`grooves.spacing` returns a fresh quantity in angstroms; use
`grooves.with_spacing(Q_(1, "um"))` for replacement. Import `Q_` from
`pyrite.units`. Legacy positional `spacing_ang` values, numeric records and
geometry lowering retain their existing units and behavior. Supplying both
spellings raises `TypeError`; `dataclasses.replace` uses `spacing_ang`.
