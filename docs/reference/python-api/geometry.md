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

`RectangularBody` describes a solid in millimetres; `Target` and `Filter` assign
its scene role. See [body construction and limits](../../reference/python-scene-recipes.md#reuse-bodies-as-targets-and-filters).

## Crystal cuts

`CrystalCut` sets the lattice orientation relative to a flat slab's entrance
face. Target tilt moves face and lattice together. See the
[cut convention and limits](../../reference/python-scene-recipes.md#crystal-cuts).
