# API reference

Auto-generated from the package docstrings. Each module page lists its public
functions and classes, with `[source]` links into the code.

The material entry point is `cxr_mc.materials.CATALOG`, an immutable
`MaterialCatalog`. Its public frozen record types are `CrystalInfo`,
`CrystalSpec`, `MediumSpec`, `MaterialSpec`, `ScanSpec`, and `LayerSpec`;
`load_material_catalog(path)` validates either the bundled offline catalog or an
explicit complete catalog. `CATALOG.material_keys` is an ordered tuple in catalog
declaration order. The mapping-style `CRYSTALS`, `MATERIALS`, and
`MATERIAL_LABELS` names remain compatibility projections, not independent
registries.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   cxr_mc.materials
   cxr_mc.materials.catalog
   cxr_mc.materials.atomic
   cxr_mc.materials.crystal
   cxr_mc.materials.attenuation
   cxr_mc.check_config
   cxr_mc.config
   cxr_mc.detectors
   cxr_mc.detectors.eaglexo_response
   cxr_mc.montecarlo
   cxr_mc.plots
   cxr_mc.results
   cxr_mc.run
   cxr_mc.sweep
   cxr_mc.detectors.grating
   cxr_mc.detectors.timepix_response
   cxr_mc.cli
   cxr_mc.scan
   cxr_mc.export
```
