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
   cxr_mc.analyze
   cxr_mc.archive
   cxr_mc.blaze
   cxr_mc.check
   cxr_mc.check_config
   cxr_mc.beam_metrics
   cxr_mc.config
   cxr_mc.detectors
   cxr_mc.energy_grid
   cxr_mc.montecarlo
   cxr_mc.plots
   cxr_mc.profiles
   cxr_mc.results
   cxr_mc.run
   cxr_mc.remote
   cxr_mc.rebrem
   cxr_mc.reline
   cxr_mc.slim
   cxr_mc.sweep
   cxr_mc.validation_oracles
   cxr_mc.validation_background
   cxr_mc.cli
   cxr_mc.scan
   cxr_mc.export
```
