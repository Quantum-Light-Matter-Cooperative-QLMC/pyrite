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

Campaign, checkpoint, run, app-launcher, validation, and performance APIs live
in their named subpackages. The former root module paths remain compatibility
re-exports.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   cxr_mc.materials
   cxr_mc.apps.analyze
   cxr_mc.apps.check
   cxr_mc.apps.export
   cxr_mc.apps.viewer
   cxr_mc.campaign.beam_metrics
   cxr_mc.campaign.config
   cxr_mc.campaign.longitudinal
   cxr_mc.campaign.profiles
   cxr_mc.campaign.sweep
   cxr_mc.campaign.transverse
   cxr_mc.checkpoints.archive
   cxr_mc.checkpoints.campaign_lock
   cxr_mc.checkpoints.checkpoint_cleanup
   cxr_mc.checkpoints.recompute
   cxr_mc.checkpoints.slim
   cxr_mc.detectors
   cxr_mc.energy_grid
   cxr_mc.montecarlo
   cxr_mc.plots
   cxr_mc.perf.performance_analysis
   cxr_mc.perf.performance_profile
   cxr_mc.results
   cxr_mc.runs.blaze
   cxr_mc.runs.run
   cxr_mc.runs.scan
   cxr_mc.remote
   cxr_mc.validation.check_config
   cxr_mc.validation.validation_oracles
   cxr_mc.validation.validation_background
   cxr_mc.cli
```
