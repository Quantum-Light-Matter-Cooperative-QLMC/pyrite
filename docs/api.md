# Supported Python API

The objects explicitly listed on this page are PyRITE's supported library
surface. They follow semantic-versioning compatibility: removal or an
incompatible signature/meaning change requires a documented deprecation or a
major release. Public objects not listed here are usable but provisional.

Names beginning with `_`, CLI command callbacks, app launchers, remote-job
orchestration, developer tools, and package-facade re-exports retained only for
old imports are internal or compatibility surfaces. They may change without a
library deprecation. The command-line contract is documented separately in the
[CLI reference](repo-design/cli/cli-reference.md).

## Materials and crystallography

`cxr_mc.materials.CATALOG` is the immutable bundled `MaterialCatalog`.
`CRYSTALS`, `MATERIALS`, and `MATERIAL_LABELS` are compatibility projections,
not independent registries and not part of the supported API.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   cxr_mc.materials.MaterialCatalog
   cxr_mc.materials.MaterialConfigError
   cxr_mc.materials.CrystalInfo
   cxr_mc.materials.CrystalSpec
   cxr_mc.materials.MediumSpec
   cxr_mc.materials.MaterialSpec
   cxr_mc.materials.ScanSpec
   cxr_mc.materials.LayerSpec
   cxr_mc.materials.load_material_catalog
   cxr_mc.materials.atomic.cromer_mann_f0
   cxr_mc.materials.atomic.henke_dispersion
   cxr_mc.materials.atomic.atomic_form_factor
   cxr_mc.materials.crystal.reciprocal_g_vector
   cxr_mc.materials.crystal.structure_factor
   cxr_mc.materials.crystal.chi_g
   cxr_mc.materials.crystal.U_g
   cxr_mc.materials.crystal.absorption_length_ang
   cxr_mc.materials.crystal.dominant_reflections
```

## Simulation kernels

These are the supported low-level simulation entry points. Inputs and outputs
use the units stated in their docstrings; prefer keyword arguments for optional
model controls.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   cxr_mc.montecarlo.transport.simulate_trajectories
   cxr_mc.montecarlo.spectrum.mc_spectrum
   cxr_mc.montecarlo.spectrum.mc_spectrum_solid_angle
   cxr_mc.montecarlo.spectrum.mc_brem_spectrum
   cxr_mc.montecarlo.detector.detector_efficiency
   cxr_mc.montecarlo.detector.convolve_detector
   cxr_mc.montecarlo.runner.run_case
   cxr_mc.montecarlo.runner.run_cases
```

## Detector response

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   cxr_mc.detectors.DetectorSpec
   cxr_mc.detectors.eaglexo_response.EagleResponse
   cxr_mc.detectors.eaglexo_response.get_response
   cxr_mc.detectors.timepix_response.TimepixResponse
   cxr_mc.detectors.timepix_response.get_response
   cxr_mc.detectors.grating.Grating
   cxr_mc.detectors.grating.SimpleCCD
   cxr_mc.detectors.grating.disperse_spectrum
   cxr_mc.detectors.grating.detected_image
```

## Result analysis

Result dictionaries are currently the supported interchange type. Their
storage identity and lifecycle are documented in [dataset identity and
storage](repo-design/storage/dataset-identity-and-storage.md).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   cxr_mc.results.Settings
   cxr_mc.results.records
   cxr_mc.results.filter_results
   cxr_mc.results.select_results
   cxr_mc.results.best_azimuth
   cxr_mc.results.line_metrics
   cxr_mc.results.summary_table
```
