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

`pyrite.materials.CATALOG` is the immutable bundled `MaterialCatalog`.
`CRYSTALS`, `MATERIALS`, and `MATERIAL_LABELS` are compatibility projections,
not independent registries and not part of the supported API.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.materials.MaterialCatalog
   pyrite.materials.MaterialConfigError
   pyrite.materials.CrystalInfo
   pyrite.materials.CrystalSpec
   pyrite.materials.MediumSpec
   pyrite.materials.MaterialSpec
   pyrite.materials.ScanSpec
   pyrite.materials.LayerSpec
   pyrite.materials.load_material_catalog
   pyrite.materials.atomic.cromer_mann_f0
   pyrite.materials.atomic.henke_dispersion
   pyrite.materials.atomic.atomic_form_factor
   pyrite.materials.crystal.reciprocal_g_vector
   pyrite.materials.crystal.structure_factor
   pyrite.materials.crystal.chi_g
   pyrite.materials.crystal.U_g
   pyrite.materials.crystal.absorption_length_ang
   pyrite.materials.crystal.dominant_reflections
```

## Simulation kernels

These are the supported low-level simulation entry points. Inputs and outputs
use the units stated in their docstrings; prefer keyword arguments for optional
model controls.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.montecarlo.transport.simulate_trajectories
   pyrite.montecarlo.spectrum.mc_spectrum
   pyrite.montecarlo.spectrum.mc_spectrum_solid_angle
   pyrite.montecarlo.spectrum.mc_brem_spectrum
   pyrite.montecarlo.detector.detector_efficiency
   pyrite.montecarlo.detector.convolve_detector
   pyrite.montecarlo.runner.run_case
   pyrite.montecarlo.runner.run_cases
```

## Detector response

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.detectors.DetectorSpec
   pyrite.detectors.eaglexo_response.EagleResponse
   pyrite.detectors.eaglexo_response.get_response
   pyrite.detectors.timepix_response.TimepixResponse
   pyrite.detectors.timepix_response.get_response
   pyrite.detectors.grating.Grating
   pyrite.detectors.grating.SimpleCCD
   pyrite.detectors.grating.disperse_spectrum
   pyrite.detectors.grating.detected_image
```

## Result analysis

Result dictionaries are currently the supported interchange type. Their
storage identity and lifecycle are documented in [dataset identity and
storage](repo-design/storage/dataset-identity-and-storage.md).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.results.Settings
   pyrite.results.records
   pyrite.results.filter_results
   pyrite.results.select_results
   pyrite.results.best_azimuth
   pyrite.results.line_metrics
   pyrite.results.summary_table
```
