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

## Scene simulation

The root package exposes the preferred high-level API. A `Scene` contains one
scalar `Beam`, target, and `Detector`; `Numerics` contains sampling and execution
controls. `simulate` lowers those objects to the established typed `Case` and
calls the existing Monte Carlo runner directly. It neither reads nor writes a
checkpoint.

```python
import pyrite as pr

beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0)
detector = pr.Detector()
result = pr.simulate(
    beam,
    target,
    detector,
    numerics=pr.Numerics(n_electrons=450, n_electrons_brem=100),
)
result.spectrum
result.provenance["identity_digest"]
```

`Result.spectrum` and `Result.background` are intrinsic photon-density arrays
per incident electron per eV per sr. Their coordinates are `energy_eV` and
`background_energy_eV`. `Result.provenance` records the resolved scene,
numerics, content identity, backend/device, and library versions.

`Sweep` expresses a Cartesian product as ordered paths into a scalar base
scene. Paths are checked when the sweep is constructed, including indexed
segments:

```python
sweep = pr.Sweep(
    base=pr.Scene(beam, target, detector),
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
    },
)
cases = sweep.cases(pr.Numerics())
```

For a `Stack`, paths such as `target.layers[1].thickness_ang` address a
particular layer. A misspelled field or out-of-range index raises at `Sweep`
construction.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.Beam
   pyrite.Slab
   pyrite.Stack
   pyrite.Layer
   pyrite.Footprint
   pyrite.BlazedGrooves
   pyrite.Scene
   pyrite.Sweep
   pyrite.Convergence
   pyrite.Numerics
   pyrite.Analysis
   pyrite.Result
   pyrite.simulate
```

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
   pyrite.materials.MaterialValidationSpec
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
model controls. `Case` is the frozen, validated input record; its `to_dict()`
method preserves the legacy mapping representation for serialization.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.montecarlo.Case
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

   pyrite.detectors.Detector
   pyrite.detectors.EnergyBins
   pyrite.detectors.Timepix3
   pyrite.detectors.EagleXO
   pyrite.detectors.LegacyEDS
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

Single-shot simulations return `pyrite.Result`. Resumable campaign checkpoints
continue to use result dictionaries; their storage identity and lifecycle are
documented in [dataset identity and
storage](repo-design/storage/dataset-identity-and-storage.md). `Settings` is the
D7 compatibility surface for those checkpoint-analysis functions; new code
uses `Analysis` for presentation controls.

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
