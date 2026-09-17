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

For a task-oriented walkthrough, including detector scoring and the boundary
between in-memory results and checkpoint campaigns, see the
[Python API workflow](guides/python-api-workflow.md).

## Scene simulation

The root package exposes the preferred high-level API. A `Scene` contains one
scalar `Beam`, target, and either the compatibility scalar `Detector` or a
physical `PlanarDetector`; `Numerics` contains sampling and execution controls.
`simulate` lowers those objects to the established typed `Case` and calls the
existing Monte Carlo runner directly. It neither reads nor writes a checkpoint.

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

For a scalar `Detector`, `Result.spectrum` and `Result.background` are
response-free source photon-density arrays per incident electron per eV per sr:
they exclude acceptance scaling, source current, quantum efficiency, and
measured-energy redistribution. For a `PlanarDetector`, the scalar arrays are
instead filter-attenuated, solid-angle-weighted observation averages in the same
per-sr units; they still exclude the configured detector response. Their
coordinates are `energy_eV` and `background_energy_eV`. `Result.provenance`
records the resolved scene, numerics, content identity, backend/device, and
library versions. Coherent or
`both` emission also exposes `coherent_spectrum`; coherent-only simulation
selects that array as `spectrum`. `spectrum` and `coherent_spectrum` are
PXR/CBS only; characteristic radiation is the separate
`characteristic_spectrum`, and `Result.line_total()` returns their sum
(`coherent=True` for the coherent line, `characteristic=False` to omit it).
Before issue #123 `spectrum` already included characteristic radiation. The
high-level API does not currently accept
an external background array, so both `none` and `external` return a zero
background on the resolved continuum grid.

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

`Sweep.expand()` returns ordered `(label, Scene)` pairs; `Sweep.cases()` lowers
them to typed cases but does not execute or persist them. `simulate` accepts
the separate beam, target, and detector components, not a `Scene` argument.
Helpers exported from `pyrite.api` but not listed on this page are provisional
lowering or compatibility seams.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.Beam
   pyrite.Slab
   pyrite.Stack
   pyrite.Layer
   pyrite.Footprint
   pyrite.BlazedGrooves
   pyrite.PlanarPose
   pyrite.PixelGrid
   pyrite.FilterPlate
   pyrite.PlanarDetector
   pyrite.PixelScorer
   pyrite.PixelRayMap
   pyrite.SpectralFactors
   pyrite.SpatialResult
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

Each `MaterialSpec` carries a `MaterialIdentity` — `formula`, `phase`,
`full_name`, `cut`, `cut_frame` — and its `label` is derived from those fields
rather than authored, so a display string cannot drift from the record it
describes. The cut is the crystal's declared slab normal, reduced to its
primitive representative; it is unrelated to the pinned `hkl_families`
reflections.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyrite.materials.MaterialCatalog
   pyrite.materials.MaterialConfigError
   pyrite.materials.MaterialIdentity
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
   pyrite.materials.attenuation.linear_attenuation_inv_mm
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
D7 compatibility surface for those checkpoint-analysis functions. `Analysis`
is the corresponding presentation-control value object and legacy-conversion
target; it is not currently consumed by `simulate`.

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
