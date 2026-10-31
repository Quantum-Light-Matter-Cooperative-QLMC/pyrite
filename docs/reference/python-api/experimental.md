# Experimental and provisional interfaces

Provisional APIs are not guaranteed stable and do not imply experimentally validated science. See the [experimental lifecycle](../experimental-api.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

## Experimental optimization

The provisional `pyrite.optimization` API operates above immutable Scenes and
existing executors. It is outside the [stable compatibility contract](compatibility.md).
See the [study reference](../../reference/optimization-studies.md) for persistence,
constraints and normalization, and the [optimization tutorial](../../notebooks/specialized/11-optimizing-xray-source.ipynb)
for a bounded fixed-window PXR/CBS search.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.optimization.Integer
   pyrite.optimization.ParameterSpace
   pyrite.optimization.Objective
   pyrite.optimization.Constraint
   pyrite.optimization.OptunaOptimizer
   pyrite.optimization.FixedWindowYield
   pyrite.optimization.PyriteEvaluator
   pyrite.optimization.OptimizationStudy
```

### CUDA execution tuning

`tune_kernels` tunes execution controls (`KernelTuning`), not physical
parameters, and is separate from `OptimizationStudy`. See
[tune CUDA execution controls](../../guides/batch-chunk-tuning.md).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.optimization.tune_kernels
   pyrite.optimization.KernelSearchSpace
   pyrite.optimization.KernelTrial
   pyrite.optimization.KernelTuningResult
   pyrite.optimization.KernelPhaseResult
   pyrite.optimization.TuningProgress
```

### Separate transport and spectrum phases

`simulate_transport` runs only electron transport of a scalar-detector scene;
`score_spectra` scores its kept segments, as often as needed, without
transporting again. `SimulationProgress` is the event type of their
`progress=` callbacks and of `pyrite.simulate(..., progress=...)`. See
[run the phases yourself](../../guides/batch-chunk-tuning.md#run-the-phases-yourself).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.phases.simulate_transport
   pyrite.phases.score_spectra
   pyrite.phases.TransportedScene
   pyrite.phases.SimulationProgress
```

## Photon aperture interfaces

The governed `pyrite.experimental.observations` namespace exposes the bounded
photon simulation and reduced-artifact operations below. They remain
**experimental**, with the restrictions and normalization described here and
in their linked contracts.

### Continuum aperture observations

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.experimental.observations.PhotonContinuumObservation
   pyrite.experimental.observations.simulate_photon_continuum
   pyrite.experimental.observations.PhotonContinuumTally
   pyrite.experimental.observations.save_photon_continuum
   pyrite.experimental.observations.load_photon_continuum
```

### Physical-event aperture observations

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.experimental.observations.PhotonEventObservation
   pyrite.experimental.observations.simulate_photon_events
   pyrite.experimental.observations.PhotonEventTally
   pyrite.experimental.observations.PhotonRelaxationBudget
   pyrite.experimental.observations.save_photon_events
   pyrite.experimental.observations.load_photon_events
```

### Selection and persistence contracts

The provisional photon-continuum reference has a separate
[aperture tally artifact](../../research/physics/photon-scene-handoff.md#explicit-reduced-observation-persistence):
`PhotonContinuumObservation.save(path)` and
`pyrite.photon_observation.load_photon_continuum(path)` preserve reduced
aperture spectra and provenance. These experimental objects are outside the
stable API contract. Their aperture-integrated spectra have units
photons/(incident electron eV) and must not be added to `Result.brem`.
The experimental low-level
[`score_photon_sources`](../../research/physics/photon-secondary-feedback.md#explicit-event-source-aperture-scoring)
boundary also scores explicit characteristic/annihilation source batches using
the original incident-electron denominator.
[`pyrite.photon_events.simulate_photon_events`](../../research/physics/photon-secondary-feedback.md#scene-physical-event-source-boundary)
selects one physical event component from a Scene with retained charged cascades.
Both photon Scene entry points accept independent `physics=`, `numerics=`,
`quantum=` and `observation=` through the shared simulation resolver. Omitted
quantum inherits legacy Scene settings; `quantum=None` disables that setup.
Their ideal, unpixelated aperture restrictions still apply to observations.
Both remain outside the supported API contract. Physical event observations
have a separate [reduced event tally artifact](../../research/physics/photon-secondary-feedback.md#reduced-event-observation-persistence),
loaded with `pyrite.photon_event_observation.load_photon_events(path)`.
These analysis artifacts are not production checkpoints or replay inputs.

The experimental `Physics(photon_transport=PhotonTransport(...))` selector now
adds a separate `Result.aperture` observation through the ordinary scalar
simulation path, using one primary transport. `aperture_dataframe()` exposes
histogram densities and bin counts; `current_nA=` scales to photons/s without
reapplying acceptance. Enabled Case/checkpoint/CAS records preserve the aperture
and its original pins; full artifact replay is supported and bounded replay is
refused. See the [selection, units and persistence contract](../../repo-design/compute/photon-aperture.md).
`PhotonTransport` and the aperture APIs remain provisional; pixel response,
catalog/CLI selectors and full-track external validation remain open.

### Scalar simulation selector

`PhotonTransport` is the experimental selector accepted through `Physics`.
Its root import remains governed pending canonical ownership review; documenting
it here does not introduce a new configuration import.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.PhotonTransport
```

## Provisional optical diagnostic response

`pyrite.materials.optical.OpticalResponse`, `gold_rakic_response`, and
`pyrite.energy_grid.component_support.{ComponentSupport,component_grid}` are
provisional source-only diagnostic APIs. They require explicit passive scalar
response and per-component supported bands; they do not lower the production
X-ray floor or supply optical detector/transport models. See
[explicit optical TR response](../../physics/radiation-physics/transition-radiation.md#explicit-optical-response-for-bounded-diagnostics)
for units, sample/provenance metadata, interpolation policy and an example.

`pyrite.materials.optical.fused_silica_franta_response()` provides pinned
fitted fused-silica complex-index data with absorption retained.
`OpticalResponse.require_band(lower_eV, upper_eV)` requires whole-interval
coverage within one segment, preventing integration across unsupported gaps.
These provisional inputs do not establish production Cherenkov validity or
detector calibration; see the [reference-input contract](../../research/physics/production-cherenkov.md#fused-silica-reference-input).

`pyrite.detectors.optical_energy.CalibratedPulseEnergyDetector.predict_batch`
integrates collected photon densities shaped `(..., energy)` on a shared
nonuniform energy grid. Its `OpticalPulseBatchReading` carries per-pulse
joules, isolated-pulse volts, noise/saturation and optional average watts over
the batch axes. NumPy/CuPy inputs select the backend; `predict` retains the
single-spectrum interface returning host scalars. Calibration remains explicit
and bounded; these provisional operators supply no measured response by default.
`OpticalSpectralCoherency`, calibrated Jones transfers and
`polarized_interferogram` preserve that backend through the forward reductions.
See [device operators and transfer boundaries](../../physics/radiation-physics/optical-wavefront-and-detectors.md#device-resident-field-operators-and-cuda-parity).

Finite PEC source entry points in `pyrite.montecarlo.spectrum.optical_finite_screen`
and `optical_finite_bunch`, `optical_finite_collection.finite_screen_bunch_plane_band`,
and the three `runner.finite_screen_emission.case_finite_screen_*` adapters
accept an explicit `xp=np` (default) or active CuPy namespace. Meshes, Bessel
evaluation, complex fields and streamed self/pair moments stay on that backend;
screen objects, energy-node iteration, source-law metadata and formation guards
remain host control. `FiniteScreenBunchAngular.collect_cone(to_host=False)`
retains its scalar on device; the default returns a host float. Device band
arrays can feed `CalibratedPulseEnergyDetector.predict_batch` directly.
