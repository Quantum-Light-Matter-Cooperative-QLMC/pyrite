# Experimental and provisional interfaces

Provisional APIs are not guaranteed stable and do not imply experimentally validated science. See the [experimental lifecycle](../experimental-api.md).

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.optimization.FixedWindowYield
   pyrite.optimization.PyriteEvaluator
   pyrite.optimization.OptimizationStudy
```

## Experimental optimization

The provisional `pyrite.optimization` API operates above immutable Scenes and
existing executors. It is outside the [stable compatibility contract](compatibility.md).
See the [study reference](../../reference/optimization-studies.md) for persistence,
constraints and normalization, and the [visual tutorial](../../guides/optimization-studies.md)
for a complete fixed-window PXR/CBS workflow.

## Photon aperture interfaces

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
