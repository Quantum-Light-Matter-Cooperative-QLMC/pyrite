# Observations and acquisitions

Specify what is measured independently of the physical scene. Measurement requests belong here; computed acquired counts belong to `pyrite.results`.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

## Measurement requests

Compose `Observation` with an acquisition request and, when needed,
`pyrite.config.AngularSampling`. Geometry belongs to [detectors](detectors.md),
response is independent of acquisition, and computed counts belong to
[results](results.md#acquired-counts).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.observations.Observation
   pyrite.observations.Acquisition
   pyrite.observations.NativeAcquisition
```

## Advanced resolved observation and planning

`ResolvedObservation`, `ObservationPlan` and `PixelSampling` remain
advanced-stable supported imports. They describe lowering and evaluated
sampling, rather than the preferred user request. Construct `Observation`
for ordinary workflows; grouping these interfaces here does not withdraw support.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.observations.ResolvedObservation
   pyrite.observations.ObservationPlan
   pyrite.observations.PixelSampling
```

## Compatibility imports

`PixelScorer` is the same class as `pyrite.config.AngularSampling`; prefer
the configuration spelling for angular quadrature. `AcquisitionBatch` belongs
to `pyrite.results`; this observation import and the root import remain
supported, with identical objects and advanced-stable status.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.observations.PixelScorer
   pyrite.observations.AcquisitionBatch
```
