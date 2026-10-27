# Beams and particle distributions

Describe incident beams, particle distributions, and bunch-front tilt. The public `pyrite.beams` interface preserves the established object identities.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.beams.TransverseSpot
   pyrite.beams.BunchFrontTilt
   pyrite.beams.PointSpot
   pyrite.beams.PointBunch
   pyrite.beams.BunchLength
   pyrite.beams.Offsets
   pyrite.beams.GdfSource
   pyrite.campaign.beam_metrics.capture_beam_metrics
```

`Spot` is the supported compatibility alias for `TransverseSpot`.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.beams.Spot
```

## Electron bunch-front tilt

`BunchFrontTilt` adds a position-dependent arrival time without rotating the
beam. A front parallel to the target entrance face need not phase-match
radiation in the observation direction. `Scene.phase_matched_bunch_front_tilt`
returns a front for one explicit reflection and direction on a flat single slab.
See [front timing, supported sources and examples](../../reference/python-scene-recipes.md#tilt-the-bunch-front-and-crystal-cut).

## Saved beam diagnostics

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `capture_beam_metrics` | Beam diagnostics, advanced-stable | Reads a bounded deterministic prefix of stored initial particles (including missed entries); returns beam metrics plus sampled/available counts. Metric field units follow `BeamMetrics`; invalid prefix limits or missing artifact data raise. No new transport. |
