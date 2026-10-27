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

## Transverse spot dimensions with units

`TransverseSpot` (or `Spot`) accepts explicit, keyword-only Pint lengths as
`fwhm_x` and `fwhm_y`, or the existing numeric `fwhm_x_mm` and `fwhm_y_mm`
fields. The spellings for *each axis* are mutually exclusive. A missing y
width means the spot is round; `fwhm_y` reads the effective width even when
the numeric `fwhm_y_mm` field is `None`.

```python
from pyrite.beams import Spot
from pyrite.units import Q_

spot = Spot(fwhm_x=Q_(200, "micrometer"))  # round: 0.2 mm on both axes
x_um = spot.fwhm_x.to("micrometer")
y_um = spot.fwhm_y.to("micrometer")
wider = spot.with_fwhm_x(Q_(0.3, "mm"))  # stays round
elliptical = spot.with_fwhm_y(Q_(400, "micrometer"))
round_again = elliptical.with_fwhm_y(None)
```

Read-back returns fresh quantities in mm. Semantic inputs require positive,
finite scalar lengths; bare numbers and dimensionally incompatible quantities
are refused. Legacy positional construction, `Spot` identity, numeric
`dataclasses.replace`, beam lowering and saved specifications do not change.
`dataclasses.replace(spot, fwhm_x=...)` is **not** supported because numeric
fields are forwarded; use `with_fwhm_x` / `with_fwhm_y` instead.

## Electron bunch-front tilt

`BunchFrontTilt` adds a position-dependent arrival time without rotating the
beam. A front parallel to the target entrance face need not phase-match
radiation in the observation direction. `Scene.phase_matched_bunch_front_tilt`
returns a front for one explicit reflection and direction on a flat single slab.
See [front timing, supported sources and examples](../../reference/python-scene-recipes.md#tilt-the-bunch-front-and-crystal-cut).

## Bunch timing with units

Construct `BunchLength(rms_duration=quantity)` with a finite positive scalar
time quantity. Construct `Offsets(arrival_offsets=quantity)` with a nonempty
one-dimensional time array; signed offsets and zero are valid.

```python
from pyrite.beams import BunchLength, Offsets
from pyrite.units import Q_

bunch = BunchLength(rms_duration=Q_(0.25, "ps"))
duration_ps = bunch.rms_duration.to("ps")
longer = bunch.with_rms_duration(Q_(500, "fs"))
timed = Offsets(arrival_offsets=Q_([-125, 0, 250], "fs"))
offsets_ps = timed.arrival_offsets.to("ps")
retimed = timed.with_arrival_offsets(Q_([-250, 0, 500], "fs"))
```

Read-back uses the shared Pint registry and canonical fs; offset magnitudes
are fresh arrays. The legacy `rms_fs` and `offsets_fs` fields, positional
constructors and `shape` labels remain supported. Passing both numeric and
semantic spellings raises `TypeError`, including with equivalent values.
`dataclasses.replace` and nested numeric sweep paths continue to use the
numeric field names; use the methods above for quantity-aware replacement.
For a beam, replace its `longitudinal` descriptor with the new timing object.

Beam lowering and saved records contain numeric fs values. Reading quantities
after loading neither samples arrival times nor changes RNG state. See the
[timing boundary contract](../../repo-design/physical-units.md#bunch-timing-quantity-interfaces)
for shapes, representability, replacement and compatibility.

## Saved beam diagnostics

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `capture_beam_metrics` | Beam diagnostics, advanced-stable | Reads a bounded deterministic prefix of stored initial particles (including missed entries); returns beam metrics plus sampled/available counts. Metric field units follow `BeamMetrics`; invalid prefix limits or missing artifact data raise. No new transport. |
