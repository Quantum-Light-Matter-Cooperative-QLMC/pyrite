# Detectors and response

Separate angular acceptance, physical detector geometry, and instrument response. `AngularDetector`, `Timepix3Response` and `EagleXOResponse` are supported semantic aliases for existing classes, not new physics.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

## Physical detector geometry

`PlanarPose`, `PixelGrid` and `PlanarDetector` describe the physical sensor. Passive plates belong to [geometry](geometry.md); numerical angular scoring belongs to [configuration](config.md#angular-sampling).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.PlanarPose
   pyrite.detectors.PixelGrid
   pyrite.detectors.PlanarDetector
```

## Angular acceptance and energy selection

`Detector` is the established scalar angular observation interface, and the experimental `PointDetector` is the same observation along one direction with zero acceptance (source densities per sr, no counting scale); see the [semantic aliases](#semantic-response-aliases). `EnergyBins` is detector-facing energy selection, not passive geometry.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.Detector
   pyrite.detectors.PointDetector
   pyrite.detectors.AngularAcceptance
   pyrite.detectors.SensorAcceptance
   pyrite.detectors.SlitAcceptance
   pyrite.detectors.EnergyBins
```

## Response specifications

Use `Timepix3Response` and `EagleXOResponse` in new observation configurations (see [semantic aliases](#semantic-response-aliases)). `NativeSpectrum` is a detector response value and `NativeBinnedResponse` is the native-bin capability protocol; acquired count batches belong to [results](results.md#acquired-counts).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.IdealPhotonCounter
   pyrite.detectors.NativeSpectrum
   pyrite.detectors.NativeBinnedResponse
```

## Semantic response aliases

These advanced-stable spellings preserve the identity and behavior of
`Detector`, `Timepix3`, and `EagleXO`, respectively.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.detectors.AngularDetector
   pyrite.detectors.Timepix3Response
   pyrite.detectors.EagleXOResponse
```

## Advanced numeric response and prepared operators

These advanced-stable numeric functions and grid-dependent operators implement response application. The Timepix factory is `pyrite.detectors.timepix_response.get_response`; the Eagle XO factory is `pyrite.detectors.eaglexo_response.get_response`. They prepare operators from grids and response settings, rather than define physical pixel geometry.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.response.detector_efficiency
   pyrite.detectors.response.convolve_detector
   pyrite.detectors.eaglexo_response.EagleResponse
   pyrite.detectors.eaglexo_response.get_response
   pyrite.detectors.timepix_response.TimepixResponse
   pyrite.detectors.timepix_response.get_response
```

## Advanced grating scientific API

`Grating` remains detector scientific API at `pyrite.detectors.grating`, alongside CCD and spectral image operations. Its exploratory scientific applicability is separate from its advanced-stable import contract.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.grating.Grating
   pyrite.detectors.grating.SimpleCCD
   pyrite.detectors.grating.disperse_spectrum
   pyrite.detectors.grating.detected_image
```

## Advanced opt-in quantity helpers

These supported wrappers accept explicit quantities around the numeric scientific interfaces. They do not replace numeric records, defaults or persisted arrays. See [units](units.md).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.quantities.detector_efficiency_q
   pyrite.detectors.quantities.convolve_detector_q
   pyrite.detectors.quantities.grating_q
   pyrite.detectors.quantities.grating_parameters
   pyrite.detectors.quantities.ccd_q
   pyrite.detectors.quantities.ccd_parameters
   pyrite.detectors.quantities.wavelength_q
   pyrite.detectors.quantities.groove_spacing_q
   pyrite.detectors.quantities.diffraction_angle_q
   pyrite.detectors.quantities.angular_dispersion_q
   pyrite.detectors.quantities.grating_throughput_q
   pyrite.detectors.quantities.grating_reflectivity_q
   pyrite.detectors.quantities.detector_position_q
   pyrite.detectors.quantities.resolving_power_q
   pyrite.detectors.quantities.disperse_spectrum_q
   pyrite.detectors.quantities.bin_to_pixels_q
   pyrite.detectors.quantities.detected_image_q
   pyrite.detectors.quantities.coating_number_density_q
   pyrite.detectors.quantities.qe_absorption_q
   pyrite.detectors.quantities.charge_cloud_sigma_q
   pyrite.detectors.quantities.energy_fwhm_q
```

## Compatibility spellings

`Timepix3` and `EagleXO` retain object identity with their semantic response aliases. `LegacyEDS` remains supported. The detector import of `FilterPlate` is retained; prefer `pyrite.geometry.FilterPlate` for passive geometry.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.FilterPlate
   pyrite.detectors.Timepix3
   pyrite.detectors.EagleXO
   pyrite.detectors.LegacyEDS
```

## Detector response

Use these objects for detector efficiency, energy redistribution and grating
images. Physical pixel scoring and acquisition are illustrated in the
[workflow guide](../../reference/python-scene-recipes.md#place-a-finite-filter-over-detector-pixels).

## Angular acceptance and observation with units

Scalar detector angles and spherical-patch acceptance support Pint inputs with
explicit semantic names alongside the established numeric degree/sr fields:

```python
from pyrite.detectors import AngularAcceptance, Detector
from pyrite.units import Q_

patch = AngularAcceptance(polar=Q_(2, "degree"), azimuthal=Q_(0.1, "rad"))
detector = Detector(observation_angle=Q_(0.5, "rad"), acceptance=patch)
detector.observation_angle.to("rad")  # fresh quantity, stored degrees unchanged
rotated = detector.with_observation_angle(Q_(90, "degree"))
resized = patch.with_polar(Q_(4, "degree"))
```

`AngularAcceptance` requires exactly two of `polar`/`polar_deg`,
`azimuthal`/`azimuthal_deg`, or `solid_angle`/`solid_angle_sr`.
Semantic inputs must be real, finite scalar Pint angles or solid angles, not
bare numbers, arrays, or arbitrary dimensionless quantities. The three
semantic properties return only the *specified* values as fresh degree/sr
quantities; the omitted member stays `None` until `resolve()` is called at a
given observation angle. `with_polar`, `with_azimuthal` and
`with_solid_angle` update an existing member; use a new object to change the
specified pair. `Detector.observation_angle` is a fresh degree quantity and
`with_observation_angle` preserves the acceptance and read-time response.

Both APIs reject the numeric and semantic spelling together, including
positional numeric arguments. `dataclasses.replace` continues to use the
numeric `*_deg`/`*_sr` fields. Numeric scoring, angular integration,
`resolve()` output, stored records, and source/case identities are unchanged.

## Sensor dimensions with units

Use scalar length quantities for the source-facing rectangle:

```python
from pyrite.detectors import Detector, SensorAcceptance
from pyrite.units import Q_

sensor = SensorAcceptance(distance=Q_(40, "cm"), width=Q_(14, "mm"), height=Q_(7, "mm"))
sensor.distance.to("cm")  # fresh scalar quantity; canonical read-back is mm
wider = sensor.with_width(Q_(28, "mm"))
detector = Detector(135, acceptance=wider)
```

`distance` is radial distance; `width` and `height` are full local azimuthal
and polar dimensions. Each semantic input requires a positive finite scalar
Pint length and conflicts with its corresponding `*_mm` spelling.
`with_distance`, `with_width` and `with_height` preserve the other dimensions.
Legacy positional and `*_mm` arguments remain numeric-compatible;
`dataclasses.replace` uses those numeric fields. The exact numeric
`resolve()` tuple retains degree spans and steradians, and detector lowering
and saved records retain their existing numeric representation.

## Spectrometer entrance slit

Model a grating-spectrometer entrance slit with a scalar detector by giving
`SlitAcceptance` the source-to-slit distance, the narrow `width`, the long
`length`, and a `roll` measured from the length axis to the local azimuthal
tangent. With `roll=0` the narrow width lies along the polar direction, so
dispersion runs across the polar (energy) gradient:

```python
from pyrite.detectors import Detector, SlitAcceptance
from pyrite.units import Q_

slit = SlitAcceptance(
    distance=Q_(500, "mm"), width=Q_(50, "um"), length=Q_(10, "mm"), roll=Q_(0, "degree")
)
detector = Detector(observation_angle=Q_(135, "degree"), acceptance=slit)
narrower = detector.acceptance.with_width(Q_(25, "um"))  # sweep slit width, distance or roll
```

`resolve()` returns the polar span, azimuthal span and exact solid angle, which
lower onto the same scalar case fields as `SensorAcceptance`; `roll=0` is
identical to `SensorAcceptance(distance, height=width, width=length)`. The
derivation is in [Detector solid-angle integration](../../physics/detectors/detector-solid-angle.md#slit-acceptance).

This is the idealized fast path for sweeps and optimization. It models a point
source on the slit normal and omits line curvature along the slit, source
size, diffraction and grating coupling. Scoring multiplies the central
direction by $\Omega$, and the Gaussian polar-span broadening treats the polar
span as uniform, which is exact only at $\psi=0^\circ$ and $90^\circ$ (the true
distribution is trapezoidal in between).

## Response contracts

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `detector_efficiency`, `convolve_detector` | Detectors, advanced-stable | Photon grid (eV), window thicknesses (nm) to dimensionless efficiency; Gaussian FWHM (eV) convolution preserves spectral-density units. Optional outside mass has density × eV units. Reads attenuation tables when efficiency is evaluated; invalid grid/array inputs propagate numerical errors. Neither function applies an exposure. |
