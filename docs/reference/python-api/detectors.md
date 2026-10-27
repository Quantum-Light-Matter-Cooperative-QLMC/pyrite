# Detectors and response

Separate angular acceptance, physical detector geometry, and instrument response. `AngularDetector`, `Timepix3Response` and `EagleXOResponse` are supported semantic aliases for existing classes, not new physics.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.detectors.PlanarPose
   pyrite.detectors.PixelGrid
   pyrite.detectors.FilterPlate
   pyrite.detectors.IdealPhotonCounter
   pyrite.detectors.PlanarDetector
   pyrite.detectors.response.detector_efficiency
   pyrite.detectors.response.convolve_detector
   pyrite.detectors.Detector
   pyrite.detectors.AngularAcceptance
   pyrite.detectors.SensorAcceptance
   pyrite.detectors.EnergyBins
   pyrite.detectors.NativeSpectrum
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
   pyrite.detectors.NativeBinnedResponse
```

## Detector response

Use these objects for detector efficiency, energy redistribution and grating
images. Physical pixel scoring and acquisition are illustrated in the
[workflow guide](../../reference/python-scene-recipes.md#place-a-finite-filter-over-detector-pixels).

## Response contracts

| Entry | Owner / status | Scientific contract and effects |
| --- | --- | --- |
| `detector_efficiency`, `convolve_detector` | Detectors, advanced-stable | Photon grid (eV), window thicknesses (nm) to dimensionless efficiency; Gaussian FWHM (eV) convolution preserves spectral-density units. Optional outside mass has density × eV units. Reads attenuation tables when efficiency is evaluated; invalid grid/array inputs propagate numerical errors. Neither function applies an exposure. |
