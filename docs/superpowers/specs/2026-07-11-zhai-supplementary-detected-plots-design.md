# Zhai supplementary detected plots

## Goal

Make the Zhai supplementary validation figures display complete, detected
coherent spectra in the same observable units as the Zhai Fig. 1c validation.

## Scope

The changes are confined to `checks/anchor_figures.py`, its fast figure tests,
and the explanatory copy in `notebooks/validation_app.py`.  Transport,
intrinsic PXR+CBS calculation, and the detector-response implementation remain
unchanged.

## Rendering and data flow

`model_coherent_spectra` continues to calculate and cache intrinsic coherent
spectral densities per electron per steradian.  The supplementary figure
builders convert each returned spectrum for display by:

1. locating its intrinsic peak;
2. computing the same Gaussian FWHM used by Fig. 1c: the quadrature sum of the
   Zhai EDS resolution and 16.6-degree aperture broadening at 200 keV;
3. applying `convolve_detector` to obtain the detected coherent spectrum; and
4. multiplying by the Fig. 1c collection solid angle (0.066 sr) and the
   1-nA electron rate (6.2415e9 electrons/s).

The displayed y-axis is `Intensity (Phs/eV/s/nA)`.  This reuses the existing
detector physics and changes only which already-supported observable is shown.

## Layout and bounds

The TMD four-panel canvas will be reduced to fit the validation app's output
region while preserving the 2-by-2 comparison.  Every supplementary axis will
have a lower y limit of zero and explicit x limits equal to the configured
study window, including the nominal upper endpoint even though the 1-eV input
grid excludes that endpoint.

## Tests and verification

Fast synthetic-spectrum tests will assert the detected, scaled plotted values,
the zero lower y bound, the explicit study x limits, the unit label, and the
TMD figure width.  The focused test module and its lint check will run after
the change; a rendered synthetic TMD figure will be visually inspected for the
right-column clipping regression.
