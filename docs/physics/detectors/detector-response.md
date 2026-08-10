# Detector and instrument response

Detector response is a downstream transformation of source spectra. PyRITE
keeps source generation, geometric acceptance, absorption/efficiency, energy
redistribution, and presentation distinct so that instrument effects can be
enabled and audited independently.

## Response stages

1. The source kernels produce differential photon yield per incident electron,
   photon energy, and steradian.
2. Dataset scaling records detector solid angle and source cadence/charge
   without changing cached source arrays.
3. Quantum efficiency or optical throughput weights photons by true energy.
4. Energy redistribution maps true photon energy into measured energy through
   broadening and, for pixel detectors, charge diffusion/sharing.
5. Plotting and tabulation select units and normalization.

Applying solid angle twice is an error. The stored result scale already carries
the configured acceptance; see [Detector solid angle](detector-solid-angle.md).

## Supported response families

The Timepix response uses silicon absorption and a cached response matrix with
thresholding, statistical broadening, and charge-sharing behavior. The Eagle
XO path shares silicon-sensor physics but normally models a CCD as solid angle
times measured quantum efficiency; an optional low-occupancy mode adds
Fano/read-noise energy resolution. The grazing-grating/ALEXS path combines
optical mapping with silicon QE, diffusion, and resolution models.

Profile detector fields currently provide geometry plus portable response
metadata. The metadata joins dataset identity, but remains inert until a named
adapter explicitly consumes it. Detector-model modules also contain
experiment-specific placeholders; absolute predictions require those values to
be replaced by calibrated hardware geometry and operating conditions.

A response matrix $R(E_m\mid E)$ acts on a true-energy spectrum as

```{math}
N_m(E_m)=\int R(E_m\mid E)\,\eta(E)\,N(E)\,dE,
```

where $\eta(E)$ is efficiency/throughput. Matrix columns must conserve the
accepted probability appropriate to the selected detector model; any loss
belongs in efficiency, not an unexplained normalization.

## Assumptions and interpretation

- response parameters describe an idealized configured instrument, not an
  automatic calibration of a particular physical detector;
- convolution cannot create source photons and should be inspected after the
  unconvolved components;
- measured-energy binning and interpolation can affect narrow features;
- external background subtraction is analysis-only and is not detector
  response or source bremsstrahlung.

Validation claims include `detector-solid-angle`, `detector-line-broadening`,
`alexs-qe-absorption`, `alexs-charge-diffusion`, `detector-eaglexo`, and
`grazing-reflectivity`; consult the [validation
ledger](../../validation/physics-validation-ledger.md) for current status.
Implementation owners are `pyrite.montecarlo.detector` and
`pyrite.detectors.*_response`.
