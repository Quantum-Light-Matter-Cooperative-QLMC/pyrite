# External bremsstrahlung comparison and subtraction

PyRITE compares its Born+Elwert background with external, already
detector-normalized spectra ingested by `load_external_brem`. External identity
must name dataset, version, condition, source file, and checksum. “NIST
DTSA-II” names simulation software, not one canonical NIST dataset or spectrum.

## Versioned Zhai fixture

`src/pyrite/validation/reference_data/external_brem/v1/` preserves Zhai et al. Figure 3b's
25 keV, 1 mm HOPG, 0.066 sr condition. Numeric strings were copied without
modification from authors' deposited DR-NTU dataset:

- V1, DOI [`10.21979/N9/WZAMZ0`](https://doi.org/10.21979/N9/WZAMZ0), UNF
  `UNF:6:nJhYFFasbKoHid6aS6g/PA==`, CC BY-NC 4.0;
- bremsstrahlung: file 287301, `Figure 3b Bremsstrahlung.tab`, MD5
  `930509a7e99c67a61936f94b7ad0510b`;
- experiment: file 287302, `Figure 3b Experiment.tab`, MD5
  `68c414f7c14a8d323abdd994d6340152`.

Units are detected intensity in `Phs/eV/s/nA`. Do not reapply solid angle,
detector efficiency, resolution, channel width, live time, or beam current.
`load_external_brem` performs interpolation only and returns zero outside
fixture support.

Zhai [SI S3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-66063-6/MediaObjects/41467_2025_66063_MOESM1_ESM.pdf)
says experimental bremsstrahlung was subtracted with DTSA-II through a
numerical method citing Clayton, Duerden, and Cohen's PIXAN work. SI omits fit
details needed for exact reconstruction. PyRITE's method below is independently
specified, not claimed as exact Zhai processing.

NIST describes DTSA-II simulations as absolute spectra parameterized by dose
and detector properties; its 2022 study reported continuum agreement within
about 10% over 1–10 keV for tested elements. This supports use as comparator,
not universal normalization. See [Newbury and Ritchie
2022](https://www.nist.gov/publications/simulating-electron-excited-energy-dispersive-x-ray-spectra-nist-dtsa-ii-open-source).

## Fit, subtraction, and comparison

`fit_external_background` fits non-negative amplitude `a` for external shape
`b_i` against measured sidebands `y_i`:

```text
chi2(a) = sum_i ((y_i - a b_i) / sigma_i)^2
a       = sum_i(w_i b_i y_i) / sum_i(w_i b_i^2),  w_i = 1/sigma_i^2
```

Figure 3b sidebands exclude 900–1040 eV, containing broadened coherent peak.
`subtract_external_background` returns `y - a b`, fitted background, and fit
diagnostics. Fit has no offset, slope, energy shift, or added broadening.
Standard error assumes independent supplied uncertainties and exact external
shape.

`compare_external_background` compares PyRITE and external detected spectra
without rescaling: integrated-intensity ratio, mean-normalized RMSE, and
Pearson shape correlation. Validation app tab **External brem + subtraction**
shows source-backed fit, subtraction, and matching 25 keV PyRITE comparison.

Limitations: deposited table lacks DTSA-II version, detector definition, seed,
dose inputs, and raw pre-fit EDS channels. Only this 25 keV, 1 mm condition is
fixture-backed. Low-electron PyRITE preview runs retain Monte Carlo noise.
