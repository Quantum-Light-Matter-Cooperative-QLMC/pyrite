# bugfix/zhai-detector-response

The h-BN 921 nm supplementary comparison (`checks/anchor_figures.py`, Zhai 2025 SI Fig. S5b)
was reading ~1.9-2.2x above the paper's peaks. Two detector-model defects were found and
fixed: `montecarlo/detector.py::aperture_fwhm_eV` had a misplaced parenthesis giving a
sqrt(3)x-too-wide aperture line-broadening term instead of Zhai SI Eq. (14)'s prefactor, and
`checks/anchor_figures.py::_supplementary_detected_spectrum` omitted window efficiency
(`detector_efficiency`), now applied before convolution. After both fixes a residual,
unexplained 1.35-1.76x gap remains, growing with beam energy, tracked as
`zhai-hbn-921-detected` (discrepancy) in `docs/physics-validation-ledger.md`.

## Remaining work

- Re-derive the `montecarlo/spectrum.py::mc_spectrum` intrinsic emission prefactor against
  Zhai SI Eq. (2) in fresh context.
- Decide/confirm whether Zhai's experimental SI spectra are already window-efficiency
  (QE) corrected — SI S3/S4 is silent, so our QE application is currently a modeling choice.
- Consider tabulated Mott partial cross sections (e.g. CASINO) for the angular-tail shape
  instead of the first-moment-calibrated screened-Rutherford law; transport elastic-model
  choice (mott vs sr) was ruled out as the source of the 2x gap (<10% effect) but the tail
  shape specifically has not been isolated.
- Regenerate supplementary caches/checkpoints — all local+remote checkpoints were
  deliberately cleared 2026-07-11 after the tilt-convention flip and still need regen.
