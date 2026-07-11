# Validation status: Zhai supplementary coherent-emission studies

**Scope.** `checks/anchor_figures.py::ZHAI_SUPPLEMENTARY_STUDIES` - the
WSe2/MoSe2/h-BN reproductions rendered by the validation app's "Zhai
supplementary" section and `cxr check --export`.

This is a provenance record, not a physics re-derivation. The inputs
parameterize existing ledgered claims (`line-energy-dispersion`,
`coherent-line-spectrum`, and `electron-transport`); their ledger states are
unchanged, and only a human marks a claim `signed-off`.

## Supplementary Fig. 5 inputs

| material | thicknesses (nm) | beam energy (keV) | polar tilt theta_til (deg) | azimuth phi_til (deg) |
|----------|------------------|-------------------|----------------------------|------------------------|
| WSe2 | 42, 55, 75 | 200 | 10, 15, 17.5, 20 | not reported |
| MoSe2 | 47, 112, 147 | 200 | 10, 15, 17.5, 20 | not reported |
| h-BN | 921 | 17.5, 20, 22.5, 25 | 17 | 130 |

The TMD values come from Section S7 and Supplementary Fig. 5: the beam was
aligned to the `[001]` zone axis using Kikuchi lines and then tilted with a
double-tilt TEM holder. The paper reports the four polar tilts but does not
report the corresponding holder azimuth. Consequently, the study metadata
stores `azimuth_deg=None`. Modeling requires an explicitly supplied
`exploratory_azimuth_deg`, which is labeled unreported in the app and included
in the cache key.

The 921 nm h-BN values come from Supplementary Table 4. Its Fig. 5 panel is an
SEM beam-energy series at one fitted sample orientation; it is not a four-tilt,
200 keV series.

## Supplementary Table 4 orientation transcription

The paper explains that surface roughness, holder unevenness, and bending make
the SEM sample angles nonzero. It obtains each pair by minimizing the residual
sum of squares between measured and predicted spectra.

| sample | thickness | theta_til (deg) | phi_til (deg) |
|--------|-----------|-----------------|---------------|
| graphite film | 29 +/- 6 nm | 9.5 | 120 |
| graphite film | 76 +/- 5 nm | 13.0 | 120 |
| graphite film | 150 +/- 13 nm | 6.5 | 180 |
| bulk graphite (HOPG) | ~17 um | 11.0 | 60 |
| bulk graphite (HOPG) | ~500 um | 11.5 | 40 |
| bulk graphite (HOPG) | ~1 mm | 9.5 | 40 |
| h-BN film | 42 +/- 3 nm | 13.5 | 115 |
| h-BN film | 219 +/- 17 nm | 11.5 | 65 |
| h-BN film | 109 +/- 5 nm | 13.5 | 130 |
| h-BN film | 219 +/- 17 nm | 17.0 | 130 |
| h-BN film | ~659 nm | 14.5 | 105 |
| h-BN film | ~921 nm | 17.0 | 130 |
| bulk h-BN | ~170 um | 20.0 | 65 |

## Correction to the former azimuth assumption

The previous implementation called `tilted_geometry` without its azimuth
argument, implicitly evaluating every supplementary spectrum at `phi=0`. The
former provenance note then described zero as the SI's established reference
plane. That conclusion was unsupported: Table 4 explicitly reports nonzero
azimuths for graphite and h-BN, while Section S7 leaves the TMD azimuth
unreported.

The corrected implementation passes the reported 130 degree azimuth for the
921 nm h-BN study. For MoSe2 and WSe2 it retains `None` in published metadata
and accepts only a separate, explicit exploratory value. Cache schema 2
prevents reuse of spectra generated under the old implicit-zero assumption.

## Status

Provenance: transcribed from Zhai et al. Supplementary Table 4, Section S7, and
Supplementary Fig. 5. Reported h-BN orientation: encoded. TMD azimuth:
unreported and exposed only as an exploratory input.

**Not signed off.** Regenerated spectra and intensity comparisons still require
fresh-context verification against the published curves before any validation
claim advances.
