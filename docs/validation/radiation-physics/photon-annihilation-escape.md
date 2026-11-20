# 511 keV photon escape against Geant4

Claim: `photon-annihilation-escape`. Status: **anchored** (2026-11-15), after
fresh-context adjudication; PyRITE arm re-pinned 2026-11-20 (#1231).

PyRITE's photon-only transport passes all 14 pre-registered gates. That holds
against the Geant4 `empenelope` reference and against the report-only
`emlivermore` arm. Narrow-beam 511 keV escape matches an independent
`exp(−μd)` calculation within 0.1%. Band-integrated Compton build-up agrees
within 5%. Since #1231 PyRITE's Compton sampler includes impulse-approximation
Doppler broadening and relaxes the Compton shell vacancy; the earlier W
backscatter deficit (see below) is gone within counting statistics. The W
backscatter spectral shape is still compared only in ungated bands.

## Reference and scope

`checks/photon_full_track/annihilation_escape.py` tests only the photon chain
that carries annihilation photons out of a target. Positron transport and
annihilation sampling have their own claims (`heitler-annihilation`,
`positron-annihilation-at-rest`). Photons of 510.99895 keV travel along +z in
the thick-target slabs (W 4 mm, Si 25 mm; 5 cm footprint), from one of two
sources:

- `ext`: 1 nm before the entrance face;
- `int`: at mid-depth, standing in for an annihilation point.

The two 10 cm apertures sit 10 cm in front of the slab (R) and behind it (T).
Both engines kill photon-born charged particles. Geant4 uses complete vacancy
cascades. Each engine ran 10⁶ histories per configuration in ten independent
shards of 100 history blocks. The check README owns the settings and commands.

## Pre-registered gates

Gates were fixed in code before any spectrum from either engine was
inspected. An observable is gated when the reference holds at least 1000
counts. It passes when `|P − G| ≤ max(tol · G, 3σ)` per incident photon:

| Observable | Tolerance |
|---|---|
| Uncollided counts, 504–520 keV | 2% |
| Redistributed counts, 10–100, 100–300, 300–504 keV | 5% |
| W only: raw W Kα window, 57–60 keV | 10% |

## Results

Gated observables (Geant4 jobs 13030/13031; PyRITE rerun job 14196 at
commit `8fd3b1a16`; PyRITE / Geant4):

| Config | Aperture | Observable | vs Penelope | vs Livermore |
|---|---|---|---|---|
| W ext | T | direct 511 | 0.9964 | 1.0001 |
| W ext | T | redist 300–504 | 1.002 | 0.997 |
| W ext | R | redist 10–100 | 1.025 | 1.000 |
| W ext | R | redist 100–300 | 0.986 | 1.038 |
| W int | T | direct 511 | 0.9967 (−2.9σ) | 0.9986 |
| W int | T | redist 300–504 | 0.996 | 0.994 |
| Si ext | T | direct 511 | 1.0005 | 1.0000 |
| Si ext | T | redist 100–300 | 0.993 | 0.989 |
| Si ext | T | redist 300–504 | 1.001 | 0.998 |
| Si ext | R | redist 100–300 | 1.002 | 1.003 |
| Si int | T | direct 511 | 1.0005 | 1.0003 |
| Si int | T | redist 100–300 | 1.022 | 1.013 |
| Si int | T | redist 300–504 | 0.996 | 1.003 |
| Si int | R | redist 100–300 | 0.975 | 0.988 |

All 14 pass within tolerance against both arms. Before #1231, W ext R
100–300 keV (0.950 against Penelope) passed only via its 3σ limit. Direct
511 keV yields are unchanged by the Compton model. The ungated W Kα windows
(450–900 counts) agree within 10% (W ext T 1.093 against Penelope, 1.4σ);
they rose 1–5% because Compton K/L vacancies now fluoresce.

## Ungated W backscatter

W cells on R with fewer than 1000 reference counts are not gated. Without
Doppler broadening (records up to #1231) PyRITE was low in each; the #1231
rerun (job 14196) brings each within counting statistics:

| Source | Band | Free-electron PyRITE / Penelope | #1231 PyRITE / Penelope | #1231 vs Livermore |
|---|---|---|---|---|
| internal | 100–300 keV | 41 / 80 (0.51) | 81 / 80 (1.01) | 81 / 61 |
| internal | 300–504 keV | 3 / 11 | 8 / 11 | 8 / 9 |
| external | 300–504 keV | 6 / 30 (0.20) | 25 / 30 (0.83) | 25 / 30 |

The adjudication below explains the original deficit. Single Compton backscatter into R gives 170–181 keV. PyRITE's peak is that
narrow, while both Geant4 arms spread it over about 150–220 keV, with a tail
beyond 300 keV. Near backscatter, bound-electron momentum shifts the scattered
energy by about 1.7 keV per atomic unit. W's M, L and K electrons therefore
broaden the peak by tens of keV, and up to more than 100 keV for the K shell.
PyRITE's incoherent-scattering-function Compton keeps the binding suppression
but omits this broadening.

The adjudicator modelled a single backscatter with the xraylib W Compton
profile and escape back through the slab. Broadening raises the escaping
yield by 1.70× for the internal source but only 1.015× for the external one.
That predicts an internal 100–300 keV ratio of about 0.59 (observed 0.51
against Penelope, 0.67 against Livermore). It also explains why the gated
external cell passes. The predicted 300–504/100–300 keV count ratios, 0.0065
(external) and 0.098 (internal), compare with 0.011 and 0.14 for Geant4 and
0.002 and 0.07 for PyRITE. The external 300–504 keV deficit is therefore the
high-momentum K/L-shell tail of the same broadening, enhanced by easier
escape, not a separate multiple-scattering effect. The pure Rayleigh
backscatter at 510 keV agrees (PyRITE 19, Penelope 17, Livermore 24).

With broadening, PyRITE's 300–504/100–300 keV count ratios are 0.0091
(external, 25/2762) and 0.099 (internal, 8/81), against Geant4 Penelope 0.011
and 0.14. Doppler broadening alone (superseded commit `3c52ed602`, records not
committed) already gave the same W ratios, so the change is the broadening,
not the vacancy hand-off. The internal-source R W Kα window holds 3 PyRITE
counts against 8 (Penelope) and 2 (Livermore); it is a counting-statistics
window, not a resolved difference.

One further ungated cell, Si ext T 10–100 keV, differs mainly between the
Geant4 arms: Livermore 638, Penelope 508, PyRITE 541 (510 before #1231).

**Limitation (narrowed by #1231).** The W backscatter spectrum on R is
compared only through ungated band integrals of 8–80 reference counts; its
window-resolved shape is not gated. The ≈2× deep-source deficit is removed.

## Reproducibility

`checks/photon_full_track/escape-results/` holds merged history blocks for
all twelve engine/configuration records, both reports and `provenance.json`.
Regressions recompute both reports, require all 14 gates in each arm, and pin
the ungated backscatter ratios and record hashes. The PyRITE records were
re-pinned once, by the #1231 rerun (job 14196, tested commit `8fd3b1a16`);
`provenance.json` keeps the superseded record hashes under
`pyrite_rerun_1231`.

## Fresh-context adjudication (2026-11-15)

Verdict: `rederived`; recommended status `anchored`.

- **Reports.** An independent reducer reproduced every mean, SE, limit, gated
  flag and decision of both reports (largest relative difference 9e-16). All
  12 record hashes and the tested source hashes match.
- **Direct yields.** With xraylib μ/ρ including Rayleigh scattering (W
  0.13381, Si 0.08667 cm²/g) and acceptance 1, PyRITE matches `exp(−μd)`
  within 0.1% in every configuration. Livermore matches within 0.13%.
  Penelope is 0.33–0.40% high in W, so PyRITE's small W direct residual
  against Penelope is on the reference side.
- **Harness.** Edges, direct/redistributed classification, frame mapping and
  shard independence are sound. `merge` now rejects duplicate shards.
- **Wording.** The claim was narrowed to band integrals: no W Kα cell reaches
  the gating threshold, and W backscatter shape differs while its band
  integral passes. The backscatter attribution was corrected as above.
- **Anchor gap.** The escape anchors replay committed records. The live
  narrow-beam attenuation test in `test_photon_histories.py` covers uncollided
  transport. A slow live W test against `exp(−μd)` would close the gap
  further.

