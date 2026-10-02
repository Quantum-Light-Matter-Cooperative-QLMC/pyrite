# Pair-conversion spectra against Geant4 (issue #275)

An opt-in comparison of `pair_production.sample_pair` with the electron and
positron produced by photon conversion in Geant4 11.4.2 TestEm5. It is the
independent comparison for the `pair-production-sampling` ledger claim and
does not sign it off.

## Reference setup

The Geant4 installation and TestEm5 source are those of the issue #182
full-track check (`../full_track_bremslib/GEANT4_BUILD.md`): Geant4 tag
`v11.4.2`, commit `8cc04f65977807f1848da7b958c421cd5e162f26`, from
conda-forge. Copy `examples/extended/electromagnetic/TestEm5` from that
source and run `python3 patch_stepping.py TestEm5/src/SteppingAction.cc`.
The patch writes one line per conversion of the primary photon to `$PAIR_OUT`:
event id, photon kinetic energy (eV) and direction before the step, then the
PDG code, kinetic energy (eV) and direction of every secondary created in that
step. The copy also carries the #182 histogram patch, which does not affect
these records. Configure with `-DCMAKE_PREFIX_PATH=<geant4 prefix>` so CMake
finds the bundled CLHEP.

`make_macros.py` writes one macro per case and physics list. Each macro shoots
a mono-energetic photon pencil beam at a thick slab with a 1 m lateral size,
fluorescence off, and secondaries killed at creation
(`/testem/stack/killSecondaries 2`). It runs single-threaded with fixed seeds.
`run_pairs.sbatch` runs the twelve macros as a SLURM array on the lab box's
CPU partition; the full set takes a few minutes.

| Case | Material | Thickness | Photon energy | Events |
|---|---|---:|---:|---:|
| `c_2mev` | Graphite (Z = 6) | 30 cm | 2 MeV | 6,000,000 |
| `c_5mev` | Graphite | 30 cm | 5 MeV | 1,000,000 |
| `pb_2mev` | Lead (Z = 82) | 3 cm | 2 MeV | 1,000,000 |
| `pb_5mev` | Lead | 3 cm | 5 MeV | 500,000 |

Physics lists:

- **`empenelope`:** `G4PenelopeGammaConversionModel`, the same PENELOPE
  §2.4 model as PyRITE, so this is a transcription check.
- **`emstandard_opt0`:** `G4BetheHeitlerModel` with modified-Tsai angles,
  an independent model.
- **`emstandard_opt4`:** `G4BetheHeitler5D`, an independent model that
  produces explicit triplets.

## Reduction and comparison

`compare.py reduce RAW_DIR` keeps only conversions at the full beam energy,
so the photon has lost no energy before converting. It takes angles about
the photon's own pre-step direction, counts three-secondary (triplet) events,
and bins the electron energy share `x = E_- / (k - 2 m c^2)` and both polar
cosines into `reference/*.json`. The raw records are not committed; the
binned references are.

`compare.py` samples `sample_pair` at the same photon energy and Z (200,000
pairs per case, fixed seeds). It then reports the two-sample chi-square, the
Kolmogorov distance and the mean of each observable. `empenelope` must agree
within statistics, or the script exits 1. The two independent models are
reported, not gated.

```bash
PYRITE_MC_BACKEND=cpu uv run python checks/pair_production_geant4/compare.py
```

## Results (2026-10-01, PyRITE `8ca14d0c`, SLURM jobs 332/344)

Pairs at the full beam energy per case: 37k–39k for C at 2 MeV, 51k–60k for C at 5 MeV, 91k–95k for Pb at 2 MeV, and 193k–196k for Pb at 5 MeV. Each entry gives chi-square/dof and the Kolmogorov distance $D$ of PyRITE against Geant4.

| Case | List | Energy share $x$ | $\cos\theta_-$ | $\cos\theta_+$ |
|---|---|---|---|---|
| C, 2 MeV | `empenelope` | 0.95, $D$ 0.003 | 1.08, 0.006 | 1.20, 0.003 |
| C, 5 MeV | `empenelope` | 0.81, 0.002 | 0.86, 0.002 | 0.96, 0.005 |
| Pb, 2 MeV | `empenelope` | 0.83, 0.006 | 0.95, 0.006 | 0.92, 0.002 |
| Pb, 5 MeV | `empenelope` | 1.23, 0.002 | 0.88, 0.001 | 1.21, 0.002 |
| C, 5 MeV | `opt0` | 1.95, 0.005 | 3.69, 0.029 | 4.12, 0.032 |
| Pb, 5 MeV | `opt0` | 2.38, 0.004 | 8.55, 0.033 | 7.57, 0.032 |
| C, 5 MeV | `opt4` | 1.12, 0.005 | 4.08, 0.014 | 4.02, 0.021 |
| Pb, 5 MeV | `opt4` | 1.26, 0.002 | 9.05, 0.019 | 9.67, 0.021 |
| C, 2 MeV | `opt4` | 6.82, 0.016 | 9.64, 0.052 | 9.11, 0.050 |
| Pb, 2 MeV | `opt4` | 19.95, 0.015 | 20.65, 0.047 | 21.02, 0.047 |

- **`empenelope`:** agrees within statistics in every case and observable (p = 0.10–0.81), so the PyRITE transcription matches Geant4's implementation of the same model.
- **Energy sharing, independent models:** at 5 MeV it agrees with both to $D\le0.005$. At 2 MeV the 5D model differs by $D\approx0.016$. `opt0` at 2 MeV differs strongly ($D\approx0.1$): its energy share is narrower and has no support near the endpoints, whereas PENELOPE's $F_0$-corrected DCS stays finite there. The two near-threshold Bethe–Heitler forms disagree, and this check does not decide which is closer to exact theory.
- **Angles, independent models:** the mean polar cosine differs from them by about 0.01 ($D$ = 0.014–0.052). This is the expected cost of the leading-term angular law (PENELOPE-2024 §2.4.1.1), which multiple scattering of the daughters washes out.
- **Triplets:** the 5D model gives a triplet share of 14 % (C) and 1.2 % (Pb) at 5 MeV, about three times the EPDL2025 $\sigma_{\rm pair,el}/\sigma_{\rm pair}$ (4.7 %, 0.4 %) that PyRITE folds into the pair channel. The 5D model has none at 2 MeV, below its 4 m c² threshold.
