# Bulk Cherenkov yield against Geant4 (issue #587)

An opt-in independent reference for the `frank-tamm-homogeneous-dielectric`
ledger claim: Geant4 11.4.2 `G4Cerenkov` photons from a single charged
primary crossing a slab with a tabulated `RINDEX`. It does not sign the claim
off.

## Reference setup

`cherenkov.cc` is a minimal Geant4 application (no example patching):
a slab of `G4_WATER` or `G4_SILICON_DIOXIDE`, 1 m transverse, in
`G4_Galactic`; `G4EmStandardPhysics_option4` plus `G4OpticalPhysics` with
every optical process except Cherenkov switched off; at most 300 photons and
1 % beta change per step; all secondaries killed at creation, so only the
primary radiates. For each primary step inside the slab it writes the
step-mean beta `(beta_pre + beta_post)/2` (the beta `G4Cerenkov` uses), the
step length and the photon count, and histograms the photon energy and the
polar angle about the step chord (the axis `G4Cerenkov` rotates onto).

Build against the issue #182 Geant4 environment
(`../full_track_bremslib/GEANT4_BUILD.md`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGeant4_DIR=<geant4-env>/lib/cmake/Geant4 -DCMAKE_PREFIX_PATH=<geant4-env> \
  -DCMAKE_CXX_COMPILER=<geant4-env>/bin/x86_64-conda-linux-gnu-g++
cmake --build build
python compare.py cases cases/   # RINDEX tables + cases.txt
```

`run_cherenkov.sbatch` runs the four cases as a SLURM array on the lab box's
CPU partition (under a minute each).

| Case | Radiator index | Particle | beta | Slab | Events |
|---|---|---|---:|---:|---:|
| `water_mu_b099` | constant 1.33 | mu- | 0.99 | 10 mm | 20,000 |
| `silica_mu_threshold` | fused silica, Malitson (1965) | mu- | 1/1.47 | 0.5 mm | 300,000 |
| `n150_alpha_b095` | constant 1.50 | alpha (z = 2) | 0.95 | 5 mm | 10,000 |
| `water_p_below` | constant 1.33 | proton | 0.74 | 10 mm | 2,000 |

The photon band is 1.5–5.9 eV in 88 bins; angles are 0–90° in 0.1° bins.
In the silica case the threshold `n(E) = 1/beta` lies inside the band, so it
exercises the dispersive threshold; the proton case is below threshold
everywhere.

## Reduction and comparison

`compare.py reduce RAW_DIR` writes `reference/<case>.json`: the histograms,
the RINDEX table, and the primary path length grouped by step-mean beta in
1e-6 bins. The raw step files are not committed.
`tests/montecarlo/spectrum/test_cherenkov_geant4_reference.py` sums `frank_tamm_yield`
over the recorded steps with the same linearly interpolated index and
compares totals (Poisson), the energy spectrum and the angle histogram
(reduced chi-square over bins with at least 10 expected photons).

Results (job 2280, 2026-10-23):

| Case | Geant4 photons | Ratio to PyRITE | z | Energy chi2/ndf | Angle chi2/ndf |
|---|---:|---:|---:|---:|---:|
| `water_mu_b099` | 13,773,785 | 1.00018 | 0.68 | 0.74 | 0.46 |
| `silica_mu_threshold` | 541,847 | 1.00146 | 1.08 | 1.09 | 0.91 |
| `n150_alpha_b095` | 16,511,681 | 0.99971 | −1.18 | 1.15 | 1.38 |
| `water_p_below` | 0 | — (0 expected) | — | — | — |

`G4Cerenkov` normalizes with `369.81 / (eV cm)` and the trapezoidal integral of
`1/n^2` over the table nodes; both differ from PyRITE's constant and exact
integral of the interpolated index by well under 1e-5, below the statistical
precision. Constant-index cases put every photon in one 0.1° angle bin, so
their angle check is a cone-position check only.
