# Thick-target photon full-track check (issue #341)

An opt-in comparison of PyRITE's experimental CPU photon transport with
Geant4 11.4.2 TestEm5 (`empenelope`) for electrons stopping in thick W and Si.
It is the evidence for the `photon-full-track-thick-brem` ledger claim and
does not sign it off. Heavy runs go to the lab box. The repository keeps the
scripts, compact Geant4 references (`reference/*.json.gz`) and compact
PyRITE results (`results/*.json.gz`).

## Cases

Electron pencil beams at normal incidence on a 5 cm × 5 cm slab at least as
thick as the CSDA range (`cases.py`):

| Target | Energies (MeV) | Thickness (mm) |
|---|---|---|
| Si, 2.33 g/cm³ | 0.3, 1, 3, 10 | 0.5, 2.5, 8, 25 |
| W, 19.3 g/cm³ | 0.3, 1, 3, 10 | 0.1, 0.5, 1.5, 4 |

Photons are scored on two ideal 10 cm × 10 cm square apertures in vacuum,
10 cm from the slab faces: `T` behind the exit face facing +z, and `R` in
front of the entrance face facing −z. Spectra are counts per incident
electron on 0.5 keV bins from 1 to 100 keV, then 60 logarithmic bins to E₀.
Each count is split into direct (no interaction) and redistributed photons.
Redistributed photons include Compton/Rayleigh-scattered photons and
fluorescence after photoabsorption.

## Physics scope

Both sides transport only the primary electron, as in #439. They use no loss
fluctuations, a 10 keV electron stop and no delta rays. Bremsstrahlung
photons above about 1 keV are transported with photoelectric absorption by
subshell plus fluorescence, Compton, Rayleigh and pair conversion. Every
photon-born charged particle is dropped without transport (PyRITE's
photon-only mode). Auger emission and charged-particle impact ionization
fluorescence are off.

- **Geant4.** `make_macros.py`: `empenelope`, Goudsmit–Saunderson,
  `/process/eLoss/fluct false`, `fluo true`, `auger false`, `pixe false`, 10 keV
  `lowestElectronEnergy`, 1 nm gamma and 1 km e± range cuts.
  `patch_testem5.py` adds `killSecondaries 3` (kill charged secondaries at
  creation and keep photons). It makes the upstream `G4bool` switch a
  `G4int`, and prints one `GAM` line per photon leaving the vacuum world.
  Apply it on top of `../full_track_bremslib/geant4-testem5-emission.patch`
  and `../mev_full_track/patch_tracking.py`. Each case runs `SHARDS[case]`
  macros of 10⁶ events with distinct seeds, sized for ≥ 2×10⁴ hits on the
  weaker aperture from 10⁵-event pilot rates.
- **PyRITE.** `pyrite_bench.py`: the #439 electron stack (ELSEPA, SBETHE,
  coupled BremsLib at `kc` = 1 keV, Kawrakow deflection, midpoint energy, no
  straggling, per-electron CPU core). Each 50-primary batch then drives the
  weighted continuous BremsLib source (`score_bremslib_continuum`). It uses
  400 proposals per primary over the full sphere and 1 keV–E₀, with photon
  and electron cutoffs of 1 keV. It does not attenuate the source and runs one
  ordinary-hit pass per aperture. The v2 source uses a defensive mixture:
  75% log-uniform / 25% uniform energy and 75% forward / 25% isotropic
  direction, with the complete mixture density in the importance weight.
  `run_pyrite.sbatch` runs 20 shards of
  1,000 primaries per case.

Known model differences, which are not separated by this check:

- BremsLib against PenBrem source (1–4 % in photon energy in #439);
- Geant4 Penelope Compton creates shell-vacancy fluorescence (`compt` photons),
  which PyRITE's incoherent-scattering-function Compton does not;
- Geant4 fluorescence data (`fluor`, EADL-based) against PyRITE EADL2025.

## Gates

These gates were fixed before any binned PyRITE result was inspected. The
only earlier PyRITE output was a 50-primary smoke run, of which only the
in/out-of-band detection rates were printed. For every case and aperture, an
observable passes when `|P − G| ≤ max(tol × G, 3σ)`. σ combines both
estimators' standard errors: 100 equal event blocks for Geant4 and
batch-to-batch scatter for PyRITE. Correlated photons from one history are
therefore never treated as independent. A band or line is gated only when
it holds ≥ 1,000 Geant4 counts.

| Observable | Tolerance | Status |
|---|---|---|
| Counts per primary, 1–10 and 10–30 keV | 10 % | reported (see revision) |
| Counts per primary, 30–100 keV, 0.1–0.3, 0.3–1, 1–3, 3–10 MeV (to E₀) | 7 % | reported (see revision) |
| Redistributed fraction in each band | 0.02 absolute | gated |
| Redistributed counts per primary in each band with ≥ 1,000 Geant4 redistributed counts | 10 % | gated (added in revision) |
| W Kα (57–60 keV) and Kβ (66.5–69.5 keV) net line counts, linear side-window continuum | 10 % | gated |

The 7 % continuum tolerance was meant to allow for the 1–4 % BremsLib/PenBrem
source difference and the few-percent electron-transport agreement of
#182/#439. Low-energy bands are dominated by attenuation and fluorescence and
got 10 %.

### Revision after the first results (2026-10-23)

The gates were revised after the Si 0.3/1/3 MeV results had been inspected.
Results are reported against both sets.

Under the original gates, 4 total-count bands failed:

| Case | Aperture | Band | PyRITE/Geant4 |
|---|---|---|---|
| Si 0.3 MeV | T | 30–100 keV | 1.094 |
| Si 1 MeV | T | 100–300 keV | 1.084 |
| Si 1 MeV | T | 0.3–1 MeV | 1.083 |
| Si 3 MeV | R | 0.3–1 MeV | 0.894 |

In every failing band, the uncollided (direct) photons carried the deviation.
The redistributed photons agreed within 1–2σ. The direct component is the
unattenuated source times straight-line attenuation, so these failures test
the bremsstrahlung source (BremsLib partial-wave DDCS against PenBrem's
angular shape) and the electron transport, not the photon tracker.
`photon-full-track-thick-brem` is a photon-transport claim. Totals and direct
counts are therefore reported, not gated. Redistributed counts and fractions
and W K fluorescence lines are gated. The source difference stays a
documented, reported discrepancy, and a separate creation-level
emission-angle comparison is the follow-up.

The same revision fixed a reducer error: Geant4 fluorescence photons that
escaped uncollided were classed as direct, whereas PyRITE's direct means an
uncollided *source* photon. Direct now also requires the `eBrem` creator.

## Running

```bash
# lab box, staging root ~/.local/opt/pyrite-issue-341
python3 make_macros.py                      # writes *.mac and cases.txt
GEANT4_ENV=... MICROMAMBA=... sbatch --export=ALL run_geant4.sbatch
PYRITE_CHECKOUT=$PWD/code PYRITE_HOME=$PWD/home sbatch --export=ALL run_pyrite.sbatch
# reduce
python compare.py reduce-geant4 CASE results/CASE_s*.log.gz
python compare.py reduce-pyrite CASE results/CASE_pyrite_*.npz
python compare.py --output report.json
```

`PYRITE_HOME` is a dedicated data root holding the pinned tables and
datasets, including `epdl-differential`. The staging root holds its own
`git archive` checkout and uv environment, so neither the shared remote
workspace nor its table store is touched.

## Results

Collected 2026-10-24: Geant4 job 2022 and mixture-proposal PyRITE job 2324,
20,000 PyRITE primaries per case. **98 of 100 revised gates pass.** Both
failures are reflected W at 300 keV in the 1–10 keV band:

| Observable | PyRITE | Geant4 | Combined SE | Gate limit |
|---|---|---|---|---|
| Redistributed photons / primary | 0.00012457 | 0.00010420 | 0.00000369 | 0.00001106 |
| Redistributed fraction | 0.34205 | 0.30737 | 0.00905 | 0.02715 |

The redistributed yield is 19.6% high (5.5 combined SE); the fraction is
0.03468 high (3.8 combined SE). The cause remains unresolved. The claim is
`discrepancy`, not `anchored`. The v1 uniform-proposal job 2048 passed only
with broad, heavy-tailed errors and is not anchoring evidence; its reductions
remain in `results-v1-uniform/` on the box. In the gated redistributed-yield
bands of v2, PyRITE SE / Geant4 SE ranges from 0.36 to 1.42.

The reducer now preserves paired history-block covariance for redistributed
fractions and continuum-subtracted line counts. Previously it treated the
fraction denominator as exact and combined line-window errors as independent.
The two failures persist with these repairs. Gate tolerances are unchanged.

`provenance.json` pins the original and committed reduction hashes, job IDs,
proposal, and remote-verified source hashes. The tested pre-rebase commit is
`c2cb361a`; its source sampler, emission scorer and benchmark hashes match
the implementation in `13dc5f2b`. The compact gzip records retain each
history block's counts projected onto every reported band and line window,
plus sample sizes and original metadata. They discard fine spectral shape;
they preserve every mean, SE and covariance used by this check. The full
65 MB JSON reductions remain in the staging checkout on the box.

To compact a freshly reduced file, then reproduce the failing comparison:

```bash
uv run python checks/photon_full_track/compare.py compact RAW.json OUTPUT.json.gz
uv run python checks/photon_full_track/compare.py --output report.json
```

The comparison requires all eight cases and returns 1 for the two known
failures. Fast regression tests require exactly those failures, so a changed
outcome requires explicit adjudication rather than silently relaxing a gate.

`emission-angles.txt` records job 2293 against the Geant4 100,000-primary
pilot logs (10,000 PyRITE primaries per case). It is a report-only diagnostic:
some large photon-energy fractions have more forward PyRITE emission, with
mean-cosine differences up to 9.9 nominal SE. These selected-photon errors
are not history-clustered; they are descriptive, not significance gates.
This source difference does not identify the cause of the low-energy W
redistribution failure. See the [validation evidence](../../docs/validation/radiation-physics/photon-full-track-thick-brem.md).
