# MeV full-track transport against Geant4 (issue #439)

An opt-in comparison of PyRITE's production electron transport with Geant4
11.4.2 TestEm5 for W and Si at 3, 10, 30, 50 and 100 MeV. It is the evidence
for the `mev-full-track-transport` ledger claim and does not sign it off.
Heavy runs go to the lab box; the repository keeps the scripts, the reduced
Geant4 references (`reference/*.json`) and the reduced PyRITE results
(`results/*.json`).

## Cases

Electron pencil beams at normal incidence on a 1 cm × 1 cm slab:

| Target | Thickness | t/X₀ | Energies | Primaries per case |
|---|---:|---:|---|---:|
| W, 19.3 g/cm³ | 100 µm | 0.029 | 3, 10, 30, 50, 100 MeV | 100,000 |
| Si, 2.329 g/cm³ (Geant4 2.33) | 1 mm | 0.011 | 3, 10, 30, 50, 100 MeV | 100,000 |

Both targets are below the thick-target screen, so photon transport is not
needed, and radiative loss is a large share of the energy loss (it dominates
in W above ~10 MeV).

## Geant4 reference

The TestEm5 source is the #182 emission build
(`../full_track_bremslib/GEANT4_BUILD.md`, `geant4-testem5-emission.patch`:
histogram 63 and one `EBREM` line per primary eBrem photon). On top of it,
`patch_tracking.py` prints one `EXIT` line per primary: event, fate (`T`
transmitted, `R` reflected, `A` otherwise), kinetic energy and direction. A
primary that ends with `A` and nonzero energy left through a side face.

`make_macros.py` writes the ten macros. The physics matches PyRITE's scope:

- `empenelope`, Goudsmit–Saunderson multiple scattering;
- no loss fluctuations, no fluorescence;
- 10 keV electron stop;
- photons above the 990 eV PenBrem floor (1 nm gamma range cut);
- no delta rays (1 km electron range cut);
- every secondary killed at creation without depositing energy
  (`killSecondaries 2`), so only the primary is transported.

TestEm5's printed "energy deposit" is `E0 − leakage`, so with killed photons
it includes their energy. The comparison therefore takes the continuous
deposit per event from the logs, as `E0 − E_exit − Σk`.

`run_geant4.sbatch` runs one macro per array task (`cases.txt` lists them).
Lab-box job 1706 (2026-10-22) produced the logs reduced into `reference/`.

## PyRITE

`pyrite_bench.py` runs the production stack on the same cases:

- ELSEPA elastic scattering (Si: catalog muffin-tin table);
- SBETHE collision stopping;
- coupled BremsLib radiative loss at `kc` = 1 keV, with Ter-Mikaelian
  suppression at the layer plasma energy and the SB85 e–e term;
- Kawrakow atomic-electron deflection;
- midpoint energy, no straggling, 10 keV stop, per-electron CPU core;
- the same 1 cm footprint.

Only the primary is transported. Elastic scattering is analog, so a primary
scattered to grazing can need ~10⁵ steps (W at 3 MeV). The driver therefore
raises `max_steps` to 300,000 and runs 50-primary batches, because rows are
preallocated as batch × `max_steps`. `run_pyrite.sbatch` runs ten 10,000-primary
shards per case. `run_control.sbatch` runs the point-nucleus control below.

## Comparison

```bash
uv run python checks/mev_full_track/compare.py reduce-geant4 RAW_DIR
uv run python checks/mev_full_track/compare.py reduce-pyrite RAW_DIR
uv run python checks/mev_full_track/compare.py --output report.json
uv run python checks/mev_full_track/compare.py reduce-pyrite RAW_DIR --variant point
uv run python checks/mev_full_track/compare.py --variant point
```

The gates were fixed before any PyRITE result was inspected. An observable
passes when `|PyRITE − Geant4| ≤ max(5 % of Geant4, 3σ)`, with σ combining
both per-primary standard errors. Gated: the fraction not transmitted; the
mean continuous deposit; the mean energy loss of transmitted primaries; the
mean and rms exit angle; photon energy per primary for `k ≥ 10 keV` and
`k ≥ 100 keV`; and the photon count for `k ≥ 100 keV`.

Reported, not gated:

- photon counts above 1 and 10 keV, because PenBrem has no Ter-Mikaelian
  suppression (`γħω_p` reaches 16 keV for W at 100 MeV);
- the isolated model check: BremsLib vacuum moments on Geant4's own
  track-length spectrum (h63) against Geant4's photons;
- the implementation check: PyRITE's realized photons against BremsLib
  moments on its own track-length spectrum;
- Kolmogorov distances of the binned loss, angle and photon spectra.

## Results

Lab-box job 1840 (2026-10-22), 100,000 primaries per case, PyRITE/Geant4:

| Case | Not transmitted | Deposit | Transmitted loss | θ_mean | θ_rms | Photon E ≥ 10 keV | Photon N ≥ 100 keV | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W 3 MeV | 1.029 | 1.035 | 1.027 | 0.993 | 0.993 | 1.023 | 1.039 | pass |
| W 10 MeV | 0.917 | 1.013 | 1.001 | 0.996 | 0.993 | 0.988 | 1.010 | pass |
| W 30 MeV | 0.63 | 1.005 | 1.009 | 0.991 | 0.965 | 1.012 | 1.008 | pass |
| W 50 MeV | 0.83 | 1.001 | 1.024 | 0.987 | **0.928** | 1.029 | 1.020 | θ_rms fails |
| W 100 MeV | 0.81 | 0.998 | 1.023 | 0.989 | **0.926** | 1.026 | 1.019 | θ_rms fails |
| Si 3 MeV | 1.082 | 1.009 | 1.009 | 1.000 | 0.999 | 1.036 | 1.052 | pass |
| Si 10 MeV | 0.91 | 1.002 | 1.009 | 1.007 | 1.006 | 1.039 | 1.032 | pass |
| Si 30 MeV | 1.13 | 1.001 | 1.005 | 0.997 | 0.982 | 1.008 | 1.028 | pass |
| Si 50 MeV | 1.00 | 1.000 | 1.017 | 0.994 | 0.949 | 1.032 | 1.011 | pass |
| Si 100 MeV | 0.33 | 0.999 | 1.006 | 0.999 | 0.959 | 1.008 | 1.015 | pass |

Non-transmitted fractions above 10 MeV are a few to a few tens of primaries,
so their ratios carry large errors. All of them pass within 3σ.

- **Exit-angle tail.** For W at 50 and 100 MeV, the rms over θ below
  0.2–0.5 rad agrees within 0.5 %. The θ_rms deficit comes from the 0.1–1 %
  of primaries that leave above ~0.3 rad, where PyRITE has 15–45 % fewer.
  Geant4's GS Mott correction uses the Mott DCS of a point nucleus
  (`G4GSMottCorrection.hh`). PyRITE's ELSEPA tables use a Fermi charge
  distribution. In W, ELSEPA's Fermi/point DCS ratio is 0.37 at 1 rad and
  50 MeV, against an observed tail ratio of 0.32 ± 0.08. At 0.5 rad and
  100 MeV it is 0.54, against 0.60 ± 0.08.
- **Point-nucleus control.** Lab-box job 1968 reran W 50 and 100 MeV with a
  point-nucleus table from `point_nucleus_table.py` (`run_control.sbatch`,
  `pyrite_bench.py --elastic-npz`). Every gate passes. θ_rms is 0.981 and
  0.988 and θ_mean 0.998 and 0.999; the other observables are unchanged
  (`results/<case>_point.json`). The two failures are therefore a difference
  in the reference's nuclear model, not in transport.
- **Photons.** PyRITE's realized hard photons match its own BremsLib partition
  on its own track-length spectrum within 1.5 % in every case (the
  implementation check). On Geant4's track, BremsLib gives 1–4 % more photon
  energy than PenBrem (the isolated model check). Photon counts above 1 keV
  are 15–24 % below Geant4 for W at 30 MeV and above, because PenBrem has no
  dielectric suppression.
- **Loss spectrum.** Without fluctuations the transmitted-loss peak is nearly
  deterministic. For W at 30 and 100 MeV it falls on a log-bin edge, which
  is why `ks_loss` is about 0.5 there. The mean losses differ by less than
  0.5 %.

`tests/montecarlo/transport/test_mev_full_track.py` pins these reductions and the
control. It also reruns Si 30 MeV and W 10 MeV at 300 primaries against the
Geant4 gates; these runs are marked slow.

### ELSEPA point-nucleus runs

One `elscata` run of the point-nucleus deck over all 61 production energies
returns a first transport cross section that stops falling above ~1 MeV: 7×
the Fermi value at 25 MeV and 92× at 100 MeV. Run alone, the same energy
gives 1.09× and 1.27×. The first control (job 1948) used such a table and is
discarded. `point_nucleus_table.py` now runs one energy per call. Before
writing the table, it checks that the Fermi deck run the same way reproduces
the production table within 1e-6, so the production table is not affected.

## Lab-box table incident

The first PyRITE attempts on the lab box ran against 28 ELSEPA stub tables in
the box's workspace store (#654). These gave near-isotropic scattering, so
95 % of 1 MeV electrons backscattered from 10 µm Si. The stubs were
quarantined and the verified tables restored before job 1775. No result here
comes from the stubs.
