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
shards per case.

## Comparison

```bash
uv run python checks/mev_full_track/compare.py reduce-geant4 RAW_DIR
uv run python checks/mev_full_track/compare.py reduce-pyrite RAW_DIR
uv run python checks/mev_full_track/compare.py --output report.json
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

## Lab-box table incident

The first PyRITE attempts on the lab box ran against 28 ELSEPA stub tables in
the box's workspace store (#654). These gave near-isotropic scattering, so
95 % of 1 MeV electrons backscattered from 10 µm Si. The stubs were
quarantined and the verified tables restored before job 1775. No result here
comes from the stubs.
