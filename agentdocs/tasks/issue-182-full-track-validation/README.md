# Issue #182 — independent coupled BremsLib full-track validation

Branch: `issue-182-full-track-validation`; canonical backlog item: GitHub #182.
This branch holds a preliminary Geant4 11.4.2 TestEm5 benchmark. The reference
source and macros are also preserved under `~/dev/geant4/issue-182/` on the
original workstation. The maintained evidence is in
`checks/full_track_bremslib/` and the two BremsLib validation write-ups.

## Completed slice

- Pinned Geant4 TestEm5 source and patch to score `eBrem` photons separately;
  W and Si 300 keV, 10 keV electron stop, 10,000 events each.
- Ran exact PyRITE CPU per-electron W/Si at hard cutoffs 1, 5, 10 keV;
  recorded raw JSON, source spectra, electron terminal fractions, energy
  debit, and full-track event/recoil checks. Repeated W with screened-Rutherford
  elastic scattering to investigate the Mott electron mismatch.
- Recorded provenance, model differences, approximate screening tolerances,
  and preliminary findings. Ledger status remains `rederived`.

- Added 800 keV W/Si (Mott) plus W and Si screened-Rutherford controls.
  Photon yields ≥10 keV agree within 1.91σ in all seven 1 keV runs. The
  comparison now asserts radiative observables only; terminal fractions are
  reported. Neither PyRITE elastic model matches Geant4 GS across the grid;
  Si 800 keV Mott backscatter is 4× Geant4 (+9.67σ), a candidate Mott defect
  tracked in #183.

- Added the 100k-primary emission benchmark (2026-09-26). The Geant4 patch
  logs primary eBrem kinematics and scores the primary track-length spectrum.
  PyRITE runs 1, 5 and 10 keV cutoffs from a checkout verified against
  `fa20e0dd`. `test_compare_emission.py` passes all 76 checks against
  tolerances fixed before the PyRITE results were inspected: isolated
  BremsLib versus PenBrem on Geant4 paths, the implementation on PyRITE paths,
  hard plus soft cutoff convergence, photon angle, and recoil in a common
  convention. The Geant4 PenBrem recoil convention (deflected electron) is
  documented. Both ledger rows stay `rederived`, with notes prepared for
  sign-off.

## Remaining acceptance work

- Human review and sign-off of both ledger rows (agents must not mark
  `signed-off`).
- The Si 800 keV Mott backscatter excess, tracked in #183, is outside the
  BremsLib claims. It blocks electron-yield use of Mott at low Z, and it
  explains the direct W 300 keV full-track photon-energy excess (+10 %, from
  +11 % path length).
- The directional point-detector scorer, escape and detected yield remain
  unvalidated (#171 above 800 keV).
- The comparison was run with the pre-`ruff format` `test_benchmark.py`
  (sha256 `5b255052…`); the committed version differs only in formatting.
- Issue #172 can use `checks/full_track_bremslib/README.md` as its
  validation gate once the rows are signed off.

## Reproduction

Follow `checks/full_track_bremslib/README.md` and `GEANT4_BUILD.md`.
The emission runs used `geant4-testem5-emission.patch`, `*_emission.mac` and
`run_emission.sbatch`; the compute node has its own home, so stage the
pinned checkout and driver there. The reference runs used Geant4 commit
`8cc04f65977807f1848da7b958c421cd5e162f26`; PyRITE runs used
`fa20e0ddc4aa1dcd412dd9837e295f22210a09ae` with locked dependencies.
All heavy transport was run on the lab CPU node.
