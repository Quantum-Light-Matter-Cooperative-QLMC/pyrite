# Comparator engines for the equal-precision benchmark

Feasibility notes for the non-Geant4 comparators in
[#510](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/510),
gathered from upstream sources on 2026-10-18. Nothing here was built or run;
it fixes what an adapter must do and which settings it must record. The frozen
matrix and capability audit live in the
[protocol page](../../validation/beam-transport/equal-precision-protocol.md).

## Recommendation

Build the PenRed adapter next, and label it **PenRed (PENELOPE-2018 physics)**,
never standalone PENELOPE-2024. It is the only PENELOPE-family code that is
publicly downloadable, builds with CMake, exposes the per-interaction hooks a
creation-level photon tally needs, and reports per-history statistics. Request
PENELOPE-2024 in parallel through the channel below, since it is the
comparator the issue names. EGSnrc is worth a third arm only after the
two-engine contract works, but it is the closest match to PyRITE's
continuous-loss (straggling-off) core and should not be dropped outright.

## PENELOPE-2024 (standalone)

- **Access:** distributed by the OECD/NEA Data Bank. The NEA states that the
  United States is not a Data Bank participating country and that codes reach
  US users through exchange with RSICC at Oak Ridge; the penEasy manual (for
  PENELOPE-2018) also names RSICC as the North American distributor. Whether
  RSICC lists the 2024 release is **unverified**; check its catalog before
  filing a request.
- **Build/run:** Fortran sources with a user main program (penmain or a custom
  main); controls per material are `EABS(e⁻, γ, e⁺)`, `C1`, `C2`, `WCC`, `WCR`,
  and per body `DSMAX`.
- **Blocker:** licensed access only. Record the request date and outcome as an
  explicit deferral in the matrix until it arrives.

## PenRed

| Item | Value |
| --- | --- |
| Source pin | `145a8a357f08affa551f592300fa6dfcab98a762` (2026-10-01, CHANGELOG 1.15.2) |
| License | AGPL-3.0-or-later; keep the custom tally as a separate patch and publish it with the evidence |
| Physics | PENELOPE-based; its material files are written in the `PENELOPE (v. 2018)` format (`src/kernel/materials/source/pen_material.cpp`) |
| Build | C++14, CMake ≥ 3.11, `cd src && bash compile.sh`, executable `src/compiled/mains/pen_main`; threads and MPI are compile options. `pip install pyPenred` gives a Python binding |
| Run | `pen_main config.in`; keys are paths such as `materials/si/C1 0.05` |
| Seeds | `simulation/seed1`, `seed2`, or `seedPair` (the only option with threads/MPI) |

**Scene mapping.** Use a `PEN_QUADRIC` box of 10 × 10 mm × t in void. Void
outside one convex body gives first-exit termination with no re-entry. Use
the generic source with `kpar "electron"`, a `POINT` spatial sampler on the
entrance face, a zero-aperture `CONE` direction and a `MONOENERGETIC` energy.
Materials are created at runtime from `elements/Z` mass fractions plus
`density`, with `force-creation true`, so composition and density match the
matrix exactly. Record the generated `.mat` file's hash.

**Controls per case.**

- `eabs/electron 1.0e4`, matching the 10 keV tracking threshold.
- `eabs/gamma` and `eabs/positron` above E₀. Photons are then absorbed at
  creation, as in the PyRITE source tally.
- `WCR 1.0e3`, matching the 1 keV hard-photon threshold, but tested at
  0.5–2 keV for convergence.
- `WCC` above E₀ to suppress delta rays. All inelastic loss then becomes soft,
  which changes PENELOPE's step control.
- `C1`, `C2` and `geometry/dsmax` refined until observables move less than a
  third of the statistical budget, starting at 0.05/0.05.

Because PENELOPE always samples soft-loss fluctuations, pair PenRed with the
PyRITE `straggling` variants, not with the straggling-off core.

**Tallies.** Built-in tallies (`EMERGING_PART_DISTRIB`, `SECONDARY_GEN`,
`IMPACT_DET`) report per-history means and uncertainties, but they do not keep
per-primary arrays or creation-level photons. A small custom tally is needed.
It writes one record per history, in the schema of `pyrite.perf.comparison`:

- **Terminal state:** in `tally_endPart`, for the primary (`ILB[0] == 1`),
  classify by exit face or by absorption below `EABS`.
- **Transmitted energy:** the primary's energy at its forward exit.
- **Source photons:** record `state.E` in `tally_step`, which runs after the
  soft loss is applied. Then, in `tally_knock` with
  `icol == BETAe_HARD_BREMSSTRAHLUNG` (2), the photon energy is that value
  minus the post-knock `state.E` (`src/mains/includes/pen_loops.hh`, step at
  lines 212–219, knock at 326–331).

Validate the hook arithmetic on a few histories against the photon energies
written to a phase-space file before trusting it.

## EGSnrc

- **Pin:** latest release tag `v2026` (`bd49a47aae3938f49f4100b8e018091861408b19`).
  It is open source (AGPL) on GitHub.
- **Fit:** class II condensed history (PRESTA-II) with no straggling below
  `AE`/`AP`. It can therefore express PyRITE's continuous-loss core more
  closely than PENELOPE. Electron single scattering is limited to boundary
  regions by the EXACT boundary-crossing algorithm.
- **Starting point:** the `tutor7pp` C++ application already scores reflected,
  transmitted and deposited energy in a slab. It also enables `AfterBrems`
  `ausgab` calls (`HEN_HOUSE/user_codes/tutor7pp/tutor7pp.cpp`), which is the
  hook for creation-level photon scoring.
- **Controls to record:** `ECUT`, `PCUT`, `AE`, `AP`, `ESTEPE`, `XImax`, the
  boundary-crossing algorithm and skin depth, spin effects, the brems cross
  sections and angular sampling, and relaxation switches.
- **Status:** deferred until the PyRITE–Geant4 and PyRITE–PenRed accuracy and
  scoring contracts pass.

## Geant4 control mapping (existing arm)

The pilot's TestEm5 controls are listed in the
[pilot report](../../validation/beam-transport/equal-precision-pilot.md).
For the full matrix, dump the converted production thresholds per material and
particle for every case, since a 1 nm range cut converts differently in Si, W,
C and SiO₂. Treat `empenelope` and `emstandardSS` as separate arms.
`empenelope` is the Geant4 implementation of PENELOPE models; it does not
stand in for either PenRed or PENELOPE-2024.

## Sources

- [PenRed repository](https://github.com/PenRed/PenRed) and its bundled
  `doc/PenRed_usage_guide.pdf` (material, tally and seed sections)
- [Giménez-Alventosa et al., Comput. Phys. Commun. 267 (2021) 108065](https://doi.org/10.1016/j.cpc.2021.108065)
- [PENELOPE-2024 manual landing page](https://www.oecd-nea.org/jcms/pl_108003/penelope-2024-a-code-system-for-monte-carlo-simulation-of-electron-and-photon-transport)
- [NEA PENELOPE publication page](https://www.oecd.org/en/publications/penelope-a-code-system-for-monte-carlo-simulation-of-electron-and-photon-transport_9d2cc3d5-en.html) and [IAEA note on Data Bank and RSICC access](https://iaea.org/about/organizational-structure/department-of-nuclear-energy/iaea-member-states-access-to-the-oecd-nea-data-bank)
- [penEasy manual](https://inte.upc.edu/en/downloads/peneasy/usermanual.pdf) (RSICC as North American distributor)
- [EGSnrc repository](https://github.com/nrc-cnrc/EGSnrc) and [transport controls](https://nrc-cnrc.github.io/EGSnrc/doc/pirs898/common.html)
