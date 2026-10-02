# Third-Party Notices

PyRITE is distributed under the UCLA Academic Software License (see
`LICENSE.txt`; academic/nonprofit use only). The external data and codes below
carry their own licenses and attribution requirements. The external codes are
integrated as separate driver-tier tooling (`pyrite.xsgen`, tracked under
`agentdocs/specs/2026-09-21-external-fortran-code-integration.md` and GitHub
issue #161); their notices were prepared in advance of that integration,
per issue #162.

None of PyRITE's own source is licensed under any of the terms below. This
file exists to satisfy the attribution obligations that follow from
redistributing or downloading these codes and data, from invoking them, and from redistributing
tables derived or resampled from their output.

## EPICS2025 EADL atomic relaxation data

- **Author**: D. E. Cullen, Livermore Evaluated Atomic Data Library (EADL),
  `NDS-IAEA-224`.
- **License**: Creative Commons Attribution 4.0 International,
  <https://creativecommons.org/licenses/by/4.0/>, as stated on the
  [EPICS2025 distribution page](https://nuclear.llnl.gov/EPICS/index.html).
- **Source**: the unmodified ENDF-6 `EADL2025.ALL` file from
  <https://nuclear.llnl.gov/EPICS/ENDF2025/EADL2025.ALL>, evaluated August
  2023 and distributed January 2025. SHA-256:
  `78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c`.
- **Nature of PyRITE's use**: PyRITE does not ship the file. `pyrite tables
  fetch eadl` downloads the published file unchanged from the URL above (or,
  when it is unreachable, from PyRITE's private mirror release
  `mirror-eadl2025-1`, which holds the same bytes) into the user's data
  directory, and PyRITE reads its File 28 (MT 533) subshell binding energies, occupancies, and
  radiative and nonradiative transition data for the characteristic-radiation
  relaxation cascade.

## EPICS2025 EPDL photon interaction data

- **Author**: D. E. Cullen, Livermore Evaluated Photon Data Library (EPDL),
  `NDS-IAEA-225`, distributed by the IAEA Nuclear Data Section.
- **License**: Creative Commons Attribution 4.0 International,
  <https://creativecommons.org/licenses/by/4.0/>, as stated for the EPICS
  library on the
  [EPICS2025 distribution page](https://nuclear.llnl.gov/EPICS/index.html).
- **Source**: the ENDF-6 file
  <https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL>, evaluated August 2023
  and distributed January 2025. SHA-256:
  `59bbd8c559685dda0bf0de2762bc43126f599cd154d635940f17b6a59c1c43fd`.
- **Nature of PyRITE's use**: PyRITE redistributes a **derived** table,
  `epdl2025_mf23.npz`, not the upstream file. It is published on PyRITE's
  release `tables-epdl-1` and installed with `pyrite tables fetch epdl`. It keeps five File 23 integrated cross sections (MT 522, 502, 504, 517,
  515) for Z = 1--100. **Modifications:** knots were removed where lin-lin
  interpolation through the kept knots reproduces every removed upstream value
  to 5e-4 relative, and cross sections are stored as float32. The generator is
  `scripts/release_epdl_table.py`; see the table's README for provenance.

## EPICS2025 EEDL electron interaction data

- **Author**: D. E. Cullen, Livermore Evaluated Electron Data Library (EEDL),
  `NDS-IAEA-226`, translated from ENDL to ENDF-6 and distributed by the IAEA
  Nuclear Data Section.
- **License**: Creative Commons Attribution 4.0 International,
  <https://creativecommons.org/licenses/by/4.0/>, as stated on the
  [EPICS2025 distribution page](https://nuclear.llnl.gov/EPICS/index.html).
- **Source**: the unmodified ENDF-6 file from
  <https://nuclear.llnl.gov/EPICS/ENDF2025/EEDL2025.ALL>, evaluated August
  2023 and distributed January 2025, installed as `EEDL.endf`. SHA-256 of
  the installed bytes:
  `f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c` (the
  published file with its final CRLF removed; see
  `src/pyrite/data/characteristic_cross_sections/README.md`).
- **Nature of PyRITE's use**: PyRITE does not ship the file. `pyrite tables
  fetch eedl` downloads it from the URL above (or, when it is unreachable,
  from PyRITE's private mirror release `mirror-eedl2025-1`, which holds the
  installed bytes) into the user's data directory,
  and PyRITE reads its File 23 subshell electroionization cross sections (MT
  534--572) and its File 23/26 bremsstrahlung totals and photon spectra (MT
  527).

## NIST SRD 64 (not redistributed)

- **Source**: NIST Standard Reference Database 64, *NIST Electron
  Elastic-Scattering Cross-Section Database*, <https://srdata.nist.gov/srd64/>.
- **Terms**: NIST Standard Reference Data are copyrighted by the U.S.
  Secretary of Commerce under the Standard Reference Data Act and may not be
  reproduced or redistributed without prior permission
  (<https://www.nist.gov/srd/public-law>).
- **Nature of PyRITE's use**: PyRITE ships none of it. Earlier revisions
  packaged five exported transport cross-section tables
  (`mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv`); they were
  removed from the tree and the wheel under #263 and remain only in Git
  history. The opt-in `elastic_model="mott"` reads tables each user exports
  from SRD 64 into the directory named by the `mott.tables_dir` config key.
  The test suite uses synthetic tables in the same file format
  (`tests/data/mott_srd64_synthetic/`), whose numbers are computed from an
  analytic formula and are not NIST data.

ELSEPA and SBETHE are both CC BY-NC 3.0. PyRITE is itself distributed for
academic/nonprofit use only, so the NonCommercial clause imposes no
additional restriction, and neither carries ShareAlike, so no copyleft
reaches PyRITE's own code from either. Both require attribution, and both
require derived or adapted material to be marked as such.

## ELSEPA (2020 release)

- **Authors**: Francesc Salvat, Aleksander Jablonski, Cedric J. Powell.
- **License**: CC BY-NC 3.0 (Creative Commons Attribution-NonCommercial 3.0
  Unported), <https://creativecommons.org/licenses/by-nc/3.0>, as recorded on
  the Mendeley Data deposit.
- **Archive**: DOI [10.17632/w4hm5vymym.1](https://doi.org/10.17632/w4hm5vymym.1)
  (version-pinned). Program article: DOI
  [10.1016/j.cpc.2020.107704](https://doi.org/10.1016/j.cpc.2020.107704).
- **Nature of PyRITE's use**: PyRITE **redistributes** the ELSEPA Fortran
  source and its `database/` directory in its source distribution (the
  repository checkout and the sdist, under `vendor/xsgen/elsepa/`; not the
  wheel), so that elastic-table generation works offline. The upstream
  test-run output `dcs_1p000e03.dat` is redistributed unchanged as a test
  fixture under `tests/data/xsgen/elsepa/`. PyRITE compiles that source
  and invokes the `elscata` program to generate elastic scattering
  cross-section tables. Redistribution is unmodified unless a modification is
  recorded at the vendored tree; tables derived or resampled from ELSEPA
  output are **adaptations** under the license and are marked as such in the
  provenance manifest stored beside each generated table.

## SBETHE

- **Authors**: Francesc Salvat, Pedro Andreo.
- **License**: CC BY-NC 3.0 (Creative Commons Attribution-NonCommercial 3.0
  Unported), <https://creativecommons.org/licenses/by-nc/3.0>, as recorded in
  the Mendeley Data deposit's own license metadata (verified 2026-09-21).
- **Archive**: DOI [10.17632/7zw25f428t.2](https://doi.org/10.17632/7zw25f428t.2)
  (version-pinned), <https://data.mendeley.com/datasets/7zw25f428t/2>.
- **Nature of PyRITE's use**: PyRITE **redistributes** the SBETHE Fortran
  source `sbethe.f` in its source distribution (the repository checkout and
  the sdist, under `vendor/xsgen/sbethe/`; not the wheel), compiles it, and invokes
  the resulting program to generate material-scoped stopping-power and
  related inelastic cross-section tables. PyRITE does **not** redistribute
  the accompanying `sdbase/` database, the bundled prebuilt Windows binary,
  or the bundled documentation; `sdbase/` is downloaded on demand from the
  pinned deposit above into the user's own data directory. Tables derived or
  resampled from SBETHE output are **adaptations** under the license and are
  marked as such in the provenance manifest stored beside each generated
  table. PyRITE **redistributes** the stopping tables for its catalogue
  materials as one SHA-256-pinned archive on its release `tables-sbethe-1`,
  installed with `pyrite tables fetch sbethe-tables`; the release notes carry
  this attribution and the modifications statement.

## BremsLib (2.0.8)

- **Author**: Andrius Poškus.
- **Citation**: Poškus, Andrius (2025), “BremsLib v2.0.8”, Mendeley Data, V9,
  DOI [10.17632/6zfsc9xsz8.9](https://doi.org/10.17632/6zfsc9xsz8.9).
- **License**: two distinct sets of terms cover this deposit, and they must
  not be conflated.
  - The **Fortran sources** are GPL-3.0-or-later (GPL headers present in
    `Interpolate_DCS/Interpolate_DCS.f90` and in `Brems/{Brems,
    Bremsstrahlung, FitExp, Born_SM_appr, Brems_common}.f90` plus
    `Read_S_integrals.f90`, per source-header verification on 2026-09-21).
  - The **dataset deposit** is CC BY 4.0 International,
    <https://creativecommons.org/licenses/by/4.0>, confirmed from the
    deposit's own license metadata on 2026-09-21. Note that the deposit's own
    terms caution that further permission may be required for content within
    it identified as belonging to a third party — which is how the GPL-3
    sources bundled inside it are to be read.
- **Nature of PyRITE's use**: PyRITE reads a precomputed BremsLib library
  from a local checkout that the user obtains themselves, and interpolates
  from it; it does not build or run any BremsLib program. The three artifacts
  are treated differently:
  - PyRITE does **not** redistribute the GPL-3 Fortran sources, and does
    **not** port or translate BremsLib routines (in particular
    `Interpolate_DCS.f90` and `Brems_CS_interp.f90`) into its own code, since
    a translation would be a GPL-3 derivative work incompatible with PyRITE's
    nonprofit-only distribution terms (GPL-3 section 7 forbids adding
    field-of-use restrictions). Clean-room reimplementation from the published
    manuals is how the equivalent functionality is obtained.
  - PyRITE does **not** redistribute the precomputed data library itself,
    which is impractically large.
  - PyRITE **does** redistribute cross-section tables **derived** from that
    library, for every element its built-in catalogue materials may contain.
    They are not in the PyRITE package: they are published as one
    SHA-256-pinned archive (PyRITE release `tables-bremslib-1`) that
    `pyrite tables fetch bremslib` installs into
    the user's own data directory. These are **adaptations** of a CC BY 4.0
    work and are not the upstream data: they are parsed and restructured on
    the library's own grids, per-node angular integrals are added, the
    double-differential cross section is reduced to single precision, and the
    per-point uncertainties are removed. Andrius Poškus is credited as the
    author of the source dataset, the CC BY 4.0 licence is linked above, and
    each table carries a provenance manifest recording the upstream deposit
    version and the nature of the modifications, as CC BY 4.0 requires.
- **Documentation**: the manuals `BremsLib_v2.0.pdf` and
  `Interpolate_DCS.pdf` are published alongside the library in the same
  deposit and are the basis for any clean-room reimplementation.
