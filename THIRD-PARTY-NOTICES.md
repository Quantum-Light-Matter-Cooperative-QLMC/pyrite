# Third-Party Notices

PyRITE is distributed under the UCLA Academic Software License (see
`LICENSE.txt`; academic/nonprofit use only). The external codes below are
integrated as separate driver-tier tooling (`pyrite.xsgen`, tracked under
`agentdocs/specs/2026-09-21-external-fortran-code-integration.md` and GitHub
issue #161) and carry their own licenses and attribution requirements. This
notice lists them in advance of that integration landing, per issue #162.

None of PyRITE's own source is licensed under any of the terms below. This
file exists to satisfy the attribution obligations that follow from
redistributing these codes, from invoking them, and from redistributing
tables derived or resampled from their output.

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
  source and its `database/` directory inside the PyRITE distribution, so
  that elastic-table generation works offline. PyRITE compiles that source
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
- **Archive**: DOI [10.17632/7zw25f428t.1](https://doi.org/10.17632/7zw25f428t.1)
  (version-pinned), <https://data.mendeley.com/datasets/7zw25f428t/1>.
- **Nature of PyRITE's use**: PyRITE **redistributes** the SBETHE Fortran
  source `sbethe.f` inside the PyRITE distribution, compiles it, and invokes
  the resulting program to generate material-scoped stopping-power and
  related inelastic cross-section tables. PyRITE does **not** redistribute
  the accompanying `sdbase/` database, the bundled prebuilt Windows binary,
  or the bundled documentation; `sdbase/` is downloaded on demand from the
  pinned deposit above into the user's own data directory. Tables derived or
  resampled from SBETHE output are **adaptations** under the license and are
  marked as such in the provenance manifest stored beside each generated
  table.

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
- **Nature of PyRITE's use**: PyRITE redistributes **nothing** from
  BremsLib — not the GPL-3 Fortran sources, not the data library, and not
  tables derived from it. PyRITE reads a precomputed BremsLib library from a
  local checkout that the user obtains themselves, and interpolates from it;
  it does not build or run any BremsLib program. PyRITE does **not** port or
  translate BremsLib routines (in particular `Interpolate_DCS.f90` and
  `Brems_CS_interp.f90`) into its own code, since a translation would be a
  GPL-3 derivative work incompatible with PyRITE's nonprofit-only
  distribution terms (GPL-3 section 7 forbids adding field-of-use
  restrictions). Clean-room reimplementation from the published manuals is
  how the equivalent functionality is obtained.
- **Documentation**: the manuals `BremsLib_v2.0.pdf` and
  `Interpolate_DCS.pdf` are published alongside the library in the same
  deposit and are the basis for any clean-room reimplementation.
