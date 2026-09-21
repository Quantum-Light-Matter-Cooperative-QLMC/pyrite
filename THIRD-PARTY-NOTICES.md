# Third-Party Notices

PyRITE is distributed under the UCLA Academic Software License (see
`LICENSE.txt`; academic/nonprofit use only). The external codes below are
integrated as separate driver-tier tooling (tracked under
`agentdocs/specs/2026-09-21-external-fortran-code-integration.md` and GitHub
issue #161) and carry their own licenses and attribution requirements. This
notice lists them in advance of that integration landing, per issue #162.

None of PyRITE's own source is licensed under any of the terms below. This
file exists to satisfy the attribution obligations that follow from invoking
these codes and from redistributing tables derived or resampled from their
output.

## ELSEPA (2020 release)

- **License**: CC BY-NC 3.0 (Creative Commons Attribution-NonCommercial 3.0
  Unported), as labeled on the official Mendeley Data archive.
- **Archive / citation**: ELSEPA 2020/2021, archive version 1, DOI
  [10.17632/w4hm5vymym.1](https://doi.org/10.17632/w4hm5vymym.1); program
  article DOI [10.1016/j.cpc.2020.107704](https://doi.org/10.1016/j.cpc.2020.107704).
- **Nature of PyRITE's use**: PyRITE invokes the ELSEPA `elscata` program to
  generate elastic electron/positron scattering cross-section tables and does
  not vendor or redistribute ELSEPA source. Because PyRITE is itself
  distributed under a nonprofit-only license, the CC BY-NC clause imposes no
  additional restriction. Tables derived or resampled from ELSEPA output are
  adaptations under the license and are marked as such at their point of use.

## SBETHE

- **License**: CC BY-NC 3.0 (Creative Commons Attribution-NonCommercial 3.0
  Unported), verified against source headers and upstream documentation on
  2026-09-21 (see the design spec cited above).
- **Nature of PyRITE's use**: PyRITE invokes the SBETHE program to generate
  material-scoped stopping-power and related inelastic-scattering tables and
  does not vendor or redistribute SBETHE source. Because PyRITE is itself
  distributed under a nonprofit-only license, the CC BY-NC clause imposes no
  additional restriction. Tables derived or resampled from SBETHE output are
  adaptations under the license and are marked as such at their point of use.

## BremsLib (2.0.8)

- **License**: GPL-3.0-or-later for the Fortran sources (GPL headers present
  in `Interpolate_DCS/Interpolate_DCS.f90` and in `Brems/{Brems,
  Bremsstrahlung, FitExp, Born_SM_appr, Brems_common}.f90` plus
  `Read_S_integrals.f90`, per source-header verification on 2026-09-21). The
  CC BY 4.0 terms that apply to the separately published dataset deposit
  apply to that dataset only, not to the code.
- **Nature of PyRITE's use**: PyRITE invokes BremsLib as an external
  subprocess to generate bremsstrahlung cross-section tables and does **not**
  vendor the BremsLib Fortran sources and does **not** port or translate
  BremsLib routines (in particular `Interpolate_DCS.f90` and
  `Brems_CS_interp.f90`) into PyRITE's own code, since a translation would be
  a GPL-3 derivative work incompatible with PyRITE's nonprofit-only
  distribution terms (GPL-3 section 7 forbids adding field-of-use
  restrictions). Running the program and consuming its output is
  unrestricted by the GPL, which does not reach program output. Tables
  derived or resampled from BremsLib output are adaptations under the
  license and are marked as such at their point of use.
