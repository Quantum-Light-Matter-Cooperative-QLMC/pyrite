# Characteristic-radiation cross sections

`EEDL.endf` is the 2025 Livermore Evaluated Electron Data Library (EEDL),
translated from ENDL to ENDF-6 by D. E. Cullen and distributed by the IAEA
Nuclear Data Section. PyRITE reads subshell electroionization cross sections
from ENDF File 23, MT 534--572. The source file identifies itself as
`NDS-IAEA-226`, evaluated August 2023 and distributed January 2025.

The packaged bytes are the EPICS2025 distribution downloaded verbatim from
<https://nuclear.llnl.gov/EPICS/ENDF2025/EEDL2025.ALL>, pinned by SHA-256
`f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`.
Upstream records are 75 columns with CRLF line endings. The pin asserts those
published bytes, so `.gitattributes` marks this file `-text` to exempt it from
the repository's `eol=lf` normalization; re-normalizing it would break the pin.
Characteristic line energies (where xraydb tabulates the transition), optional
Elam fluorescence yields, and natural atomic-level widths are supplied
separately by xraydb at runtime. PyRITE sums the initial- and final-level widths to obtain
each transition's Lorentzian FWHM. The resolved xraydb version is included in
PyRITE's characteristic-model identity marker so a relaxation-database update
cannot reuse an older checkpoint.

`EADL2025.ALL` is the EPICS2025 Livermore Evaluated Atomic Data Library
(EADL), also by D. E. Cullen. It supplies atomic relaxation data in ENDF-6
File 28, MT 533, for elements Z=1--100. Its header identifies the evaluation
as `NDS-IAEA-224`, evaluated August 2023 and distributed January 2025.
The file was downloaded without modification from
<https://nuclear.llnl.gov/EPICS/ENDF2025/EADL2025.ALL> and is pinned by
SHA-256 `78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c`.
It retains the upstream 75-column CRLF records through the `-text`
`.gitattributes` rule. EPICS distributes the data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the repository's
[third-party notices](../../../../THIRD-PARTY-NOTICES.md) for attribution.

PyRITE reads EADL File 28 for the characteristic-radiation relaxation
cascade: subshell binding energies, occupancies, and radiative and
nonradiative transition probabilities and energies. xraydb still supplies line
energies where it tabulates the same level pair, natural level widths, and the
optional Elam fluorescence yields. The EADL checksum prefix is part of the
characteristic-model identity marker.
