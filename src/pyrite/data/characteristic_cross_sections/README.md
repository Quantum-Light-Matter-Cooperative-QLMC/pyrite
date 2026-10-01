# Characteristic-radiation cross sections

`EEDL.endf` is the 2025 Livermore Evaluated Electron Data Library (EEDL),
translated from ENDL to ENDF-6 by D. E. Cullen and distributed by the IAEA
Nuclear Data Section. PyRITE reads subshell electroionization cross sections
from ENDF File 23, MT 534--572. The source file identifies itself as
`NDS-IAEA-226`, evaluated August 2023 and distributed January 2025.

Neither file is packaged any more (ADR-0014, #263). Both are hash-pinned
fetched datasets: `pyrite tables fetch eedl` and `pyrite tables fetch eadl`
download them from the URLs below, or copy a local file given with
`--archive PATH`, verify the SHA-256, and install them at
`<data root>/datasets/eedl/EEDL.endf` and `<data root>/datasets/eadl/EADL2025.ALL`.
The data root is the platform user data directory, or the selected workspace
(`PYRITE_HOME`, `workspace.root`). `pyrite remote sync` ships them to the box.
The pins live in `pyrite/datasets.py`; the loaders verify them before first use,
and a missing file fails naming the fetch command. This README keeps the
provenance record.

EEDL comes from the EPICS2025 distribution at
<https://nuclear.llnl.gov/EPICS/ENDF2025/EEDL2025.ALL>. The installed bytes
are pinned by SHA-256
`f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`.
Upstream records are 75 columns with CRLF line endings. The published file
(SHA-256 `ce37912435e0b8002f85878f98ccf7c5840cb168f1d46af3c9e915cd16c70ccc`,
`Last-Modified` 2025-02-01, checked 2026-09-30) ends with one more CRLF than
the bytes PyRITE vetted and packaged until #263. The fetch accepts either form
and installs the vetted form by dropping that final CRLF, so the pin, and the
model identity markers built from it, stay unchanged; no ENDF record differs.
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
It is installed without modification from
<https://nuclear.llnl.gov/EPICS/ENDF2025/EADL2025.ALL> and is pinned by
SHA-256 `78ccf8a4e07c1c120a2e3d94ff051aab2180d151f35e8bc3406d52df5af5e88c`,
with the upstream 75-column CRLF records. EPICS distributes the data under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see the repository's
[third-party notices](../../../../THIRD-PARTY-NOTICES.md) for attribution.

PyRITE reads EADL File 28 for the characteristic-radiation relaxation
cascade: subshell binding energies, occupancies, and radiative and
nonradiative transition probabilities and energies. xraydb still supplies line
energies where it tabulates the same level pair, natural level widths, and the
optional Elam fluorescence yields. The EADL checksum prefix is part of the
characteristic-model identity marker.
