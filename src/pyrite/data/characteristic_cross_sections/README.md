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
Characteristic line energies, fluorescence yields, conditional line
intensities, and natural atomic-level widths are supplied separately by
xraydb at runtime. PyRITE sums the initial- and final-level widths to obtain
each transition's Lorentzian FWHM. The resolved xraydb version is included in
PyRITE's characteristic-model identity marker so a relaxation-database update
cannot reuse an older checkpoint.
