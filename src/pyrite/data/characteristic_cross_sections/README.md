# Characteristic-radiation cross sections

`EEDL.endf` is the 2025 Livermore Evaluated Electron Data Library (EEDL),
translated from ENDL to ENDF-6 by D. E. Cullen and distributed by the IAEA
Nuclear Data Section. PyRITE reads subshell electroionization cross sections
from ENDF File 23, MT 534--572. The source file identifies itself as
`NDS-IAEA-226`, evaluated August 2023 and distributed January 2025.

The packaged bytes are the EPICS2025 distribution downloaded verbatim from
<https://nuclear.llnl.gov/EPICS/ENDF2025/EEDL2025.ALL>, pinned by SHA-256
`ce37912435e0b8002f85878f98ccf7c5840cb168f1d46af3c9e915cd16c70ccc`.
Upstream records are 75 columns with CRLF line endings. The pin asserts those
published bytes, so `.gitattributes` marks this file `-text` to exempt it from
the repository's `eol=lf` normalization; re-normalizing it would break the pin.
Characteristic line energies, fluorescence yields, and conditional line
intensities are supplied separately by xraydb's Elam tables at runtime. The
resolved xraydb version is included in PyRITE's characteristic-model identity
marker so a relaxation-database update cannot reuse an older checkpoint.
