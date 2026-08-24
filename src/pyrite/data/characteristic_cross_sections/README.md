# Characteristic-radiation cross sections

`EEDL.endf` is the 2025 Livermore Evaluated Electron Data Library (EEDL),
translated from ENDL to ENDF-6 by D. E. Cullen and distributed by the IAEA
Nuclear Data Section. PyRITE reads subshell electroionization cross sections
from ENDF File 23, MT 534--572. The source file identifies itself as
`NDS-IAEA-226`, evaluated August 2023 and distributed January 2025.

The packaged bytes are pinned by SHA-256
`f3ef54f66efaa606a4a5ea7afb3cfe10e35a22b543887dafb3fc7ec830d1769c`.
Characteristic line energies, fluorescence yields, and conditional line
intensities are supplied separately by xraydb's Elam tables at runtime. The
resolved xraydb version is included in PyRITE's characteristic-model identity
marker so a relaxation-database update cannot reuse an older checkpoint.
