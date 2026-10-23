# NIST ESTAR electron stopping and range tables

Text output of [NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html)
(NIST Standard Reference Database 124; ICRU Report 37 methods), downloaded on
2026-10-22. Each file has the full default table: seven space-delimited columns
on the default 10 keV–1 GeV grid.

| Column | Quantity | Unit |
| --- | --- | --- |
| 1 | Kinetic energy | MeV |
| 2 | Collision stopping power | MeV cm²/g |
| 3 | Radiative stopping power | MeV cm²/g |
| 4 | Total stopping power | MeV cm²/g |
| 5 | CSDA range | g/cm² |
| 6 | Radiation yield | — |
| 7 | Density-effect parameter δ | — |

Files are byte-identical to the downloads; consumers pin their SHA-256. Do not
reformat them.

| File | ESTAR material | Catalog materials containing it |
| --- | --- | --- |
| `silicon.txt` | Silicon | silicon, 4H/6H-SiC |
| `tungsten.txt` | Tungsten | tungsten, WS₂, WSe₂, WTe₂ |
| `carbon-amorphous.txt` | Amorphous carbon (2.0 g/cm³) | diamond stand-in |
| `graphite.txt` | Carbon (graphite, 1.7 g/cm³) | HOPG |
| `silicon-dioxide.txt` | Silicon dioxide | SiO₂ medium |
| `boron.txt`, `nitrogen.txt` | Boron, nitrogen | hBN |
| `molybdenum.txt` | Molybdenum | MoS₂, MoSe₂, MoTe₂ |
| `niobium.txt` | Niobium | NbS₂, NbSe₂, NbTe₂ |
| `hafnium.txt` | Hafnium | HfS₂, HfSe₂, HfTe₂ |
| `sulfur.txt`, `selenium.txt`, `tellurium.txt` | Chalcogens | transition-metal dichalcogenides |

ESTAR uses its own density and mean excitation energy per material, and
compounds built from elemental files by Bragg additivity ignore chemical-binding
and condensed-phase density-effect differences. Treat these files as independent
reference values, never as PyRITE inputs.

Consumers: the #510 benchmark matrix (`checks/equal_precision/matrix-spec.json`)
takes CSDA ranges for silicon, tungsten, diamond and SiO₂ from here.
`tests/montecarlo/test_hard_radiative.py` pins graphite, silicon, molybdenum
and tungsten collision, radiative and total stopping to 100 MeV (#439).
