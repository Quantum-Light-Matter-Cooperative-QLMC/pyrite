---
orphan: true
---

# Axial photon-source depth calibrations (#993)

**Status:** research evidence for
[`cr-axial-depth-calibration`](../../validation/ledger-channeling.md#cr-axial-depth-calibration).
A budget recorded here is **not** admission. #994 owns the repository lookup and
the tolerance it accepts.

## Cases

All cases are Si, with the fibre source of #992 (one cell, continuum TDS strings,
electronic kicks, ownership boundary $U_\max$, EPDL attenuation). Observable:
on-axis photons/sr per incident electron, integrated over a band, for yield at
crystal thickness.

| Case | Axis | Energy | Thickness | Tilt | Checkpoints (Å) |
|---|---|---|---|---|---|
| A | ⟨100⟩ | 3.5 MeV | 0.3 µm | 0.088° | 1000, 2000, 3000 |
| B | ⟨100⟩ | 3.5 MeV | 0.3 µm | 0° | 1000, 2000, 3000 |
| C | ⟨111⟩ | 2.99 MeV | 1.1 µm | 0° | 3000, 6000, 11000 |

## Arms

Every arm shares the control's seed and 100-primary chunks.

| Probe | Change from control |
|---|---|
| control | window $M=9$, box $B=20$, cut $U_\max+\Delta U$, step 15 Å, kick harmonics 4, density grid 32 |
| return flux | $P=R$ (no cut) |
| unresolved | box $B=28$ |
| matching surface | cut $U_\max+2\Delta U$ |
| grid | $M=11$, step 7.5 Å |
| source model | kick harmonics 6, density grid 48 |
| all-on | all of the above |

Control fates come from the #883 instrument. The matching window is declared at
10 eV: the gap-midpoint cut sits up to 5.4 eV from the target in a smoke run.
All-on fates come from the all-on arm's unit-weight events.

## Results

Pending remote runs.
