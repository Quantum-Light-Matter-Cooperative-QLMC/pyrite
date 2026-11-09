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
| A | ⟨100⟩ | 3.5 MeV | 0.3 µm | 0.088° | 750, 1500, 3000 |
| B | ⟨100⟩ | 3.5 MeV | 0.3 µm | 0° | 750, 1500, 3000 |
| C | ⟨111⟩ | 2.99 MeV | 1.0995 µm (733 × 15 Å) | 0° | 3000, 6000, 10995 |

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

Remote runs on qlmc at 07c0f5f3e (job arrays 6147–6170; drivers
`checks/axial_depth_calibration_993{.sbatch,.sh}`; every arm's settings are in the record). Band 800 eV to
$E_\max$ (10 keV for A and B, 8 keV for C). Joint confidence 0.95. Records:
`docs/validation/check-records/axial-source/depth-calibration-993-{A,B,C}.json`.

Bound $\varepsilon$ on the relative photon-source error, per interval
$(c-15\,\text{Å}, c]$. Components (photons/sr per incident) are at the last
checkpoint (3000 Å for A and B, 10995 Å for C).

| Case | ε at 750 Å | ε at 1500 Å | ε at 3000 Å | Return | Unresolved | Grid | Source model | Reference LCB |
|---|---|---|---|---|---|---|---|---|
| A | 5.35% | 4.50% | 3.97% | 4.9e-6 | 1.15e-5 | 2.2e-6 | 2.2e-6 | 5.24e-4 |
| B | 12.2% | 10.8% | 9.71% | 4.8e-6 | 4.6e-6 | 2.1e-6 | 2.8e-6 | 1.47e-4 |

| Case | ε at 3000 Å | ε at 6000 Å | ε at 10995 Å | Return | Unresolved | Grid | Source model | Reference LCB |
|---|---|---|---|---|---|---|---|---|
| C | 65.7% | 54.3% | 49.0% | 5.7e-5 | 8.2e-5 | 2.0e-5 | 1.6e-5 | 3.57e-4 |

**A and B.** Every paired probe difference is within 1.6 SE of zero; the all-on interaction
bound is about 2e-6 in both cases. The budget is therefore resolution-limited:
it is set by $t$ = 3.42 times the paired SEs and by the fate terms, not by any
resolved model change. Unresolved is the largest term in A, at
$C_u F_u$ with $C_u = 6.6\times10^{-4}$ and $F_u^\text{ctl} = 0.87\%$.
B carries about the same absolute bounds on a yield 3.5 times smaller.
Doubling ratios are 0.92–1.03. More primaries would shrink every
term except the fate cap terms.

**C: negative result at the control settings.** The fate terms dominate: by
10995 Å the control retains 42% of electrons, with 51% outgoing and 7.9%
unresolved ($C_u = 9.2\times10^{-4}$, $C_o = 9.3\times10^{-5}$). Unlike A
and B, C resolves model changes at 10995 Å: grid $+1.8\%$ (4.5 SE), return
$+1.4\%$ (3.6 SE), matching $+1.0\%$ (2.8 SE), interaction bound 3.5%. So
$M=9$ is under-converged here, and the fate caps turn the resulting window
leakage into a bound of about 50%. Doubling ratios are 0.99–1.01. A usable
C budget needs a larger retained window and box ($M \ge 11$, larger $B$) as
the control, not more primaries.

The merged C control's Watson & Koehler peak comparison (200 eV FWHM) finds
2.95 and 4.15 keV against measured 3.36 and 4.9 keV (−12%, −15%), and no
peak near 1.63 keV. The `nanvar` warnings during the C merge come from that
missing peak (all bootstrap draws NaN), not from the calibration inputs.
