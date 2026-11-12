---
orphan: true
---

# Planar line anchor evidence (#1057)

Harness: `checks/planar_line_anchors.py`; claims `cr-line-shape`,
`cr-yield-assembly`. Comparison record only, no pass/fail gate. Sources and
conditions are in `tests/data/channeling/{genz-1996-diamond-planar,
kephart-1989-si-planar-yields,azadegan-2006-diamond110-widths,
klein-1985-diamond-planar-widths}.json`. Write-up:
[planar line anchors](../../../research/physics/planar-line-anchors-1057.md).

All arrays ran on qlmc partition `cpu` (2 CPUs, 6 GiB per task) from isolated
rsync copies of the branch; every task logged `Exit status: 0`. The revision
in each output is `PYRITE_SOURCE_REVISION` from the submit script.

| Array | Script | Revision | Arms | Content |
|---|---|---|---|---|
| 11693 | `submit.sh` | `b7d61539e` | 30 | Genz (13 cases), Kephart (110)/(100), Azadegan 14.6/17/30/34 MeV at the default grid (8 Bloch cells, `H = 16`, `dz = 100 Å`, 6 depth bins); convergence arms (16 cells, `H = 24`, 50 Å, 12 bins); divergence ×2; no-history-reuse control |
| 11726 | `submit_tds.sh` | `64c1b6a9d` | 3 | TDS kernel ×1.3 (correlation stand-in) on Genz 9/5.2 MeV and Azadegan 14.6/17 MeV |
| 12000 | `submit_klein.sh` | `5f4d071ab` | 3 | Klein (110) 16.9 and 30.5 MeV, 23 µm, plus a 30.5 MeV convergence arm |

`5f4d071ab` is the pre-amend label of `3219648de`; the harness and data are
byte-identical between the two. `report.json` is
`checks/planar_line_anchors.py --report arms-*/arm-*.json` at `3219648de`.
