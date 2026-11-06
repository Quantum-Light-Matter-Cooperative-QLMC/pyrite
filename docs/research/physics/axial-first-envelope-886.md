# First axial run envelope: currently empty

**No axial case is admitted.** `Physics(channeling_model="axial")` and every raw
axial case refuse in `pyrite._channeling_admission` before work or writes. This page
records what would have to be true to change that. It does not describe a supported
capability.

## Why the envelope is empty

[#886](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/886)
requires every claim an envelope uses to be `anchored`, and requires a measured depth
budget for each admitted case. Neither exists for a material case:

| Requirement | Current state | Owner |
| --- | --- | --- |
| Physical outgoing instrument | `cr-axial-zone-handoff-physical`, for the instrument only | Done, #883 |
| Budget arithmetic and refusal guard | `cr-axial-depth-budget`, with no measured input | #884 |
| Finite-grid source partition and on-axis mapping | `cr-axial-source-partition`, a finite reference only | #885 |
| Validated material photon source | `cr-axial-persisted-channel` is `unverified` | [#992](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/992) |
| Measured depth budget per (material, axis, energy) | None | [#993](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/993) |
| Native launch, scoring and persistence adapters | None | [#994](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/994) |

Check the [ledger](../../validation/ledger-channeling.md) for current claim status.

## Candidate scope

The [integration design](axial-run-integration.md#frames-depth-and-eligibility)
proposes where to look first. This is an engineering restriction, not validated range:

- electrons in a single flat, nonmosaic silicon or diamond slab;
- isotropic sourced displacement data and rectangular projected cells;
- 1–30 MeV, fixed counts and CPU execution.

The existing Si peak-energy evidence covers specific ⟨110⟩ and ⟨111⟩ cases around
2–3 MeV and does not establish general yields. Current Si [001] 5 MeV studies measure
electron fractions, not photon-source error, so they supply no depth budget.

## Explicitly unsupported

These stay refused even after a first case is admitted, until the listed evidence
exists.

| Case | Evidence needed to widen |
| --- | --- |
| Off-axis or azimuth-resolved observation, physical detectors | Photon recoil across Bloch sectors and the full polarization matrix |
| Layered materials, anisotropic axial jumps | Site thermal tensors and layered TDS support (#535) |
| Positrons | A positron potential, its source and handoff validation |
| Coupled radiative loss, secondaries | A joint zone and radiation-loss contract |
| Stacks, grooves, mosaic crystals, GDF inputs | Geometry-resolved source and handoff path mapping |
| Temporal or coherent emission | A multi-electron source contract |
| CUDA continuation | The joint launch contract, covered on both backends |
| Adaptive precision | A per-electron channeling-radiation variance contract |
| Capture and block transport | A source that survives block-local solves |
| Replay, line repair and trajectory scoring | A persisted, reproducible axial source identity |
| Remote submission | The same admission as local runs, before sync or staging |
| W and other diagnostic-only anchors | Their own depth budget and source evidence |

The refusal is unconditional today, so these consumers are covered by the existing
sentinel tests in `tests/remote/test_axial_admission.py`, `tests/checkpoint/` and
`tests/montecarlo/quantum/test_channeling_axial_preintegration_gates.py`, along with
`tests/montecarlo/quantum/test_channeling_axial_envelope.py`.

## Rule for admitting a case

A case enters the envelope only from a calibration held in the repository, keyed by
its exact resolved case and observable and backed by retained evidence. A
caller-constructed `AxialPhotonDepthEvidence`, a research constructor or a disjoint
radiation-ownership tag cannot grant admission.
