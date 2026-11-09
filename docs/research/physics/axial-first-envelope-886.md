# First axial run envelope: two calibrated cases, none runnable yet

**No axial case runs yet.** Two silicon cases are calibrated and held in the
repository lookup (`pyrite.quantum._axial_envelope.ADMITTED_AXIAL_CASES`). The standard
runner can launch, score and persist them, but
`RUNNABLE_AXIAL_CASES` is empty, so every axial case still refuses before work or
writes. A case becomes runnable when every claim its run uses is `anchored`. This page
does not describe a supported capability.

## Calibrated cases

Selected with `Physics(channeling_model="axial", channeling_axial={"version": 2,
"axis_uvw": [0, 0, 1]})` and observed with `pyrite.detectors.PointDetector()`. Every numeric comes from the lookup; the version 1 recipe of
#624 stays parseable and refuses at run.

| Key | Case | Depth budget ε (95%) | In-cut free→free block | Evidence |
| --- | --- | --- | --- | --- |
| `si-fibre/A` | Si ⟨100⟩, 3.5 MeV, 0.3 µm, tilt 0.088° | 3.97% | +7.83(8)%, owner: native CBS on zone rows | `depth-calibration-993-A.json`, `free-free-994.json` |
| `si-fibre/B` | Si ⟨100⟩, 3.5 MeV, 0.3 µm, 0° | 9.71% | +18.19(23)%, omitted; declared one-sided bound 18.9% | `depth-calibration-993-B.json`, `free-free-994.json` |

Records are under `docs/validation/check-records/axial-source/`. ε bounds the on-axis
yield integrated over 800 eV–10 keV, not individual bins. Free→free above the
retained cut is not bounded for B. ε, the free→free owner and bound, and the evidence
IDs are stored in the resolved `Case.channeling` record and so persist with results
and checkpoints.

A calibrated case must also be: an electron beam with no angular or energy spread, a
single laterally unbounded slab cut normal to the axis, midpoint transport on the CPU
core, incoherent emission, and on-axis scoring with zero aperture on a line grid inside
the band.
Everything else refuses.

## What still blocks a run

| Requirement | Current state | Owner |
| --- | --- | --- |
| Physical outgoing instrument | `cr-axial-zone-handoff-physical` | Done, #883 |
| Fibre photon source | `cr-axial-fiber-regression`, `anchored` | Done, #992 |
| Measured depth budget per case | `cr-axial-depth-calibration`, `anchored` for A and B | Done, #993 |
| Native launch, scoring and persistence adapter | Implemented (`montecarlo/runner/axial.py`); `cr-axial-persisted-channel` is `unverified` | [#994](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/994) |
| A: owner of the in-cut free→free block | Open: native CBS on A's zone rows gives 2.33e-4 /sr in band against the measured 4.17e-5 /sr, with a different spectral shape | #994 |

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

These consumers refuse a calibrated case by name. Sentinel tests:
`tests/montecarlo/channeling/test_axial_fibre_runner.py`,
`tests/remote/test_axial_admission.py`, `tests/checkpoint/` and
`tests/montecarlo/quantum/test_channeling_axial_preintegration_gates.py`, along with
`tests/montecarlo/quantum/test_channeling_axial_envelope.py`.

## Rule for admitting a case

A case enters the envelope only from a calibration held in the repository, keyed by
its exact resolved case and observable and backed by retained evidence. A
caller-constructed `AxialPhotonDepthEvidence`, a research constructor or a disjoint
radiation-ownership tag cannot grant admission.
