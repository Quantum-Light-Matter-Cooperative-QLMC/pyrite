# First axial run envelope: two admitted silicon cases

Two silicon cases run on the standard CPU runner. They are calibrated and held in the
repository lookup (`pyrite.quantum._axial_envelope.ADMITTED_AXIAL_CASES`) and marked
runnable (`RUNNABLE_AXIAL_CASES`) because every claim their run uses is `anchored`.
Every other axial case refuses before work or writes. The selector and
`PointDetector` are experimental.

## Admitted cases

Selected with `channeling=pyrite.quantum.Channeling("axial", axial={"version": 2,
"axis_uvw": [0, 0, 1]})` (before #1017: `Physics(channeling_model="axial", ...)`)
and observed with `pyrite.detectors.PointDetector()` (profile: `detector = "point"`). Every numeric comes from the lookup; the version 1 recipe of
#624 stays parseable and refuses at run.

| Key | Case | Depth budget ε (95%) | In-cut free→free block | Evidence |
| --- | --- | --- | --- | --- |
| `si-fibre/A` | Si ⟨100⟩, 3.5 MeV, 0.3 µm, tilt 0.088° | 3.97% | +7.83(8)%, omitted; declared one-sided bound 8.06% | `depth-calibration-993-A.json`, `free-free-994.json` |
| `si-fibre/B` | Si ⟨100⟩, 3.5 MeV, 0.3 µm, 0° | 9.71% | +18.19(23)%, omitted; declared one-sided bound 18.9% | `depth-calibration-993-B.json`, `free-free-994.json` |

Records are under `docs/validation/check-records/axial-source/`. ε bounds the on-axis
yield integrated over 800 eV–10 keV, not individual bins. Free→free above the
retained cut is not bounded for either case. Zone rows score no native PXR/CBS lines
(`cr-axial-zone-row-native-lines`), so the block is never counted twice. ε, the free→free owner and bound, and the evidence
IDs are stored in the resolved `Case.channeling` record and so persist with results
and checkpoints.

An admitted case must also be: an electron beam with no angular or energy spread, a
single laterally unbounded slab cut normal to the axis, midpoint transport on the CPU
core, incoherent emission, and on-axis scoring with zero aperture on a line grid inside
the band.
Everything else refuses.

## Running an admitted case from a profile

A profile, `pyrite run` and `pyrite run --remote` reach the same resolved case
as the Python API (#624). The profile names the packaged `point` detector
(`kind = "point"`, lowered to `PointDetector`) and a named target that gives
the cut and the unbounded footprint:

```toml
# targets/si_fibre_b.toml
kind = "slab"
material = "silicon"
thickness_ang = 3000.0
tilt_deg = 0.0
tilt_azim_deg = 0.0
allow_normal_incidence = true    # si-fibre/B only; A uses tilt 0.088, azimuth 225
footprint = "unbounded"
cut = {surface_hkl = [0, 0, 1]}

# profiles/axial_fibre_b.toml (scan grid keys are required but the target owns geometry)
thickness_ang = {values = [3000.0]}
energy_keV = {values = [3500.0]}
tilt_deg = {values = [5.0]}
tilt_azim_deg = {values = [0.0]}
materials = ["silicon"]
cxr_histories = {values = [4]}
brem_histories = {values = [6]}
beam = "example"
detector = "point"
emission = "incoherent"
energy_model = "midpoint"
radiative_model = "uncoupled"
E_grid_line = {linspace = {start = 800.0, stop = 10000.0, num = 47}}

[targets]
silicon = "si_fibre_b"

[channeling]
model = "axial"
axial = {version = 2, axis_uvw = [0, 0, 1]}
```

The lowered case equals the API case except for `name`, which gains the
campaign `ne=<line>/<brem>` suffix; `name` is not part of `case_content_key`.
Leave the continuum grid undeclared on both sides (or declare the same band:
a profile-declared brem grid extends to E0). Under the same seed the persisted
checkpoint equals the API run (`tests/campaign/test_axial_profile_parity.py`).
Human `pyrite run` output warns on stderr that the axial fibre selector is
experimental; `--output json` stays one parseable value. Remote submission
lowers the run's cases and applies the same admission before host probes,
synchronization or staging; the remote host's selected catalog must contain
the profile and target.

A point detector carries no acceptance fields, so a profile cannot reach zero
aperture through `polar_acceptance_deg`/`solid_angle_sr`; `dtheta_obs_deg`
overrides on a point profile refuse. Off-axis point detectors, finite
apertures, finite footprints and cases outside the lookup still refuse before
work or writes.

## Claims an admitted run uses

| Requirement | Claim and status | Owner |
| --- | --- | --- |
| Physical outgoing instrument | `cr-axial-zone-handoff-physical` | Done, #883 |
| Fibre photon source | `cr-axial-fiber-regression`, `anchored` | Done, #992 |
| Measured depth budget per case | `cr-axial-depth-calibration`, `anchored` for A and B | Done, #993 |
| Native launch, scoring and persistence adapter | `cr-axial-persisted-channel`, `anchored` for A and B | Done, [#994](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/994) |
| Zone rows score no native lines; A and B carry the in-cut free→free block as a declared bound | `cr-axial-zone-row-native-lines`, `anchored` | Done, #994 |

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
| W and other diagnostic-only anchors | Their own depth budget and source evidence |

These consumers refuse an admitted case by name. Sentinel tests:
`tests/montecarlo/channeling/test_axial_fibre_runner.py`,
`tests/checkpoint/` and
`tests/montecarlo/quantum/test_channeling_axial_preintegration_gates.py`, along with
`tests/montecarlo/quantum/test_channeling_axial_envelope.py`.

## Rule for admitting a case

A case enters the envelope only from a calibration held in the repository, keyed by
its exact resolved case and observable and backed by retained evidence. A
caller-constructed `AxialPhotonDepthEvidence`, a research constructor or a disjoint
radiation-ownership tag cannot grant admission.
