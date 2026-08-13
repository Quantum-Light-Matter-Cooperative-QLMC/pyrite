# Detector becomes a scorer

Branch: `refactor/detector-scorer`
Source: [core architecture RFC](../../../../docs/repo-design/core-architecture-rfc.md),
Change 5 — sequencing step 3.
Depends on: `refactor/target-geometry-surface`. Blocks:
`refactor/scene-object-model`.

## Problem and scope

Two disjoint notions share the name "detector":

- `DetectorSpec` (`src/pyrite/detectors/spec.py:54`) supplies
  `observation_angle_deg`, `polar_acceptance_deg`, and `solid_angle_sr` to the
  case. Its remaining fields — `response_model`, `qe_curve`, `pixel_pitch_um`,
  `sensor_thickness_um`, `distance_mm`, `threshold_eV` — are documented as inert
  and are read by nothing.
- The forward models `TimepixResponse` (`timepix_response.py`, 462 lines) and
  `EagleResponse` (`eaglexo_response.py`, 396 lines) are applied after the fact,
  in `store_result` (`src/pyrite/results/store.py:133`) and in the plotting
  layer.

So no object represents a detector end to end, and a run can carry exactly one
observation geometry.

Related: the emission binning, which every comparable code treats as a property
of a tally or monitor, is instead `E_grid_line` / `E_grid_line_by_energy` /
`E_grid_brem` on `Sweep`, per-material entries in the catalog, and a top-level
CLI noun with its own content-addressed artifact store, provenance, garbage
collection, verification, and golden files (`src/pyrite/energy_grid/`, 4 542
lines).

In scope:

- `Detector` owning acceptance + binning + response.
- Inert `DetectorSpec` fields either become live through `response`, or are
  deleted. An inert field in a hashed payload is a liability.
- The photon-energy grid becomes `detector.energy_bins`.

Out of scope, deliberately:

- **Deleting `src/pyrite/energy_grid/`.** The derivation machinery is genuinely
  expensive and its artifacts are worth keeping content-addressed. This task
  *demotes* it from user vocabulary; the artifact store becomes an
  implementation detail of catalog resolution. Moving the `derive` / `verify` /
  `gc` verbs is `refactor/cli-noun-surface`.
- **Multi-detector runs.** Designed for, not delivered here. A single-detector
  run must stay bit-for-bit and the stored record layout for one detector must
  not change. Reuse of one transport pass across several detectors is the payoff
  and is a deliberate second step.
- Any change to the response physics in `TimepixResponse` / `EagleResponse`.

## Target state

```python
detector = pr.Detector(
    observation_angle_deg=90.0,
    polar_acceptance_deg=...,
    solid_angle_sr=...,
    energy_bins=pr.arange_eV(100.0, 5000.0, 3.0),
    response=pr.Timepix3(),          # or None for the intrinsic spectrum
)
```

Today's separate line and bremsstrahlung grids remain, expressed as two binnings
on the detector. The physical reason for the split — fine and narrow for the
coherent lines, coarse and wide for the smooth continuum — is unchanged and
should be documented on the object rather than living as folklore.

## Implementation path

Likely owners:

| Concern | Location |
| --- | --- |
| Acceptance fields | `src/pyrite/detectors/spec.py:54` |
| Response models | `src/pyrite/detectors/timepix_response.py`, `eaglexo_response.py`, `_si_sensor.py` |
| Response application | `src/pyrite/results/store.py:133` (`store_result`), `detected_background:171` |
| Grid fields on the sweep | `src/pyrite/campaign/sweep.py:344` |
| Grid artifacts | `src/pyrite/energy_grid/` |
| Plot-time response | `src/pyrite/plots/` |

## Checklist

- [ ] A — Audit each inert `DetectorSpec` field: which response model would
      consume it, or is it dead? Produce a keep/live/delete disposition per
      field before writing the new class.
- [ ] B — Land `Detector` as a superset of `DetectorSpec`, with `DetectorSpec`
      retained as a deprecated alias under the D7 harness.
- [ ] C — Move `energy_bins` onto `Detector`; keep line and brem binnings as two
      named binnings, documented with their physical rationale.
- [ ] D — Move response application from `store_result` and the plotting layer
      onto the detector object, so `response=None` yields the intrinsic
      spectrum and a response object yields the detected one, by one code path.
- [ ] E — Remove `E_grid_line` / `E_grid_line_by_energy` / `E_grid_brem` from
      the scene objects, resolving them from the detector instead.
- [ ] F — Document the multi-detector seam without implementing it: what a list
      of detectors would need from the transport pass, and what in the stored
      record layout would have to change.
- [ ] G — Update `docs/repo_map.md` and the detector-facing physics pages.

## Decisions and open questions

- **Decided:** single detector first, bit-for-bit. Multi-detector is a separate
  step and must not be smuggled in.
- **Decided:** `energy_grid/` stays; only its user-facing prominence changes,
  and that change belongs to the CLI task.
- **Open:** does an inert field with a plausible future consumer get kept or
  deleted? Default to delete — it can be re-added live. Slice A must justify
  every survivor.
- **Open:** where response application lands. `store_result` applies it today,
  which means the stored record is already detector-convolved in some paths and
  not others. Slice D must first establish which stored arrays are intrinsic
  and which are detected; this is a prerequisite fact, not a design choice.
- **Open:** the catalog carries per-material grid entries. If `energy_bins` is a
  detector field, catalog resolution has to supply it — confirm this against
  the documented resolution order (profile → per-material override → explicit
  argument) in [configuration resolution](../../../../docs/repo-design/configuration-resolution.md)
  before slice C.

## Delegation slices and required skills

- A → `implement-task`; `scientific-library`. Audit only, no behavior change.
  Not `one-shot` — its output is a decision.
- B, C → `implement-task`; `scientific-library`.
- D → `lead-task`; touches the intrinsic-versus-detected boundary that
  `store_result` and the plots currently share. Highest-risk slice here.
- E → `implement-task`; `catalog-golden`.
- F → `implement-task-lite`; `documentation-maintenance`.
- G → `implement-task-lite`; `documentation-maintenance` + `repo-orientation`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
```

- A single-detector run reproduces its stored spectrum bit for bit.
- No inert field survives on the public detector object.
- `E_grid_line` / `E_grid_brem` are absent from the scene objects.
- The stored record layout for a one-detector run is unchanged.

## Related

- `refactor/target-geometry-surface` — RFC sequencing prerequisite.
- `refactor/cli-noun-surface` — owns the `energy-grid` verb relocation.
- TODO P2 "Detector profiles and Zhai validation modernization"
  (`feature/profile-observation-angle`) adds profile-owned detector geometry
  with a 90-degree default. It overlaps this task's acceptance surface directly.
  **Sequence them deliberately**; do not run both against `DetectorSpec` at once.
- TODO "Long-term plans → Custom detector tooling" is the eventual consumer of
  a live `response` object.
