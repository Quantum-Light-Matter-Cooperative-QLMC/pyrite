# Electron transport correctness

## Problem and scope

The August 2026 transport review identifies three definite correctness defects
in the current 1--300 keV electron path:

1. coherent CXR uses the segment midpoint position with the segment-start
   transport age;
2. a flight that crosses `E_cut_keV` is retained at its full sampled or
   boundary-limited length; and
3. electrons that exhaust `max_steps` are included in `n_stopped` rather than
   reported as computationally incomplete.

Source: [`docs/electron_transport_physics_recommendations.docx`](../../../../docs/electron_transport_physics_recommendations.docx), Stage 0.

This task owns the correctness fixes and their migration/validation surface.
It does not introduce energy-loss substeps or replace elastic/stopping models.
Those are separate dependent tasks.

The review referred to `transport(4).py`; the current owner is
`src/cxr_mc/montecarlo/transport.py`. The same algorithm now exists in four
paths, all in scope:

- lockstep ungrooved CPU: `_transport_core_ungrooved`;
- grooved/layered CPU: `_transport_core_grooved`;
- per-electron CPU: `_transport_core_ungrooved_perelectron`; and
- CUDA: `montecarlo/transport_jit_kernel.py`.

`montecarlo/spectrum.py::mc_spectrum` and `::mc_brem_spectrum` consume the
segment schema. `simulate_trajectories` owns validation and termination
diagnostics.

## Implementation path and likely owners

- Pair coherent midpoint position with midpoint transport age. Preserve
  `t_ang` as segment-start age for compatibility unless the migration decision
  below explicitly changes the schema; the smallest correction can derive
  midpoint age from `t_ang`, `L_ang`, and the current constant segment beta.
- Truncate a cutoff-crossing material flight at the first point where the
  current stopping rule reaches `E_cut_keV`, then recompute midpoint, length,
  endpoint energy, and clock advance consistently. Do this before recording in
  every transport core, including grooved and CUDA paths.
- Distinguish physical cutoff stops from step-budget exhaustion. Raise on
  incomplete histories by default and provide an explicit, visible permissive
  path only if a caller needs partial trajectories.
- Validate finite positive initial/cutoff energies with
  `0 < E_cut_by_electrons[i] < initial_E_keV[i]`, reject non-finite/zero beam
  directions before normalization, and confirm existing layer-continuity
  validation covers the review's requirement.
- Update the `electron-transport` and coherent-emission validation records.
  The existing coherent validation rederives the phase convention but does not
  test that position and time refer to the same point on a physical flight.

Likely tests: `tests/montecarlo/test_montecarlo.py`,
`test_transport_per_electron.py`, `test_transport_core_default.py`,
`test_groove.py`, and `test_coherent_emission.py`. CUDA parity requires the
remote GPU workflow; do not run a heavy/GPU sweep locally.

## Checklist

- [ ] A -- Specify segment-time compatibility and incomplete-history API;
      identify every result/count consumer before changing keys or exceptions.
- [ ] B -- Add the constant-velocity one-flight versus two-subsegment coherent
      field invariance regression, covering both coherent reduction routes.
- [ ] C -- Correct coherent midpoint time in all spectrum paths and add a new
      `Validation: <id>` marker, ledger row, assumptions, and limiting case.
- [ ] D -- Add a constructed cutoff-crossing regression; truncate the terminal
      flight consistently in lockstep, grooved, per-electron, and CUDA cores.
- [ ] E -- Add explicit cutoff-stop and step-limited termination states/counts;
      raise by default when any history is incomplete.
- [ ] F -- Add the review's input validation without changing valid runs.
- [ ] G -- Prove CPU-core parity with focused statistical/deterministic tests;
      verify CUDA behavior on the configured remote GPU.
- [ ] H -- Update physics documentation/ledger and regenerate affected golden
      data only after the implementation and fresh-context review agree.

## Decisions and open questions

- **Decided:** a physical segment's stored representative position and time
  must describe the same point.
- **Decided:** no below-cutoff portion may contribute to line or bremsstrahlung
  radiation.
- **Decided:** `max_steps` exhaustion is not a physical stop and must be visible.
- **Open:** preserve `t_ang` as start age and derive midpoint time downstream,
  or add `t_mid_ang` now. Prefer the additive field if it reduces repeated
  downstream arithmetic without forcing the Stage 1 schema prematurely.
- **Open:** exact exception/config shape for explicitly permitted incomplete
  histories. Resolve after inventorying internal and app callers.
- **Open:** whether cutoff truncation uses the present left-endpoint stopping
  rule exactly or a small local solve. Stage 0 must not silently implement the
  broader midpoint/substep model owned by the controlled-propagation task.

## Delegation slices and required skills

- A is `lead-task` + `repo-orientation`; not Serena `one-shot` because it sets
  public/internal result semantics.
- B--C are `implement-task` + `monte-carlo` + `regression-testing` +
  `physics-review`; not `one-shot` until the time-schema decision is closed.
- D--F are `implement-task` + `monte-carlo` + `regression-testing`; D is not
  `one-shot` because four execution cores must remain equivalent.
- G uses `remote-gpu-jobs`, `performance`, and `run-cxr-mc`; no local GPU sweep.
- H uses `physics-validation` with fresh context, `regen-golden`, and
  `documentation-maintenance`. Only a human may mark physics `signed-off`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_coherent_emission.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_montecarlo.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_transport_per_electron.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test tests/montecarlo/test_groove.py
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite core
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
```

- One physical constant-velocity flight and two numerical halves yield the
  same complex coherent field and spectrum within the stated numerical bound.
- A constructed cutoff crossing retains exactly the above-cutoff path; line and
  bremsstrahlung consumers cannot recover a below-cutoff contribution.
- Forced `max_steps` exhaustion raises by default; permissive mode, if retained,
  reports `n_step_limited` separately and never increments `n_stopped`.
- All valid historical CPU cases remain deterministic under their pinned core;
  CUDA and CPU per-electron implementations preserve their documented
  distribution/parity contract.
- New/changed physics has source equation, assumptions, limiting case,
  `Validation: <id>`, ledger coverage, and fresh-context validation.
